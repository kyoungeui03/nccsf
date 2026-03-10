#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "RuntimeContext.h"
#include "commons/Data.h"
#include "forest/Forest.h"
#include "forest/ForestOptions.h"
#include "forest/ForestPredictors.h"
#include "forest/ForestTrainers.h"
#include "prediction/Prediction.h"

namespace {

thread_local std::string g_last_error;

void clear_error() {
  g_last_error.clear();
}

void set_error(const std::exception& error) {
  g_last_error = error.what();
}

void set_error(const char* error) {
  g_last_error = error;
}

std::vector<double> row_major_to_col_major(const double* values,
                                           std::size_t num_rows,
                                           std::size_t num_cols) {
  std::vector<double> result(num_rows * num_cols);
  for (std::size_t row = 0; row < num_rows; ++row) {
    for (std::size_t col = 0; col < num_cols; ++col) {
      result[col * num_rows + row] = values[row * num_cols + col];
    }
  }
  return result;
}

std::vector<double> append_columns(const std::vector<double>& x_col_major,
                                   std::size_t num_rows,
                                   const std::vector<const std::vector<double>*>& extra_columns) {
  std::size_t num_x_cols = x_col_major.size() / num_rows;
  std::size_t total_cols = num_x_cols + extra_columns.size();
  std::vector<double> result(num_rows * total_cols);

  std::copy(x_col_major.begin(), x_col_major.end(), result.begin());

  std::size_t offset = num_x_cols;
  for (const std::vector<double>* column : extra_columns) {
    if (column == nullptr || column->size() != num_rows) {
      throw std::runtime_error("Invalid extra column while constructing GRF training matrix.");
    }
    std::copy(column->begin(), column->end(), result.begin() + offset * num_rows);
    ++offset;
  }

  return result;
}

std::vector<double> combine_with_train_extras(const double* x_row_major,
                                              std::size_t num_rows,
                                              std::size_t num_cols,
                                              const std::vector<const std::vector<double>*>& extra_columns) {
  return append_columns(row_major_to_col_major(x_row_major, num_rows, num_cols), num_rows, extra_columns);
}

std::vector<double> flatten_predictions(const std::vector<grf::Prediction>& predictions) {
  if (predictions.empty()) {
    return {};
  }

  std::size_t prediction_length = predictions.front().size();
  std::vector<double> flat(predictions.size() * prediction_length);

  for (std::size_t row = 0; row < predictions.size(); ++row) {
    const std::vector<double>& current = predictions[row].get_predictions();
    for (std::size_t col = 0; col < prediction_length; ++col) {
      flat[row * prediction_length + col] = current[col];
    }
  }

  return flat;
}

void copy_predictions_to_buffer(const std::vector<grf::Prediction>& predictions, double* output) {
  std::vector<double> flat = flatten_predictions(predictions);
  if (!flat.empty()) {
    std::copy(flat.begin(), flat.end(), output);
  }
}

grf::ForestOptions make_options(unsigned int num_trees,
                                std::size_t ci_group_size,
                                double sample_fraction,
                                unsigned int mtry,
                                unsigned int min_node_size,
                                bool honesty,
                                double honesty_fraction,
                                bool honesty_prune_leaves,
                                double alpha,
                                double imbalance_penalty,
                                unsigned int num_threads,
                                unsigned int seed) {
  std::vector<std::size_t> clusters;
  unsigned int samples_per_cluster = 0;
  bool legacy_seed = false;
  return grf::ForestOptions(
      num_trees,
      ci_group_size,
      sample_fraction,
      mtry,
      min_node_size,
      honesty,
      honesty_fraction,
      honesty_prune_leaves,
      alpha,
      imbalance_penalty,
      num_threads,
      seed,
      legacy_seed,
      clusters,
      samples_per_cluster);
}

struct RegressionForestHandle {
  std::size_t num_rows;
  std::size_t num_features;
  bool use_sample_weights;
  std::size_t outcome_index;
  std::size_t sample_weight_index;
  std::vector<double> y;
  std::vector<double> sample_weights;
  std::vector<double> train_matrix;
  std::unique_ptr<grf::Forest> forest;

  RegressionForestHandle(std::size_t num_rows,
                         std::size_t num_features,
                         std::vector<double> y,
                         std::vector<double> sample_weights,
                         std::vector<double> train_matrix,
                         grf::Forest&& forest)
      : num_rows(num_rows),
        num_features(num_features),
        use_sample_weights(!sample_weights.empty()),
        outcome_index(num_features),
        sample_weight_index(num_features + 1),
        y(std::move(y)),
        sample_weights(std::move(sample_weights)),
        train_matrix(std::move(train_matrix)),
        forest(std::make_unique<grf::Forest>(std::move(forest))) {}
};

struct SurvivalForestHandle {
  std::size_t num_rows;
  std::size_t num_features;
  std::size_t num_failures;
  int prediction_type;
  bool use_sample_weights;
  std::size_t outcome_index;
  std::size_t censor_index;
  std::size_t sample_weight_index;
  std::vector<double> y_relabeled;
  std::vector<double> censor;
  std::vector<double> sample_weights;
  std::vector<double> train_matrix;
  std::unique_ptr<grf::Forest> forest;

  SurvivalForestHandle(std::size_t num_rows,
                       std::size_t num_features,
                       std::size_t num_failures,
                       int prediction_type,
                       std::vector<double> y_relabeled,
                       std::vector<double> censor,
                       std::vector<double> sample_weights,
                       std::vector<double> train_matrix,
                       grf::Forest&& forest)
      : num_rows(num_rows),
        num_features(num_features),
        num_failures(num_failures),
        prediction_type(prediction_type),
        use_sample_weights(!sample_weights.empty()),
        outcome_index(num_features),
        censor_index(num_features + 1),
        sample_weight_index(num_features + 2),
        y_relabeled(std::move(y_relabeled)),
        censor(std::move(censor)),
        sample_weights(std::move(sample_weights)),
        train_matrix(std::move(train_matrix)),
        forest(std::make_unique<grf::Forest>(std::move(forest))) {}
};

struct CausalSurvivalForestHandle {
  std::size_t num_rows;
  std::size_t num_features;
  std::vector<double> train_covariates;
  std::unique_ptr<grf::Forest> forest;

  CausalSurvivalForestHandle(std::size_t num_rows,
                             std::size_t num_features,
                             std::vector<double> train_covariates,
                             grf::Forest&& forest)
      : num_rows(num_rows),
        num_features(num_features),
        train_covariates(std::move(train_covariates)),
        forest(std::make_unique<grf::Forest>(std::move(forest))) {}
};

}  // namespace

extern "C" {

const char* csf_grf_last_error_message() {
  return g_last_error.c_str();
}

void* csf_grf_regression_fit(const double* x_row_major,
                             const double* y_values,
                             const double* sample_weight_values,
                             std::size_t num_rows,
                             std::size_t num_features,
                             unsigned int mtry,
                             unsigned int num_trees,
                             unsigned int min_node_size,
                             double sample_fraction,
                             bool honesty,
                             double honesty_fraction,
                             bool honesty_prune_leaves,
                             std::size_t ci_group_size,
                             double alpha,
                             double imbalance_penalty,
                             bool compute_oob_predictions,
                             unsigned int num_threads,
                             unsigned int seed) {
  clear_error();
  try {
    grf::runtime_context.verbose_stream = nullptr;
    grf::runtime_context.forest_name = "regression";

    std::vector<double> x_col_major = row_major_to_col_major(x_row_major, num_rows, num_features);
    std::vector<double> y(y_values, y_values + num_rows);
    std::vector<double> sample_weights;
    if (sample_weight_values != nullptr) {
      sample_weights.assign(sample_weight_values, sample_weight_values + num_rows);
    }

    std::vector<const std::vector<double>*> extra_columns = {&y};
    if (!sample_weights.empty()) {
      extra_columns.push_back(&sample_weights);
    }
    std::vector<double> train_matrix = append_columns(x_col_major, num_rows, extra_columns);

    grf::Data data(train_matrix.data(), num_rows, num_features + extra_columns.size());
    data.set_outcome_index(num_features);
    if (!sample_weights.empty()) {
      data.set_weight_index(num_features + 1);
    }

    grf::ForestTrainer trainer = grf::regression_trainer();
    grf::ForestOptions options = make_options(
        num_trees,
        ci_group_size,
        sample_fraction,
        mtry,
        min_node_size,
        honesty,
        honesty_fraction,
        honesty_prune_leaves,
        alpha,
        imbalance_penalty,
        num_threads,
        seed);
    grf::Forest forest = trainer.train(data, options);

    if (compute_oob_predictions) {
      grf::ForestPredictor predictor = grf::regression_predictor(num_threads);
      predictor.predict_oob(forest, data, false);
    }

    return new RegressionForestHandle(
        num_rows,
        num_features,
        std::move(y),
        std::move(sample_weights),
        std::move(train_matrix),
        std::move(forest));
  } catch (const std::exception& error) {
    set_error(error);
    return nullptr;
  }
}

int csf_grf_regression_predict(void* handle_ptr,
                               const double* x_row_major,
                               std::size_t num_rows,
                               std::size_t num_features,
                               bool oob_prediction,
                               unsigned int num_threads,
                               double* output) {
  clear_error();
  try {
    auto* handle = static_cast<RegressionForestHandle*>(handle_ptr);
    if (handle == nullptr) {
      throw std::runtime_error("Regression forest handle is null.");
    }

    grf::ForestPredictor predictor = grf::regression_predictor(num_threads);
    grf::Data train_data(handle->train_matrix.data(), handle->num_rows, handle->num_features + 1 + (handle->use_sample_weights ? 1 : 0));
    train_data.set_outcome_index(handle->outcome_index);
    if (handle->use_sample_weights) {
      train_data.set_weight_index(handle->sample_weight_index);
    }

    std::vector<grf::Prediction> predictions;
    if (oob_prediction) {
      predictions = predictor.predict_oob(*handle->forest, train_data, false);
    } else {
      std::vector<double> x_col_major = row_major_to_col_major(x_row_major, num_rows, num_features);
      grf::Data test_data(x_col_major.data(), num_rows, num_features);
      predictions = predictor.predict(*handle->forest, train_data, test_data, false);
    }

    copy_predictions_to_buffer(predictions, output);
    return 0;
  } catch (const std::exception& error) {
    set_error(error);
    return 1;
  }
}

void csf_grf_regression_free(void* handle_ptr) {
  delete static_cast<RegressionForestHandle*>(handle_ptr);
}

void* csf_grf_survival_fit(const double* x_row_major,
                           const double* y_relabeled_values,
                           const double* censor_values,
                           const double* sample_weight_values,
                           std::size_t num_rows,
                           std::size_t num_features,
                           unsigned int mtry,
                           unsigned int num_trees,
                           unsigned int min_node_size,
                           double sample_fraction,
                           bool honesty,
                           double honesty_fraction,
                           bool honesty_prune_leaves,
                           double alpha,
                           std::size_t num_failures,
                           int prediction_type,
                           bool fast_logrank,
                           bool compute_oob_predictions,
                           unsigned int num_threads,
                           unsigned int seed) {
  clear_error();
  try {
    grf::runtime_context.verbose_stream = nullptr;
    grf::runtime_context.forest_name = "survival";

    std::vector<double> x_col_major = row_major_to_col_major(x_row_major, num_rows, num_features);
    std::vector<double> y_relabeled(y_relabeled_values, y_relabeled_values + num_rows);
    std::vector<double> censor(censor_values, censor_values + num_rows);
    std::vector<double> sample_weights;
    if (sample_weight_values != nullptr) {
      sample_weights.assign(sample_weight_values, sample_weight_values + num_rows);
    }

    std::vector<const std::vector<double>*> extra_columns = {&y_relabeled, &censor};
    if (!sample_weights.empty()) {
      extra_columns.push_back(&sample_weights);
    }
    std::vector<double> train_matrix = append_columns(x_col_major, num_rows, extra_columns);

    grf::Data data(train_matrix.data(), num_rows, num_features + extra_columns.size());
    data.set_outcome_index(num_features);
    data.set_censor_index(num_features + 1);
    if (!sample_weights.empty()) {
      data.set_weight_index(num_features + 2);
    }

    grf::ForestTrainer trainer = grf::survival_trainer(fast_logrank);
    grf::ForestOptions options = make_options(
        num_trees,
        1,
        sample_fraction,
        mtry,
        min_node_size,
        honesty,
        honesty_fraction,
        honesty_prune_leaves,
        alpha,
        0.0,
        num_threads,
        seed);
    grf::Forest forest = trainer.train(data, options);

    if (compute_oob_predictions) {
      grf::ForestPredictor predictor = grf::survival_predictor(num_threads, num_failures, prediction_type);
      predictor.predict_oob(forest, data, false);
    }

    return new SurvivalForestHandle(
        num_rows,
        num_features,
        num_failures,
        prediction_type,
        std::move(y_relabeled),
        std::move(censor),
        std::move(sample_weights),
        std::move(train_matrix),
        std::move(forest));
  } catch (const std::exception& error) {
    set_error(error);
    return nullptr;
  }
}

int csf_grf_survival_predict(void* handle_ptr,
                             const double* x_row_major,
                             std::size_t num_rows,
                             std::size_t num_features,
                             bool oob_prediction,
                             unsigned int num_threads,
                             double* output) {
  clear_error();
  try {
    auto* handle = static_cast<SurvivalForestHandle*>(handle_ptr);
    if (handle == nullptr) {
      throw std::runtime_error("Survival forest handle is null.");
    }

    grf::ForestPredictor predictor = grf::survival_predictor(num_threads, handle->num_failures, handle->prediction_type);
    grf::Data train_data(handle->train_matrix.data(), handle->num_rows, handle->num_features + 2 + (handle->use_sample_weights ? 1 : 0));
    train_data.set_outcome_index(handle->outcome_index);
    train_data.set_censor_index(handle->censor_index);
    if (handle->use_sample_weights) {
      train_data.set_weight_index(handle->sample_weight_index);
    }

    std::vector<grf::Prediction> predictions;
    if (oob_prediction) {
      std::vector<double> maybe_modified;
      const double* data_ptr = handle->train_matrix.data();
      if (x_row_major != nullptr) {
        maybe_modified = combine_with_train_extras(
            x_row_major,
            num_rows,
            num_features,
            handle->use_sample_weights
                ? std::vector<const std::vector<double>*>{&handle->y_relabeled, &handle->censor, &handle->sample_weights}
                : std::vector<const std::vector<double>*>{&handle->y_relabeled, &handle->censor});
        data_ptr = maybe_modified.data();
      }
      grf::Data oob_data(data_ptr, num_rows, num_features + 2 + (handle->use_sample_weights ? 1 : 0));
      oob_data.set_outcome_index(handle->outcome_index);
      oob_data.set_censor_index(handle->censor_index);
      if (handle->use_sample_weights) {
        oob_data.set_weight_index(handle->sample_weight_index);
      }
      predictions = predictor.predict_oob(*handle->forest, oob_data, false);
    } else {
      std::vector<double> x_col_major = row_major_to_col_major(x_row_major, num_rows, num_features);
      grf::Data test_data(x_col_major.data(), num_rows, num_features);
      predictions = predictor.predict(*handle->forest, train_data, test_data, false);
    }

    copy_predictions_to_buffer(predictions, output);
    return 0;
  } catch (const std::exception& error) {
    set_error(error);
    return 1;
  }
}

std::size_t csf_grf_survival_num_failures(void* handle_ptr) {
  auto* handle = static_cast<SurvivalForestHandle*>(handle_ptr);
  return handle == nullptr ? 0 : handle->num_failures;
}

void csf_grf_survival_free(void* handle_ptr) {
  delete static_cast<SurvivalForestHandle*>(handle_ptr);
}

void* csf_grf_causal_survival_fit(const double* x_row_major,
                                  const double* treatment_values,
                                  const double* numerator_values,
                                  const double* denominator_values,
                                  const double* censor_values,
                                  const double* sample_weight_values,
                                  std::size_t num_rows,
                                  std::size_t num_features,
                                  unsigned int mtry,
                                  unsigned int num_trees,
                                  unsigned int min_node_size,
                                  double sample_fraction,
                                  bool honesty,
                                  double honesty_fraction,
                                  bool honesty_prune_leaves,
                                  std::size_t ci_group_size,
                                  double alpha,
                                  double imbalance_penalty,
                                  bool stabilize_splits,
                                  unsigned int num_threads,
                                  unsigned int seed) {
  clear_error();
  try {
    grf::runtime_context.verbose_stream = nullptr;
    grf::runtime_context.forest_name = "causal survival";

    std::vector<double> x_col_major = row_major_to_col_major(x_row_major, num_rows, num_features);
    std::vector<double> treatment(treatment_values, treatment_values + num_rows);
    std::vector<double> numerator(numerator_values, numerator_values + num_rows);
    std::vector<double> denominator(denominator_values, denominator_values + num_rows);
    std::vector<double> censor(censor_values, censor_values + num_rows);
    std::vector<double> sample_weights;
    if (sample_weight_values != nullptr) {
      sample_weights.assign(sample_weight_values, sample_weight_values + num_rows);
    }

    std::vector<const std::vector<double>*> extra_columns = {&treatment, &numerator, &denominator, &censor};
    if (!sample_weights.empty()) {
      extra_columns.push_back(&sample_weights);
    }
    std::vector<double> train_matrix = append_columns(x_col_major, num_rows, extra_columns);

    grf::Data data(train_matrix.data(), num_rows, num_features + extra_columns.size());
    data.set_treatment_index(num_features);
    data.set_instrument_index(num_features);
    data.set_causal_survival_numerator_index(num_features + 1);
    data.set_causal_survival_denominator_index(num_features + 2);
    data.set_censor_index(num_features + 3);
    if (!sample_weights.empty()) {
      data.set_weight_index(num_features + 4);
    }

    grf::ForestTrainer trainer = grf::causal_survival_trainer(stabilize_splits);
    grf::ForestOptions options = make_options(
        num_trees,
        ci_group_size,
        sample_fraction,
        mtry,
        min_node_size,
        honesty,
        honesty_fraction,
        honesty_prune_leaves,
        alpha,
        imbalance_penalty,
        num_threads,
        seed);
    grf::Forest forest = trainer.train(data, options);

    return new CausalSurvivalForestHandle(
        num_rows,
        num_features,
        std::move(x_col_major),
        std::move(forest));
  } catch (const std::exception& error) {
    set_error(error);
    return nullptr;
  }
}

int csf_grf_causal_survival_predict(void* handle_ptr,
                                    const double* x_row_major,
                                    std::size_t num_rows,
                                    std::size_t num_features,
                                    bool oob_prediction,
                                    bool estimate_variance,
                                    unsigned int num_threads,
                                    double* output) {
  clear_error();
  try {
    auto* handle = static_cast<CausalSurvivalForestHandle*>(handle_ptr);
    if (handle == nullptr) {
      throw std::runtime_error("Causal survival forest handle is null.");
    }

    grf::ForestPredictor predictor = grf::causal_survival_predictor(num_threads);
    std::vector<grf::Prediction> predictions;

    if (oob_prediction) {
      const double* data_ptr = handle->train_covariates.data();
      std::vector<double> maybe_x;
      if (x_row_major != nullptr) {
        maybe_x = row_major_to_col_major(x_row_major, num_rows, num_features);
        data_ptr = maybe_x.data();
      }
      grf::Data data(data_ptr, num_rows, num_features);
      predictions = predictor.predict_oob(*handle->forest, data, estimate_variance);
    } else {
      grf::Data train_data(handle->train_covariates.data(), handle->num_rows, handle->num_features);
      std::vector<double> x_col_major = row_major_to_col_major(x_row_major, num_rows, num_features);
      grf::Data data(x_col_major.data(), num_rows, num_features);
      predictions = predictor.predict(*handle->forest, train_data, data, estimate_variance);
    }

    copy_predictions_to_buffer(predictions, output);
    return 0;
  } catch (const std::exception& error) {
    set_error(error);
    return 1;
  }
}

void csf_grf_causal_survival_free(void* handle_ptr) {
  delete static_cast<CausalSurvivalForestHandle*>(handle_ptr);
}

}  // extern "C"
