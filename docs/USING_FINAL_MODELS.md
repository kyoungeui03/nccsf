# Using the Final Models on a New Dataset

This guide explains how to run the finalized NC-CSF estimators on a user-provided
dataset. It is separate from the synthetic benchmark runners because most real
applications should call the model classes directly after constructing `X`, `W`,
`Z`, treatment, and outcome arrays.

## Which Model Should I Use?

Use the censored model when the outcome is a right-censored time-to-event
outcome:

```python
from censored.final_censored_model import FinalCensoredModel
```

Use the non-censored model when the outcome is fully observed:

```python
from non_censored.final_noncensored_model import FinalNoncensoredModel
```

The oracle classes are for synthetic experiments only because they require
access to the latent confounder `U`.

## Required Data Structure

### Censored Outcome Data

You need:

- `X`: observed baseline covariates, shape `(n, p_x)`.
- `W`: negative-control outcome proxies, shape `(n, p_w)`.
- `Z`: negative-control treatment proxies, shape `(n, p_z)`.
- `A`: binary treatment indicator, shape `(n,)`, coded as `0` or `1`.
- `time`: observed follow-up time, shape `(n,)`, usually `min(T, C)`.
- `event`: event indicator, shape `(n,)`, where `1` means the event was observed
  and `0` means right-censored.
- `horizon`: target horizon for RMST or survival probability.

Targets:

- `target="RMST"` estimates the treatment effect on restricted mean survival
  time up to `horizon`.
- `target="survival.probability"` estimates the treatment effect on
  `P(T > horizon)`.

### Non-Censored Outcome Data

You need:

- `X`: observed baseline covariates, shape `(n, p_x)`.
- `W`: negative-control outcome proxies, shape `(n, p_w)`.
- `Z`: negative-control treatment proxies, shape `(n, p_z)`.
- `A`: binary treatment indicator, shape `(n,)`, coded as `0` or `1`.
- `Y`: observed outcome, shape `(n,)`.

## Important Modeling Warning

The proximal interpretation requires meaningful negative-control proxies.
If a dataset does not contain defensible `W` and `Z` variables, the model can
still be run as a sensitivity analysis, but the result should not be presented
as fully identified proximal causal evidence.

For datasets without designated negative controls, prefer one of these options:

- Put all observed covariates into `X` and use the result as an all-covariate
  sensitivity model.
- Define clinically or scientifically motivated proxy splits and clearly label
  them as sensitivity specifications.
- Compare against observed-covariate baselines.

## Censored Final Model Example

```python
import numpy as np
import pandas as pd

from censored.final_censored_model import FinalCensoredModel

df = pd.read_csv("my_censored_dataset.csv")

x_cols = ["age", "weight", "baseline_cd4"]
w_cols = ["followup_cd4", "followup_cd8"]
z_cols = ["prior_treatment", "race_indicator"]

X = df[x_cols].to_numpy(float)
W = df[w_cols].to_numpy(float)
Z = df[z_cols].to_numpy(float)
A = df["treatment"].to_numpy(float)
time = df["time"].to_numpy(float)
event = df["event"].to_numpy(float)

model = FinalCensoredModel(
    target="RMST",
    horizon=1000.0,
    censoring_mode="conditional",
    random_state=42,
    n_estimators=200,
    min_samples_leaf=20,
)

model.fit_components(X=X, A=A, time=time, event=event, Z=Z, W=W)
tau_hat = model.effect_from_components(X, W, Z)

df["tau_hat_rmst"] = tau_hat
df.to_csv("my_censored_dataset_with_cate.csv", index=False)
```

For survival probability at the same horizon:

```python
model = FinalCensoredModel(
    target="survival.probability",
    horizon=1000.0,
    censoring_mode="conditional",
    random_state=42,
    n_estimators=200,
)
```

Positive CATE means the treatment increases the target:

- For RMST: more restricted survival time under `A=1` than under `A=0`.
- For survival probability: higher probability of surviving past `horizon`
  under `A=1` than under `A=0`.

## Non-Censored Final Model Example

```python
import pandas as pd

from non_censored.final_noncensored_model import FinalNoncensoredModel

df = pd.read_csv("my_noncensored_dataset.csv")

x_cols = ["x1", "x2", "x3"]
w_cols = ["w_proxy1", "w_proxy2"]
z_cols = ["z_proxy1", "z_proxy2"]

X = df[x_cols].to_numpy(float)
W = df[w_cols].to_numpy(float)
Z = df[z_cols].to_numpy(float)
A = df["treatment"].to_numpy(float)
Y = df["outcome"].to_numpy(float)

model = FinalNoncensoredModel(
    random_state=42,
    n_estimators=200,
    min_samples_leaf=20,
)

model.fit_components(X=X, A=A, Y=Y, Z=Z, W=W)
tau_hat = model.effect_from_components(X, W, Z)

df["tau_hat"] = tau_hat
df.to_csv("my_noncensored_dataset_with_cate.csv", index=False)
```

Positive CATE means the model estimates a higher expected outcome under `A=1`
than under `A=0`.

## Train/Test Usage

For evaluation or deployment, fit on training data and predict on a separate
test or target population:

```python
model.fit_components(X=X_train, A=A_train, time=time_train, event=event_train, Z=Z_train, W=W_train)
tau_test = model.effect_from_components(X_test, W_test, Z_test)
```

For the non-censored model:

```python
model.fit_components(X=X_train, A=A_train, Y=Y_train, Z=Z_train, W=W_train)
tau_test = model.effect_from_components(X_test, W_test, Z_test)
```

The test arrays must use the same column order and preprocessing as the training
arrays.

## Practical Preprocessing Checklist

- Convert categorical variables to numeric encodings before calling the model.
- Impute or remove missing values; the model classes expect numeric arrays
  without `NaN`.
- Keep `A` binary and coded as `0/1`.
- Use the same scaling/encoding for train and test sets.
- Choose `W` and `Z` before looking at model performance to avoid
  post-selection interpretation problems.
- For censored outcomes, choose `horizon` before fitting the model.

## Recommended Hyperparameters

For fast development:

```python
n_estimators=200
q_trees=300
h_n_estimators=600
cv=5
```

For the ACTG175 paper-facing analysis, the final-stage forest used:

```python
n_estimators=10000
```

For exploratory real-data analysis, start with the fast setting, inspect
stability, and only then run larger forests.

## Running Baselines

Use baselines when you need a comparison rather than the final NC-CSF model.

Non-censored observed-feature EconML baseline:

```python
from non_censored.econml_noncensored_baseline import BaseNoncensoredModel
```

Censored no-PCI baseline:

```python
from censored.censored_baseline import CensoredBaselineCensoredSurvivalForest
```

Naive censored observed-feature baseline:

```python
from censored.naive_censored_baseline import NaiveEconmlXWZCensoredSurvivalForest
```

Direct R baselines require `Rscript` and R `grf`; see
[docs/R_BASELINE_SETUP.md](/Users/kyoungeuihong/Desktop/csf_grf_new/docs/R_BASELINE_SETUP.md).

## Suggested Output Layout for New Analyses

Do not write exploratory outputs into canonical paper-result folders. Use:

```text
outputs/reruns/<analysis_name>/
```

Suggested files:

- `model_config.json`: columns, hyperparameters, target, horizon, seed.
- `cate_predictions.csv`: subject-level CATE estimates.
- `summary.csv`: mean, median, quantiles, and any subgroup summaries.
- `figures/`: diagnostic plots.

This makes it clear which outputs are paper-final and which are new reruns.

## Minimal Sanity Checks

After fitting on a new dataset, check:

- `tau_hat.shape == (n,)`.
- No `NaN` or infinite CATE values.
- Treatment groups both have adequate sample size.
- For censored data, enough events occur before the chosen horizon.
- The CATE distribution is plausible in scale relative to the outcome.
- Results are qualitatively stable across seeds or bootstrap samples.

If these checks fail, inspect proxy definitions, missingness, horizon choice,
and censoring positivity before interpreting the estimates.
