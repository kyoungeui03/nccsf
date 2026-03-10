from __future__ import annotations

import ctypes
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BUILD_SCRIPT = PROJECT_ROOT / "scripts" / "build_native.py"
NATIVE_LIB_DIR = PROJECT_ROOT / "native" / "lib"


class NativeError(RuntimeError):
    pass


def _library_filename() -> str:
    if sys.platform == "darwin":
        return "libcsfgrf.dylib"
    if sys.platform.startswith("linux"):
        return "libcsfgrf.so"
    raise NativeError(f"Unsupported platform: {sys.platform}")


def _library_path() -> Path:
    return NATIVE_LIB_DIR / _library_filename()


def _ensure_native_library() -> Path:
    subprocess.run([sys.executable, str(BUILD_SCRIPT)], check=True, cwd=PROJECT_ROOT)
    library_path = _library_path()
    if not library_path.exists():
        raise NativeError(f"Native library build did not produce {library_path}.")
    return library_path


_LIB: ctypes.CDLL | None = None


def _load_library() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB

    library_path = _ensure_native_library()
    lib = ctypes.CDLL(str(library_path))

    double_ptr = ctypes.POINTER(ctypes.c_double)
    size_t = ctypes.c_size_t
    uint = ctypes.c_uint
    bool_t = ctypes.c_bool

    lib.csf_grf_last_error_message.restype = ctypes.c_char_p

    lib.csf_grf_regression_fit.argtypes = [
        double_ptr,
        double_ptr,
        double_ptr,
        size_t,
        size_t,
        uint,
        uint,
        uint,
        ctypes.c_double,
        bool_t,
        ctypes.c_double,
        bool_t,
        size_t,
        ctypes.c_double,
        ctypes.c_double,
        bool_t,
        uint,
        uint,
    ]
    lib.csf_grf_regression_fit.restype = ctypes.c_void_p
    lib.csf_grf_regression_predict.argtypes = [ctypes.c_void_p, double_ptr, size_t, size_t, bool_t, uint, double_ptr]
    lib.csf_grf_regression_predict.restype = ctypes.c_int
    lib.csf_grf_regression_free.argtypes = [ctypes.c_void_p]
    lib.csf_grf_regression_free.restype = None

    lib.csf_grf_survival_fit.argtypes = [
        double_ptr,
        double_ptr,
        double_ptr,
        double_ptr,
        size_t,
        size_t,
        uint,
        uint,
        uint,
        ctypes.c_double,
        bool_t,
        ctypes.c_double,
        bool_t,
        ctypes.c_double,
        size_t,
        ctypes.c_int,
        bool_t,
        bool_t,
        uint,
        uint,
    ]
    lib.csf_grf_survival_fit.restype = ctypes.c_void_p
    lib.csf_grf_survival_predict.argtypes = [
        ctypes.c_void_p,
        double_ptr,
        size_t,
        size_t,
        bool_t,
        uint,
        double_ptr,
    ]
    lib.csf_grf_survival_predict.restype = ctypes.c_int
    lib.csf_grf_survival_num_failures.argtypes = [ctypes.c_void_p]
    lib.csf_grf_survival_num_failures.restype = size_t
    lib.csf_grf_survival_free.argtypes = [ctypes.c_void_p]
    lib.csf_grf_survival_free.restype = None

    lib.csf_grf_causal_survival_fit.argtypes = [
        double_ptr,
        double_ptr,
        double_ptr,
        double_ptr,
        double_ptr,
        double_ptr,
        size_t,
        size_t,
        uint,
        uint,
        uint,
        ctypes.c_double,
        bool_t,
        ctypes.c_double,
        bool_t,
        size_t,
        ctypes.c_double,
        ctypes.c_double,
        bool_t,
        uint,
        uint,
    ]
    lib.csf_grf_causal_survival_fit.restype = ctypes.c_void_p
    lib.csf_grf_causal_survival_predict.argtypes = [
        ctypes.c_void_p,
        double_ptr,
        size_t,
        size_t,
        bool_t,
        bool_t,
        uint,
        double_ptr,
    ]
    lib.csf_grf_causal_survival_predict.restype = ctypes.c_int
    lib.csf_grf_causal_survival_free.argtypes = [ctypes.c_void_p]
    lib.csf_grf_causal_survival_free.restype = None

    _LIB = lib
    return lib


def _last_error(lib: ctypes.CDLL) -> str:
    message = lib.csf_grf_last_error_message()
    if message is None:
        return "Unknown native error."
    return message.decode("utf-8")


def _check_handle(handle: int | None, lib: ctypes.CDLL, operation: str) -> ctypes.c_void_p:
    if not handle:
        raise NativeError(f"{operation} failed: {_last_error(lib)}")
    return ctypes.c_void_p(handle)


def _check_status(status: int, lib: ctypes.CDLL, operation: str) -> None:
    if status != 0:
        raise NativeError(f"{operation} failed: {_last_error(lib)}")


def _as_double_vector(values: np.ndarray | list[float]) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.float64))


def _as_double_matrix(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("Expected a 2D matrix.")
    return np.ascontiguousarray(array)


def _optional_vector_ptr(values: np.ndarray | None) -> ctypes.POINTER(ctypes.c_double) | None:
    if values is None:
        return None
    return values.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _matrix_ptr(values: np.ndarray) -> ctypes.POINTER(ctypes.c_double):
    return values.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _vector_ptr(values: np.ndarray) -> ctypes.POINTER(ctypes.c_double):
    return values.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _find_interval(values: np.ndarray, sorted_grid: np.ndarray) -> np.ndarray:
    return np.searchsorted(sorted_grid, values, side="right")


def _remap_survival_predictions(
    raw_predictions: np.ndarray,
    train_failure_times: np.ndarray,
    requested_failure_times: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if requested_failure_times is None:
        return raw_predictions, train_failure_times

    requested_failure_times = _as_double_vector(requested_failure_times)
    if requested_failure_times.ndim != 1:
        raise ValueError("failure_times must be a 1D vector.")
    if np.any(np.diff(requested_failure_times) <= 0):
        raise ValueError("failure_times must be strictly increasing.")

    indices = _find_interval(requested_failure_times, train_failure_times)
    remapped = np.ones((raw_predictions.shape[0], requested_failure_times.shape[0]), dtype=float)
    valid = indices > 0
    if np.any(valid):
        remapped[:, valid] = raw_predictions[:, indices[valid] - 1]
    return remapped, requested_failure_times


@dataclass
class NativeRegressionForest:
    handle: ctypes.c_void_p
    train_x: np.ndarray
    num_threads: int

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        num_trees: int,
        sample_fraction: float,
        mtry: int,
        min_node_size: int,
        honesty: bool,
        honesty_fraction: float,
        honesty_prune_leaves: bool,
        alpha: float,
        imbalance_penalty: float,
        ci_group_size: int,
        compute_oob_predictions: bool,
        num_threads: int,
        seed: int,
        sample_weights: np.ndarray | None = None,
    ) -> "NativeRegressionForest":
        lib = _load_library()
        X = _as_double_matrix(X)
        Y = _as_double_vector(Y)
        weights = None if sample_weights is None else _as_double_vector(sample_weights)

        handle = _check_handle(
            lib.csf_grf_regression_fit(
                _matrix_ptr(X),
                _vector_ptr(Y),
                _optional_vector_ptr(weights),
                X.shape[0],
                X.shape[1],
                int(mtry),
                int(num_trees),
                int(min_node_size),
                float(sample_fraction),
                bool(honesty),
                float(honesty_fraction),
                bool(honesty_prune_leaves),
                int(ci_group_size),
                float(alpha),
                float(imbalance_penalty),
                bool(compute_oob_predictions),
                int(num_threads),
                int(seed),
            ),
            lib,
            "Regression forest fit",
        )
        return cls(handle=handle, train_x=X, num_threads=num_threads)

    def predict(self, newdata: np.ndarray | None = None) -> np.ndarray:
        lib = _load_library()
        if newdata is None:
            output = np.empty(self.train_x.shape[0], dtype=np.float64)
            _check_status(
                lib.csf_grf_regression_predict(
                    self.handle,
                    None,
                    self.train_x.shape[0],
                    self.train_x.shape[1],
                    True,
                    int(self.num_threads),
                    _vector_ptr(output),
                ),
                lib,
                "Regression forest OOB predict",
            )
            return output

        newdata = _as_double_matrix(newdata)
        output = np.empty(newdata.shape[0], dtype=np.float64)
        _check_status(
            lib.csf_grf_regression_predict(
                self.handle,
                _matrix_ptr(newdata),
                newdata.shape[0],
                newdata.shape[1],
                False,
                int(self.num_threads),
                _vector_ptr(output),
            ),
            lib,
            "Regression forest predict",
        )
        return output

    def close(self) -> None:
        if self.handle:
            try:
                _load_library().csf_grf_regression_free(self.handle)
            except Exception:
                pass
            self.handle = ctypes.c_void_p()

    def __del__(self) -> None:
        self.close()


@dataclass
class NativeSurvivalForest:
    handle: ctypes.c_void_p
    train_x: np.ndarray
    failure_times: np.ndarray
    prediction_type: int
    num_threads: int

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        Y: np.ndarray,
        D: np.ndarray,
        *,
        failure_times: np.ndarray | None,
        num_trees: int,
        sample_fraction: float,
        mtry: int,
        min_node_size: int,
        honesty: bool,
        honesty_fraction: float,
        honesty_prune_leaves: bool,
        alpha: float,
        prediction_type: int,
        fast_logrank: bool,
        compute_oob_predictions: bool,
        num_threads: int,
        seed: int,
        sample_weights: np.ndarray | None = None,
    ) -> "NativeSurvivalForest":
        lib = _load_library()
        X = _as_double_matrix(X)
        Y = _as_double_vector(Y)
        D = _as_double_vector(D)
        weights = None if sample_weights is None else _as_double_vector(sample_weights)

        if failure_times is None:
            failure_times = np.sort(np.unique(Y[D == 1]))
        else:
            failure_times = _as_double_vector(failure_times)
            if np.any(np.diff(failure_times) <= 0):
                raise ValueError("failure_times must be strictly increasing.")

        y_relabeled = _find_interval(Y, failure_times).astype(np.float64)

        handle = _check_handle(
            lib.csf_grf_survival_fit(
                _matrix_ptr(X),
                _vector_ptr(y_relabeled),
                _vector_ptr(D),
                _optional_vector_ptr(weights),
                X.shape[0],
                X.shape[1],
                int(mtry),
                int(num_trees),
                int(min_node_size),
                float(sample_fraction),
                bool(honesty),
                float(honesty_fraction),
                bool(honesty_prune_leaves),
                float(alpha),
                int(failure_times.shape[0]),
                int(prediction_type),
                bool(fast_logrank),
                bool(compute_oob_predictions),
                int(num_threads),
                int(seed),
            ),
            lib,
            "Survival forest fit",
        )
        return cls(
            handle=handle,
            train_x=X,
            failure_times=failure_times,
            prediction_type=prediction_type,
            num_threads=num_threads,
        )

    def _predict_raw(self, matrix: np.ndarray | None, *, oob: bool, num_threads: int | None = None) -> np.ndarray:
        lib = _load_library()
        threads = self.num_threads if num_threads is None else int(num_threads)
        target_rows = self.train_x.shape[0] if matrix is None else matrix.shape[0]
        output = np.empty((target_rows, self.failure_times.shape[0]), dtype=np.float64)
        status = lib.csf_grf_survival_predict(
            self.handle,
            None if matrix is None else _matrix_ptr(matrix),
            target_rows,
            self.train_x.shape[1] if matrix is None else matrix.shape[1],
            bool(oob),
            threads,
            _vector_ptr(output.reshape(-1)),
        )
        _check_status(status, lib, "Survival forest predict")
        return output

    def predict(
        self,
        newdata: np.ndarray | None = None,
        *,
        failure_times: np.ndarray | None = None,
        num_threads: int | None = None,
    ) -> np.ndarray:
        matrix = None if newdata is None else _as_double_matrix(newdata)
        raw = self._predict_raw(matrix, oob=newdata is None, num_threads=num_threads)
        remapped, _ = _remap_survival_predictions(raw, self.failure_times, failure_times)
        return remapped

    def predict_oob(
        self,
        x_override: np.ndarray | None = None,
        *,
        failure_times: np.ndarray | None = None,
        num_threads: int | None = None,
    ) -> np.ndarray:
        matrix = None if x_override is None else _as_double_matrix(x_override)
        raw = self._predict_raw(matrix, oob=True, num_threads=num_threads)
        remapped, _ = _remap_survival_predictions(raw, self.failure_times, failure_times)
        return remapped

    def close(self) -> None:
        if self.handle:
            try:
                _load_library().csf_grf_survival_free(self.handle)
            except Exception:
                pass
            self.handle = ctypes.c_void_p()

    def __del__(self) -> None:
        self.close()


@dataclass
class NativeCausalSurvivalForest:
    handle: ctypes.c_void_p
    train_x: np.ndarray
    num_threads: int

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        treatment: np.ndarray,
        numerator: np.ndarray,
        denominator: np.ndarray,
        censor: np.ndarray,
        *,
        num_trees: int,
        sample_fraction: float,
        mtry: int,
        min_node_size: int,
        honesty: bool,
        honesty_fraction: float,
        honesty_prune_leaves: bool,
        alpha: float,
        imbalance_penalty: float,
        stabilize_splits: bool,
        ci_group_size: int,
        num_threads: int,
        seed: int,
        sample_weights: np.ndarray | None = None,
    ) -> "NativeCausalSurvivalForest":
        lib = _load_library()
        X = _as_double_matrix(X)
        treatment = _as_double_vector(treatment)
        numerator = _as_double_vector(numerator)
        denominator = _as_double_vector(denominator)
        censor = _as_double_vector(censor)
        weights = None if sample_weights is None else _as_double_vector(sample_weights)

        handle = _check_handle(
            lib.csf_grf_causal_survival_fit(
                _matrix_ptr(X),
                _vector_ptr(treatment),
                _vector_ptr(numerator),
                _vector_ptr(denominator),
                _vector_ptr(censor),
                _optional_vector_ptr(weights),
                X.shape[0],
                X.shape[1],
                int(mtry),
                int(num_trees),
                int(min_node_size),
                float(sample_fraction),
                bool(honesty),
                float(honesty_fraction),
                bool(honesty_prune_leaves),
                int(ci_group_size),
                float(alpha),
                float(imbalance_penalty),
                bool(stabilize_splits),
                int(num_threads),
                int(seed),
            ),
            lib,
            "Causal survival forest fit",
        )
        return cls(handle=handle, train_x=X, num_threads=num_threads)

    def predict(self, newdata: np.ndarray | None = None, estimate_variance: bool = False) -> np.ndarray:
        lib = _load_library()
        if newdata is None:
            output = np.empty(self.train_x.shape[0], dtype=np.float64)
            status = lib.csf_grf_causal_survival_predict(
                self.handle,
                None,
                self.train_x.shape[0],
                self.train_x.shape[1],
                True,
                bool(estimate_variance),
                int(self.num_threads),
                _vector_ptr(output),
            )
            _check_status(status, lib, "Causal survival forest OOB predict")
            return output

        newdata = _as_double_matrix(newdata)
        output = np.empty(newdata.shape[0], dtype=np.float64)
        status = lib.csf_grf_causal_survival_predict(
            self.handle,
            _matrix_ptr(newdata),
            newdata.shape[0],
            newdata.shape[1],
            False,
            bool(estimate_variance),
            int(self.num_threads),
            _vector_ptr(output),
        )
        _check_status(status, lib, "Causal survival forest predict")
        return output

    def close(self) -> None:
        if self.handle:
            try:
                _load_library().csf_grf_causal_survival_free(self.handle)
            except Exception:
                pass
            self.handle = ctypes.c_void_p()

    def __del__(self) -> None:
        self.close()
