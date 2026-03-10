import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv


class CensoringModel:
    def __init__(self, n_estimators=100, min_samples_leaf=15, n_jobs=-1):
        self.rsf = RandomSurvivalForest(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
        )
        self.failure_times = None

    def fit(self, covariates, Y, delta):
        censoring_event = (delta == 0).astype(bool)
        y_surv = Surv.from_arrays(event=censoring_event, time=Y)
        self.rsf.fit(covariates, y_surv)
        self.failure_times = self.rsf.unique_times_

    def predict_surv_and_hazard(self, covariates):
        surv_funcs = self.rsf.predict_survival_function(covariates, return_array=True)
        chf_funcs = self.rsf.predict_cumulative_hazard_function(covariates, return_array=True)

        hazard_jumps = np.zeros_like(chf_funcs)
        hazard_jumps[:, 0] = chf_funcs[:, 0]
        for k in range(1, chf_funcs.shape[1]):
            hazard_jumps[:, k] = chf_funcs[:, k] - chf_funcs[:, k - 1]

        hazard_jumps = np.clip(hazard_jumps, 0, None)
        return surv_funcs, hazard_jumps, self.failure_times
