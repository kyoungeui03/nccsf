import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression


class TreatmentProxyModel:
    def __init__(self, model=None):
        self.model = model if model is not None else LogisticRegression(max_iter=1000)

    def fit(self, Z, X, A):
        ZX = X if Z is None else np.hstack((Z, X))
        self.model.fit(ZX, A)

    def predict_proba(self, Z, X):
        ZX = X if Z is None else np.hstack((Z, X))
        return self.model.predict_proba(ZX)[:, 1]


class OutcomeProxyModel:
    def __init__(self, model_0=None, model_1=None):
        self.model_0 = model_0 if model_0 is not None else RandomForestRegressor(n_estimators=100)
        self.model_1 = model_1 if model_1 is not None else RandomForestRegressor(n_estimators=100)

    def fit(self, W, X, A, Y_ipcw):
        mask_0 = A == 0
        if np.sum(mask_0) > 0:
            WX_0 = X[mask_0] if W is None else np.hstack((W[mask_0], X[mask_0]))
            self.model_0.fit(WX_0, Y_ipcw[mask_0])

        mask_1 = A == 1
        if np.sum(mask_1) > 0:
            WX_1 = X[mask_1] if W is None else np.hstack((W[mask_1], X[mask_1]))
            self.model_1.fit(WX_1, Y_ipcw[mask_1])

    def predict(self, W, X, A=None):
        WX = X if W is None else np.hstack((W, X))
        h_0 = self.model_0.predict(WX)
        h_1 = self.model_1.predict(WX)

        if A is None:
            return h_0, h_1
        return np.where(A == 1, h_1, h_0)


def compute_ipcw_outcome(Y, delta, surv_prob_C_at_Y):
    S_c_safe = np.maximum(surv_prob_C_at_Y, 1e-10)
    return (Y * delta) / S_c_safe
