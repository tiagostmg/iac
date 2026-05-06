import numpy as np

from models.base import ModelBase


class MQO(ModelBase):
    def __init__(self):
        pass

    def fit(self, X, y, lbd=0):
        _X = np.column_stack((np.ones(len(X)), X))

        B_hat = np.linalg.pinv(_X.T @ _X + (lbd * np.eye(M=_X.shape[1],N=_X.shape[1]))) @ _X.T @ y
        
        self.B_hat = B_hat
    
    def predict(self, X_pred):
        if not hasattr(self, "B_hat"):
            raise ValueError("Modelo ainda nao foi treinado. Chame fit antes de predict.")

        X_ = np.column_stack((np.ones(len(X_pred)), X_pred))
        y_pred = X_ @ self.B_hat
        return y_pred
    
