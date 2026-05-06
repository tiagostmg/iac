import numpy as np

from models.base import ModelBase


class MediaY(ModelBase):
    def __init__(self):
        pass

    def fit(self, X, y, lbd=0):    
        self.y_mean_value = np.mean(y)
    
    def predict(self, X_pred):
        if not hasattr(self, "y_mean_value"):
            raise ValueError("Modelo ainda nao foi treinado. Chame fit antes de predict.")
        
        y_pred = np.full(X_pred.shape[0], self.y_mean_value)

        return y_pred
