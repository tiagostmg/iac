from abc import ABC, abstractmethod

from matplotlib import pyplot as plt
import numpy as np


class ModelBase(ABC):
    def __init__(self):
        pass
    
    def plot(self, X, y, xlabel="Velocidade do vento", ylabel="Potência gerada", plot=True):
        plt.scatter(X, y, alpha=0.3)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, linestyle="--", alpha=0.3)
        if plot: plt.show()

    def plot_pred(self, X, y, X_pred, y_pred):
        self.plot(X, y, plot=False)
        plt.plot(X_pred, y_pred, c="r")
        plt.show()
        
    @abstractmethod
    def predict(self, X_pred: np.ndarray) -> np.ndarray:
        ...
    
    def sse(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return np.sum((y_test - y_pred) ** 2)
    
    def mse(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return np.mean((y_test - y_pred) ** 2)
    
    def sst(self, y_test):
        y_mean = self.y_mean(y_test)
        return np.sum((y_test - y_mean) ** 2)
    
    def y_mean(self, y_test):
        return np.mean(y_test)
    
    def r_squared(self, X_test, y_test):
        return 1 - (self.sse(X_test,  y_test) / self.sst(y_test))
        
