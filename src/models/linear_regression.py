from matplotlib import pyplot as plt
import numpy as np


class LinearRegression:
    def __init__(self):
        ...

    def fit(self, X, y):
        X_ = np.column_stack((np.ones(len(X)), X))

        B_hat = np.linalg.inv(X_.T @ X_) @ X_.T @ y
        
        self.B_hat = B_hat
    
    def predict(self, X_pred):
        if not hasattr(self, "B_hat"):
            raise ValueError("Modelo ainda nao foi treinado. Chame fit antes de predict.")

        X_ = np.column_stack((np.ones(len(X_pred)), X_pred))
        y_pred = X_ @ self.B_hat
        return y_pred
    
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