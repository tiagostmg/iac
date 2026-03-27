from matplotlib import pyplot as plt


class ModelBase:
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