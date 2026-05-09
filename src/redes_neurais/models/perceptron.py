import numpy as np


class Perceptron:
    def __init__(self, X_train, y_train, learning_rate, max_epochs=1000):
        self.d = y_train
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.p, self.N = X_train.shape
        self.m = y_train.shape[0]
        self.X_train = np.vstack((
            -np.ones((1, self.N)),
            X_train
        ))
        self.w = np.random.random_sample((self.m, self.p + 1)) - .5
        self.error_history = []

    def activation_function(self, u):
        return np.where(u >= 0, 1., -1.)

    def fit(self):
        epochs = 0
        error = True

        while error and epochs < self.max_epochs:
            error = False
            errors_epoch = 0

            for k in range(self.N):
                x_k = self.X_train[:, k].reshape(self.p + 1, 1)
                d_k = self.d[:, k].reshape(self.m, 1)
                u_k = self.w @ x_k
                y_k = self.activation_function(u_k)
                e_k = d_k - y_k

                if np.any(e_k != 0):
                    error = True
                    errors_epoch += int(np.sum(e_k != 0))

                self.w = self.w + self.lr * e_k @ x_k.T

            self.error_history.append(errors_epoch)
            epochs += 1

        return self

    def decision_function(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_bias = np.vstack((
            -np.ones((1, X.shape[1])),
            X
        ))
        return self.w @ X_bias

    def predict(self, X):
        return self.activation_function(self.decision_function(X))
