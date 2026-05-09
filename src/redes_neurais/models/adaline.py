import numpy as np


class Adaline:
    def __init__(self, X_train, y_train, learning_rate, precision, max_epochs):
        self.d = y_train
        self.p, self.N = X_train.shape
        self.m = y_train.shape[0]
        self.w = np.random.random_sample((self.m, self.p + 1)) - .5
        self.X_train = np.vstack((
            -np.ones((1, self.N)),
            X_train
        ))
        self.lr = learning_rate
        self.pr = precision
        self.max_epochs = max_epochs
        self.error_history = []

    def activation_function(self, u):
        return np.where(u >= 0, 1., -1.)

    def eqm(self):
        eqm = 0
        for k in range(self.N):
            x_k = self.X_train[:, k].reshape(self.p + 1, 1)
            u_k = self.w @ x_k
            d_k = self.d[:, k].reshape(self.m, 1)
            eqm += np.sum((d_k - u_k) ** 2)
        return eqm / (2 * self.N)

    def fit(self):
        epochs = 0
        eqm_1 = 1
        eqm_2 = 0

        while epochs < self.max_epochs and abs(eqm_1 - eqm_2) > self.pr:
            eqm_1 = self.eqm()
            self.error_history.append(eqm_1)

            for k in range(self.N):
                x_k = self.X_train[:, k].reshape(self.p + 1, 1)
                u_k = self.w @ x_k
                d_k = self.d[:, k].reshape(self.m, 1)
                e_k = d_k - u_k
                self.w = self.w + self.lr * e_k @ x_k.T

            eqm_2 = self.eqm()
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
