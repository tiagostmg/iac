import numpy as np


class MultilayerPerceptron:
    def __init__(self, topology, X_train, Y_train, learning_rate, max_epochs, precision):
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.pr = precision
        self.hidden_topology = list(topology)
        self.D = Y_train
        self.m = Y_train.shape[0]
        self.p, self.N = X_train.shape
        self.topology = self.hidden_topology + [self.m]
        self.X_train = np.vstack((
            -np.ones((1, self.N)), X_train
        ))
        self.W = []

        for i, q in enumerate(self.topology):
            if i == 0:
                W = np.random.random_sample((q, self.p + 1)) - .5
            else:
                W = np.random.random_sample((q, self.topology[i - 1] + 1)) - .5
            self.W.append(W)


        self.u = [None] * len(self.W)
        self.y = [None] * len(self.W)
        self.delta = [None] * len(self.W)
        self.error_history = []

    def g(self, u):
        s = np.exp(-u)
        return (1 - s) / (1 + s)

    def g_d(self, u):
        s = self.g(u)
        return .5 * (1 - s ** 2)

    def eqm(self):
        eqm = 0
        for k in range(self.N):
            x_k = self.X_train[:, k].reshape(self.p + 1, 1)
            self.forward(x_k)
            d_k = self.D[:, k].reshape(self.topology[-1], 1)
            eqm += np.sum((d_k - self.y[-1]) ** 2)
        return eqm / (2 * self.N)

    def forward(self, x):
        for j, W in enumerate(self.W):
            if j == 0:
                self.u[j] = W @ x
            else:
                yb = np.vstack((
                    -np.ones((1, 1)), self.y[j - 1]
                ))
                self.u[j] = W @ yb

            self.y[j] = self.g(self.u[j])

        return self.y[-1]

    def backward(self, x, d):
        for j in range(len(self.W) - 1, -1, -1):
            if j == len(self.W) - 1:
                self.delta[j] = self.g_d(self.u[j]) * (d - self.y[-1])
                if j == 0:
                    self.W[j] = self.W[j] + self.lr * self.delta[j] @ x.T
                else:
                    yb = np.vstack((
                        -np.ones((1, 1)), self.y[j - 1]
                    ))
                    self.W[j] = self.W[j] + self.lr * self.delta[j] @ yb.T
            elif j == 0:
                Wnb = self.W[j + 1][:, 1:].T
                self.delta[j] = self.g_d(self.u[j]) * (Wnb @ self.delta[j + 1])
                self.W[j] = self.W[j] + self.lr * self.delta[j] @ x.T
            else:
                Wnb = self.W[j + 1][:, 1:].T
                self.delta[j] = self.g_d(self.u[j]) * (Wnb @ self.delta[j + 1])
                yb = np.vstack((
                    -np.ones((1, 1)), self.y[j - 1]
                ))
                self.W[j] = self.W[j] + self.lr * self.delta[j] @ yb.T

    def fit(self):
        epochs = 0
        eqm = self.eqm()

        while epochs < self.max_epochs and eqm > self.pr:
            for k in range(self.N):
                x_k = self.X_train[:, k].reshape(self.p + 1, 1)
                d_k = self.D[:, k].reshape(self.topology[-1], 1)
                self.forward(x_k)
                self.backward(x_k, d_k)

            epochs += 1
            eqm = self.eqm()
            self.error_history.append(eqm)

        return self

    def decision_function(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        outputs = np.empty((self.m, X.shape[1]))
        X_bias = np.vstack((
            -np.ones((1, X.shape[1])),
            X
        ))

        for k in range(X.shape[1]):
            x_k = X_bias[:, k].reshape(self.p + 1, 1)
            outputs[:, k:k + 1] = self.forward(x_k)

        return outputs

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1., -1.)
