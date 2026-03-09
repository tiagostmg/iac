import numpy as np
import matplotlib.pyplot as plt


X = np.linspace(10, 200, 20) + np.random.normal(0, 3, 20)

ones = np.ones(len(X))

X = np.column_stack((ones, X))

Y = 4 * X.T[1] + 20 + np.random.normal(0, 20, 20)

plt.scatter(X.T[1], Y)

B = np.linalg.inv(X.T @ X) @ X.T @ Y

y_pred = B[0] + B[1] * X.T[1]

order = np.argsort(X.T[1])
plt.plot(X.T[1][order], y_pred[order], color="red", label="reta ajustada")

plt.show()


