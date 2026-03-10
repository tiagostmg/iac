from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from models.linear_regression import LinearRegression

def read_data():
    file_path = Path(__file__).resolve().parent / "data" / "aerogerador.dat"

    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {file_path}")

    data = np.loadtxt(file_path, delimiter="\t", dtype=float)


    X = data[:, 0] 
    y = data[:, 1] 
    
    return (X, y)

X, y = read_data()

model = LinearRegression()
model.fit(X, y)

X_pred = np.linspace(3, 14)
y_pred = model.predict(X_pred)

print(y_pred)
model.plot_pred(X, y, X_pred, y_pred)