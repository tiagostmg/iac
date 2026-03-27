from pathlib import Path

import numpy as np

from models.mqo import MQO
from models.media_y import MediaY

def read_data():
    file_path = Path(__file__).resolve().parents[1] / "data" / "aerogerador.dat"

    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {file_path}")

    data = np.loadtxt(file_path, delimiter="\t", dtype=float)


    X = data[:, :-1]
    y = data[:, -1]
    
    return (X, y)

X, y = read_data()

mqo = MQO()
media_y = MediaY()

def fit_and_predict(model, X, y, lbd=0.0):    
    model.fit(X, y, lbd)

    X_pred = np.linspace(3, 14)
    y_pred = model.predict(X_pred)

    model.plot_pred(X, y, X_pred, y_pred)

fit_and_predict(mqo, X, y)
fit_and_predict(mqo, X, y, 0.25)
fit_and_predict(mqo, X, y, 0.5)
fit_and_predict(mqo, X, y, 0.75)
fit_and_predict(mqo, X, y, 1)
fit_and_predict(media_y, X, y)
