import numpy as np
from models.base import ModelBase

class MQOClassificador(ModelBase):
    def __init__(self):
        super().__init__()

    def fit(self, X, Y):
        """
        X: Matriz de treino (N, p)
        Y: Matriz alvo em One-Hot Encoding (N, C)
        """
        # Adiciona a coluna de 1s para estimar o intercepto (bias trick)
        _X = np.column_stack((np.ones(len(X)), X))

        # W = (X^T X)^-1 X^T Y
        # Utilizamos pinv (pseudoinversa) para evitar erros se a matriz for singular
        self.W = np.linalg.pinv(_X.T @ _X) @ _X.T @ Y

    def predict(self, X_pred):
        """
        X_pred: Matriz de teste (N_test, p)
        Retorna: Vetor com as classes preditas de 1 a 5 (N_test,)
        """
        if not hasattr(self, "W"):
            raise ValueError("Modelo ainda nao foi treinado. Chame fit antes de predict.")

        _X = np.column_stack((np.ones(len(X_pred)), X_pred))
        
        # Y_pred terá dimensão (N_test, C) com valores contínuos
        Y_pred = _X @ self.W
        
        # A classe escolhida é o índice da coluna com o maior valor na linha
        # Somamos 1 pois o argmax retorna de 0 a 4, e queremos classes de 1 a 5
        classes_preditas = np.argmax(Y_pred, axis=1) + 1
        
        return classes_preditas
