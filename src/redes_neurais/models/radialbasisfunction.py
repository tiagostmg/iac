import numpy as np

class RadialBasisFunction:
    def __init__(self, n_neurons, X_train, Y_train, learning_rate, precision, max_epochs):
        self.n_centers = n_neurons  # Quantidade de neurônios na camada oculta
        self.X_train = X_train
        self.d = Y_train
        self.p, self.N = X_train.shape
        self.m = Y_train.shape[0]
        
        self.lr = learning_rate
        self.pr = precision
        self.max_epochs = max_epochs
        
        self.centers = None
        self.sigma = None
        
        # Pesos da camada de saída (m neurônios de saída x (n_centers + 1 bias))
        self.w = np.random.random_sample((self.m, self.n_centers + 1)) - 0.5
        self.error_history = []
        
        self._initialize_centers()
        
    def _initialize_centers(self):
        # Seleciona aleatoriamente 'n_centers' amostras do conjunto de treinamento para serem os centros
        indices = np.random.choice(self.N, self.n_centers, replace=False)
        self.centers = self.X_train[:, indices]
        
        # Calcula o espalhamento (sigma) com base na distância máxima entre os centros
        d_max = 0
        for i in range(self.n_centers):
            for j in range(self.n_centers):
                dist = np.linalg.norm(self.centers[:, i] - self.centers[:, j])
                if dist > d_max:
                    d_max = dist
                    
        # Heurística padrão para RBF
        self.sigma = d_max / np.sqrt(2 * self.n_centers)
        if self.sigma == 0:
            self.sigma = 1.0  # Evita divisão por zero

    def _phi(self, X):
        # Calcula a ativação Gaussiana para a camada oculta
        N_samples = X.shape[1]
        Phi = np.zeros((self.n_centers, N_samples))
        
        for i in range(self.n_centers):
            center = self.centers[:, i].reshape(-1, 1)
            dist_squared = np.sum((X - center) ** 2, axis=0)
            Phi[i, :] = np.exp(-dist_squared / (2 * self.sigma ** 2))
            
        return Phi

    def eqm(self):
        eqm = 0
        for k in range(self.N):
            phi_k = self.Phi_bias[:, k].reshape(self.n_centers + 1, 1)
            u_k = self.w @ phi_k
            d_k = self.d[:, k].reshape(self.m, 1)
            eqm += np.sum((d_k - u_k) ** 2)
        return eqm / (2 * self.N)

    def fit(self):
        # Pré-computa a saída da camada oculta para todos os dados de treino (os centros não mudam)
        Phi = self._phi(self.X_train)
        
        # Adiciona o Bias (-1) na matriz de ativação oculta (mesmo padrão do seu Adaline)
        self.Phi_bias = np.vstack((-np.ones((1, self.N)), Phi))
        
        epochs = 0
        eqm_1 = 1
        eqm_2 = 0
        
        while epochs < self.max_epochs and abs(eqm_1 - eqm_2) > self.pr:
            eqm_1 = self.eqm()
            self.error_history.append(eqm_1)
            
            # Atualização dos pesos via Gradiente Descendente (Regra Delta)
            for k in range(self.N):
                phi_k = self.Phi_bias[:, k].reshape(self.n_centers + 1, 1)
                u_k = self.w @ phi_k
                d_k = self.d[:, k].reshape(self.m, 1)
                e_k = d_k - u_k
                
                self.w = self.w + self.lr * e_k @ phi_k.T
                
            eqm_2 = self.eqm()
            epochs += 1
            
        return self

    def decision_function(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # Passa os dados novos pela camada oculta
        Phi = self._phi(X)
        Phi_bias = np.vstack((-np.ones((1, X.shape[1])), Phi))
        
        # Multiplica pelos pesos treinados
        return self.w @ Phi_bias

    def predict(self, X):
        # Retorna a classe (+1 ou -1) baseada na saída linear
        return np.where(self.decision_function(X) >= 0, 1., -1.)