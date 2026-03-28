import numpy as np
from models.base import ModelBase

class ClassificadorGaussiano(ModelBase):
    def __init__(self, tipo_modelo="tradicional", lbd=0.5):
        """
        tipo_modelo: "tradicional", "iguais", "agregada", "naive_bayes", "friedman"
        lbd: Parâmetro lambda para o modelo de Friedman (0 a 1)
        """
        super().__init__()
        self.tipo_modelo = tipo_modelo
        self.lbd = lbd
        
        self.classes = []
        self.mu = {}
        self.sigma = {}
        self.priors = {}

    def fit(self, X, y):
        """
        X: Matriz de características (p, N)
        y: Vetor com rótulos originais de 1 a 5 (N,)
        """
        self.classes = np.unique(y)
        p, N = X.shape
        
        covs_individuais = {}
        Ns = {}
        
        # Passo 1: Calcular médias e covariâncias individuais para cada classe
        for c in self.classes:
            X_c = X[:, y == c]
            Ns[c] = X_c.shape[1]
            
            # Probabilidade a priori: P(c) = N_c / N
            self.priors[c] = Ns[c] / N
            
            # Média da classe: (p, 1)
            self.mu[c] = np.mean(X_c, axis=1, keepdims=True)
            
            # Covariância da classe: (p, p)
            covs_individuais[c] = np.cov(X_c)

        # Matriz global (para covariâncias iguais)
        cov_global = np.cov(X)
        
        # Matriz agregada (Média ponderada das covariâncias individuais)
        cov_agg = np.zeros((p, p))
        for c in self.classes:
            cov_agg += (Ns[c] / N) * covs_individuais[c]

        # Passo 2: Definir a Matriz de Covariância (Sigma) baseada no tipo de modelo
        for c in self.classes:
            if self.tipo_modelo == "tradicional": # QDA
                self.sigma[c] = covs_individuais[c]
                
            elif self.tipo_modelo == "iguais":
                self.sigma[c] = cov_global
                
            elif self.tipo_modelo == "agregada": # LDA
                self.sigma[c] = cov_agg
                
            elif self.tipo_modelo == "naive_bayes":
                # Zera tudo fora da diagonal principal (assume independência)
                self.sigma[c] = np.diag(np.diag(covs_individuais[c]))
                
            elif self.tipo_modelo == "friedman":
                # Combinação convexa: (1 - lambda)*Sigma_c + lambda*Sigma_agg
                self.sigma[c] = ((1 - self.lbd) * covs_individuais[c]) + (self.lbd * cov_agg)
            else:
                raise ValueError("Tipo de modelo inválido.")

    def predict(self, X):
        """
        X: Matriz de características de teste (p, N_test)
        Retorna: Vetor com as classes preditas (N_test,)
        """
        N_test = X.shape[1]
        
        # Matriz para armazenar o valor discriminante de cada classe para cada amostra
        discriminantes = np.zeros((len(self.classes), N_test))
        
        for idx, c in enumerate(self.classes):
            mu_c = self.mu[c]
            sig_c = self.sigma[c]
            prior_c = self.priors[c]
            
            # Adiciona um epsilon minúsculo na diagonal principal para garantir estabilidade numérica
            # Evita o erro de "Matriz Singular" na hora de calcular a inversa
            sig_estavel = sig_c + np.eye(sig_c.shape[0]) * 1e-8
            
            inv_sig = np.linalg.inv(sig_estavel)
            
            # slogdet é mais seguro que det() puro para evitar underflow logarítmico
            _, logdet = np.linalg.slogdet(sig_estavel)
            
            # (x - mu_c)
            diff = X - mu_c  # shape: (p, N_test)
            
            # Operação vetorizada para: (x - mu)^T * Sigma^-1 * (x - mu)
            # Isso acelera em centenas de vezes a execução ao invés de usar um for
            quad_term = np.sum(diff * (inv_sig @ diff), axis=0) 
            
            # Função Discriminante Gaussiana
            g_c = -0.5 * logdet - 0.5 * quad_term + np.log(prior_c)
            
            discriminantes[idx, :] = g_c
        
        # O modelo escolhe a classe que maximizou a função g_c
        indices_preditos = np.argmax(discriminantes, axis=0)
        
        # Mapeia de volta para os nomes das classes (1 a 5)
        return self.classes[indices_preditos]