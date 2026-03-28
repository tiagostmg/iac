from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from models.mqo import MQOClassificador
from models.gaussianos import ClassificadorGaussiano
from matplotlib.colors import ListedColormap
from plot_fronteira import plot_fronteiras
from melhor_lambda_monte_carlo import encontrar_melhor_lambda_kfold
from melhor_lambda_monte_carlo import simulacao_monte_carlo

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs_classificacao"
OUTPUTS_DIR.mkdir(exist_ok=True)

def read_data():
  
    file_path = Path(__file__).resolve().parents[1] / "data" / "EMGsDataset.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {file_path}")

 
    data = np.loadtxt(file_path, delimiter=",", dtype=float)
    
 
    X = data[0:2, :]  
    Y = data[2, :].astype(int) 
    
    N = X.shape[1]
    C = 5 # Classes de 1 a 5
    

    Y_bayes = np.zeros((C, N))
    # Como as classes vão de 1 a 5, subtraímos 1 para usar como índices (0 a 4)
    Y_bayes[Y - 1, np.arange(N)] = 1
    
  
    X_bayes = X   
    
  
    X_mqo = X.T       
    Y_mqo = Y_bayes.T     
    

    return X_mqo, Y_mqo, X_bayes, Y_bayes, Y

def plot_scatter(X, y_raw):

    # Dicionário com os nomes das categorias conforme o roteiro
    categorias = {
        1: "Neutro",
        2: "Sorriso",
        3: "Sobrancelhas levantadas",
        4: "Surpreso",
        5: "Rabugento"
    }
    
    plt.figure(figsize=(10, 8))
    

    cores = ['gray', 'blue', 'green', 'orange', 'red']
    
    for (classe, nome), cor in zip(categorias.items(), cores):
        # Filtra os índices onde o rótulo é igual à classe atual
        idx = (y_raw == classe)
        
   
        plt.scatter(X[0, idx], X[1, idx], label=nome, alpha=0.3, s=10, c=cor)
        
    plt.title("Gráfico de Espalhamento - Sinais EMG (Faces)")
    plt.xlabel("Sensor 1 (Corrugador do Supercílio) - Resolução ADC")
    plt.ylabel("Sensor 2 (Zigomático Maior) - Resolução ADC")
    plt.legend(loc="best")
    plt.grid(True, linestyle='--', alpha=0.6)
    
  
    plt.savefig(OUTPUTS_DIR / "scatter_plot_classes.png", dpi=300)
    plt.show()

if __name__ == "__main__":

    PARTE_EXECUTAR = 4  # <-- Mude este número para alternar!

    # Carrega e organiza os dados (necessário para todas as partes)
    X_mqo, Y_mqo, X_bayes, Y_bayes, y_raw = read_data()

    if PARTE_EXECUTAR == 2:
        print("--- Executando Parte 2: Gráfico de Espalhamento ---")
        plot_scatter(X_bayes, y_raw)


    elif PARTE_EXECUTAR == 3:
        print("--- Executando Parte 3: Treinamento e Fronteiras de Decisão ---\n")
        
        # 1. MQO Tradicional
        print("--- MQO Tradicional ---")
        modelo_mqo = MQOClassificador()
        modelo_mqo.fit(X_mqo, Y_mqo)
        pred_mqo = modelo_mqo.predict(X_mqo)
        acc_mqo = np.mean(pred_mqo == y_raw) * 100
        print(f"Acurácia: {acc_mqo:.2f}%\n")
        plot_fronteiras(modelo_mqo, X_bayes, y_raw, "MQO Tradicional", is_mqo=True)

        # 2. Gaussianos
        modelos_gaussianos = {
            "Gaussiano Tradicional": {"tipo": "tradicional", "lbd": 0},
            "Covariâncias Iguais": {"tipo": "iguais", "lbd": 0},
            "Matriz Agregada": {"tipo": "agregada", "lbd": 0},
            "Naive Bayes": {"tipo": "naive_bayes", "lbd": 0},
            "Friedman (Lambda=0.5)": {"tipo": "friedman", "lbd": 0.5}
        }

        for nome_modelo, config in modelos_gaussianos.items():
            print(f"--- {nome_modelo} ---")
            modelo_atual = ClassificadorGaussiano(tipo_modelo=config["tipo"], lbd=config["lbd"])
            modelo_atual.fit(X_bayes, y_raw)
            pred_atual = modelo_atual.predict(X_bayes)
            acc_atual = np.mean(pred_atual == y_raw) * 100
            print(f"Acurácia: {acc_atual:.2f}%\n")
            plot_fronteiras(modelo_atual, X_bayes, y_raw, nome_modelo, is_mqo=False)
            
        print("Fronteiras geradas e salvas na pasta 'outputs_classificacao'!")


    elif PARTE_EXECUTAR == 4:
        print("--- Executando Parte 4: Validação Robusta (Monte Carlo) ---\n")
        
        # 1. K-Fold para descobrir o melhor Lambda do modelo de Friedman
        melhor_lambda = encontrar_melhor_lambda_kfold(X_bayes, y_raw, k=10)
        
        # --- NOVO: PLOTAR FRONTEIRA DO MELHOR FRIEDMAN ---
        print(f"\nGerando o gráfico de fronteira para o melhor modelo de Friedman (Lambda = {melhor_lambda:.4f})...")
        
        # Instancia, treina e plota o modelo com o lambda vencedor
        modelo_melhor_friedman = ClassificadorGaussiano(tipo_modelo="friedman", lbd=melhor_lambda)
        modelo_melhor_friedman.fit(X_bayes, y_raw)
        
        titulo_grafico = f"Friedman_Otimo_Lambda_{melhor_lambda:.4f}"
        plot_fronteiras(modelo_melhor_friedman, X_bayes, y_raw, titulo_grafico, is_mqo=False)
        print("Gráfico da fronteira ótima salvo com sucesso!\n")
        
        # 2. Simulação de Monte Carlo com todos os modelos
        simulacao_monte_carlo(X_bayes, Y_bayes, Y_mqo, y_raw, melhor_lambda, rodadas=500)