from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs_classificacao"
OUTPUTS_DIR.mkdir(exist_ok=True)

def plot_fronteiras(modelo, X_raw, y_raw, titulo, is_mqo=False):
    """
    Plota as fronteiras de decisão do modelo treinado.
    X_raw: formato (2, N)
    """
    # Define os limites do gráfico com uma margem
    x_min, x_max = X_raw[0, :].min() - 50, X_raw[0, :].max() + 50
    y_min, y_max = X_raw[1, :].min() - 50, X_raw[1, :].max() + 50
    
    # Cria uma malha de pontos (grid) para testar o modelo em todo o espaço
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Prepara os pontos do grid para o predict
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    if is_mqo:
        # MQO espera formato (N, p)
        Z = modelo.predict(grid)
    else:
        # Gaussianos esperam formato (p, N)
        Z = modelo.predict(grid.T)
        
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    
    # Cores de fundo (fronteiras) e cores dos pontos
    cores_fundo = ['#d3d3d3', '#add8e6', '#90ee90', '#ffb347', '#ffcccb']
    cores_pontos = ['gray', 'blue', 'green', 'orange', 'red']
    cmap_fundo = ListedColormap(cores_fundo)
    
    # Plota as regiões de decisão
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap_fundo)
    
    # Para não sobrecarregar o gráfico com 50.000 pontos, vamos plotar uma amostra de 2500
    np.random.seed(42)
    idx_amostra = np.random.choice(X_raw.shape[1], 2500, replace=False)
    
    categorias = {1: "Neutro", 2: "Sorriso", 3: "Sobrancelhas levantadas", 4: "Surpreso", 5: "Rabugento"}
    
    for classe, cor in zip(categorias.keys(), cores_pontos):
        idx_classe = (y_raw[idx_amostra] == classe)
        plt.scatter(X_raw[0, idx_amostra][idx_classe], 
                    X_raw[1, idx_amostra][idx_classe], 
                    c=cor, label=categorias[classe], 
                    edgecolor='k', s=20, alpha=0.7)
        
    plt.title(f"Fronteiras de Decisão - {titulo}")
    plt.xlabel("Sensor 1")
    plt.ylabel("Sensor 2")
    plt.legend(loc="best")
    
    nome_arquivo = f"fronteira_{titulo.replace(' ', '_').lower()}.png"
    plt.savefig(OUTPUTS_DIR / nome_arquivo, dpi=300)
    plt.show()