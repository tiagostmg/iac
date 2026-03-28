from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from models.mqo import MQOClassificador
from models.gaussianos import ClassificadorGaussiano

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs_classificacao"
OUTPUTS_DIR.mkdir(exist_ok=True)

def encontrar_melhor_lambda_kfold(X, y, k=10):
    """
    Aplica K-Fold Cross Validation para encontrar o melhor lambda do modelo de Friedman.
    X: Matriz (p, N)
    y: Vetor (N,)
    """
    print(f"--- Iniciando {k}-Fold para encontrar o melhor Lambda (Friedman) ---")
    N = X.shape[1]
    
    np.random.seed(42)
    indices_embaralhados = np.random.permutation(N)
    
    tamanho_fold = N // k
    # Valores exatos de lambda exigidos no roteiro do trabalho
    lambdas_teste = np.array([0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    resultados_lambda = {}
    
    for lbd in lambdas_teste:
        acuracias_fold = []
        
        for i in range(k):
            inicio_teste = i * tamanho_fold
            fim_teste = (i + 1) * tamanho_fold if i != (k-1) else N
            
            idx_teste = indices_embaralhados[inicio_teste:fim_teste]
            idx_treino = np.setdiff1d(indices_embaralhados, idx_teste)
            
            X_treino, y_treino = X[:, idx_treino], y[idx_treino]
            X_teste, y_teste = X[:, idx_teste], y[idx_teste]
            
            modelo = ClassificadorGaussiano(tipo_modelo="friedman", lbd=lbd)
            modelo.fit(X_treino, y_treino)
            pred = modelo.predict(X_teste)
            
            acc = np.mean(pred == y_teste)
            acuracias_fold.append(acc)
            
        media_acc = np.mean(acuracias_fold)
        resultados_lambda[lbd] = media_acc
        print(f"Lambda = {lbd:.4f} | Acurácia Média = {media_acc * 100:.2f}%")
        
    melhor_lambda = max(resultados_lambda, key=resultados_lambda.get)
    print(f"\n=> Melhor Lambda encontrado: {melhor_lambda:.4f} (Acurácia: {resultados_lambda[melhor_lambda] * 100:.2f}%)\n")
    
    # --- NOVO: GERAR O GRÁFICO DO K-FOLD ---
    plt.figure(figsize=(8, 5))
    valores_acc = [resultados_lambda[l] * 100 for l in lambdas_teste]
    
    plt.plot(lambdas_teste, valores_acc, marker='o', linestyle='-', color='b')
    plt.axvline(x=melhor_lambda, color='r', linestyle='--', label=f'Melhor $\lambda$ = {melhor_lambda:.4f}')
    
    plt.title(f"K-Fold ({k} folds) - Busca pelo melhor Lambda (Friedman)")
    plt.xlabel("Valores de Lambda ($\lambda$)")
    plt.ylabel("Acurácia Média (%)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.savefig(OUTPUTS_DIR / "kfold_lambda_friedman.png", dpi=300)
    plt.show()
    
    return melhor_lambda


def simulacao_monte_carlo(X_raw, Y_bayes, Y_mqo, y_raw, melhor_lambda, rodadas=500):
    """
    Executa a simulação de Monte Carlo com 500 rodadas e plota o Boxplot.
    """
    print(f"--- Iniciando Simulação de Monte Carlo ({rodadas} rodadas) ---")
    N = X_raw.shape[1]
    tamanho_treino = int(0.8 * N)
    
    historico_acc = {
        "MQO": [],
        "Gauss. Tradicional": [],
        "Cov. Iguais": [],
        "Matriz Agregada": [],
        "Naive Bayes": [],
        f"Friedman ($\lambda$={melhor_lambda:.2f})": []
    }
    
    for r in range(rodadas):
        if r % 50 == 0:
            print(f"Processando rodada {r}/{rodadas}...")
            
        indices = np.random.permutation(N)
        idx_treino, idx_teste = indices[:tamanho_treino], indices[tamanho_treino:]
        
        X_treino_b, y_treino_b = X_raw[:, idx_treino], y_raw[idx_treino]
        X_teste_b, y_teste_b = X_raw[:, idx_teste], y_raw[idx_teste]
        
        X_treino_m, Y_treino_m = X_raw.T[idx_treino], Y_mqo[idx_treino]
        X_teste_m = X_raw.T[idx_teste]
        
        # 1. MQO
        modelo_mqo = MQOClassificador()
        modelo_mqo.fit(X_treino_m, Y_treino_m)
        pred_mqo = modelo_mqo.predict(X_teste_m)
        historico_acc["MQO"].append(np.mean(pred_mqo == y_teste_b))
        
        # 2. Gaussianos
        configs = [
            ("Gauss. Tradicional", "tradicional", 0),
            ("Cov. Iguais", "iguais", 0),
            ("Matriz Agregada", "agregada", 0),
            ("Naive Bayes", "naive_bayes", 0),
            (f"Friedman ($\lambda$={melhor_lambda:.2f})", "friedman", melhor_lambda)
        ]
        
        for nome, tipo, lbd in configs:
            modelo = ClassificadorGaussiano(tipo_modelo=tipo, lbd=lbd)
            modelo.fit(X_treino_b, y_treino_b)
            pred = modelo.predict(X_teste_b)
            historico_acc[nome].append(np.mean(pred == y_teste_b))
            
    # Tabela de Resultados no Terminal
    print("\n" + "="*65)
    print("RESULTADOS FINAIS - ESTATÍSTICA (MONTE CARLO)")
    print("="*65)
    print(f"{'Modelo':<25} | {'Média':<8} | {'Desv.Pad':<8} | {'Máx':<8} | {'Mín':<8}")
    print("-" * 65)
    
    dados_boxplot = []
    nomes_modelos = list(historico_acc.keys())
    
    for modelo in nomes_modelos:
        accs_perc = np.array(historico_acc[modelo]) * 100
        dados_boxplot.append(accs_perc)
        
        media = np.mean(accs_perc)
        std = np.std(accs_perc)
        maximo = np.max(accs_perc)
        minimo = np.min(accs_perc)
        
        print(f"{modelo:<25} | {media:>7.2f}% | {std:>7.2f}% | {maximo:>7.2f}% | {minimo:>7.2f}%")

    # --- NOVO: GERAR O GRÁFICO DE BOXPLOT ---
    plt.figure(figsize=(12, 6))
    
    # Cria o boxplot
    bp = plt.boxplot(dados_boxplot, patch_artist=True, labels=nomes_modelos)
    
    # Colore as caixas para ficar mais bonito
    cores = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6']
    for patch, cor in zip(bp['boxes'], cores):
        patch.set_facecolor(cor)
        
    plt.title(f"Distribuição de Acurácias - Simulação de Monte Carlo ({rodadas} rodadas)")
    plt.ylabel("Acurácia (%)")
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(OUTPUTS_DIR / "boxplot_montecarlo.png", dpi=300)
    plt.show()