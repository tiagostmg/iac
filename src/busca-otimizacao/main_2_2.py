from pathlib import Path
import time
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

def carregar_pontos_csv(caminho_csv, grupo_id=1):
    data = np.genfromtxt(caminho_csv, delimiter=',')
    origem = data[data[:, 3] == 0][0, :3]
    pontos_regiao = data[data[:, 3] == grupo_id][:, :3]
    return np.vstack([origem, pontos_regiao])

def calcular_matriz_distancias(pontos):
    n = len(pontos)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(pontos[i] - pontos[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix

def calcular_custo(perm, dist_matrix):
    distancia = dist_matrix[0, perm[0]]
    for i in range(len(perm) - 1):
        distancia += dist_matrix[perm[i], perm[i + 1]]
    distancia += dist_matrix[perm[-1], 0]
    return distancia

def inicializar_populacao(N_pop, L):
    pop = []
    for _ in range(N_pop):
        pop.append(np.random.permutation(range(1, L + 1)))
    return pop

def selecao_torneio(pop, custos, k=3):
    indices = np.random.choice(range(len(pop)), size=k, replace=False)
    best_idx = indices[np.argmin([custos[i] for i in indices])]
    return pop[best_idx]

def crossover_dois_pontos(p1, p2):
    L = len(p1)
    idx1, idx2 = sorted(np.random.choice(range(L + 1), size=2, replace=False))
    child1 = [None] * L
    child1[idx1:idx2] = p1[idx1:idx2]
    copied_set1 = set(child1[idx1:idx2])
    p2_remaining = [x for x in p2 if x not in copied_set1]
    p2_idx = 0
    for i in range(L):
        if child1[i] is None:
            child1[i] = p2_remaining[p2_idx]
            p2_idx += 1
    child2 = [None] * L
    child2[idx1:idx2] = p2[idx1:idx2]
    copied_set2 = set(child2[idx1:idx2])
    p1_remaining = [x for x in p1 if x not in copied_set2]
    p1_idx = 0
    for i in range(L):
        if child2[i] is None:
            child2[i] = p1_remaining[p1_idx]
            p1_idx += 1
    return np.array(child1), np.array(child2)

def mutacao(individuo, prob=0.01):
    if np.random.uniform() < prob:
        idx1, idx2 = np.random.choice(range(len(individuo)), size=2, replace=False)
        individuo[idx1], individuo[idx2] = individuo[idx2], individuo[idx1]
    return individuo

def proxima_geracao(pop, dist_matrix, elitismo=True, N_e=5, p_mut=0.01):
    N_pop = len(pop)
    custos = [calcular_custo(ind, dist_matrix) for ind in pop]
    nova_pop = []
    if elitismo:
        sorted_indices = np.argsort(custos)
        for i in range(N_e):
            nova_pop.append(np.copy(pop[sorted_indices[i]]))
    while len(nova_pop) < N_pop:
        parent1 = selecao_torneio(pop, custos)
        parent2 = selecao_torneio(pop, custos)
        child1, child2 = crossover_dois_pontos(parent1, parent2)
        child1 = mutacao(child1, p_mut)
        child2 = mutacao(child2, p_mut)
        nova_pop.append(child1)
        if len(nova_pop) < N_pop:
            nova_pop.append(child2)
    return nova_pop, min(custos), np.mean(custos)

def executar_ga(dist_matrix, N_pop=100, max_gen=500, elitismo=True, N_e=5, p_mut=0.01, early_stopping=50):
    L = dist_matrix.shape[0] - 1
    pop = inicializar_populacao(N_pop, L)
    best_cost = float('inf')
    best_ind = None
    cost_history = []
    mean_history = []
    generations_no_improvement = 0
    gen_reached = max_gen
    for gen in range(max_gen):
        pop, min_c, mean_c = proxima_geracao(pop, dist_matrix, elitismo, N_e, p_mut)
        cost_history.append(min_c)
        mean_history.append(mean_c)
        if min_c < best_cost:
            best_cost = min_c
            custos = [calcular_custo(ind, dist_matrix) for ind in pop]
            best_ind = np.copy(pop[np.argmin(custos)])
            generations_no_improvement = 0
        else:
            generations_no_improvement += 1
        if early_stopping and generations_no_improvement >= early_stopping:
            gen_reached = gen + 1
            break
    return best_ind, best_cost, cost_history, mean_history, gen_reached

def plot_3d_route(pontos, best_ind, best_cost):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pontos[1:, 0], pontos[1:, 1], pontos[1:, 2], c='#1f77b4', edgecolors='k', s=60, label='Pontos de Entrega')
    ax.scatter(pontos[0, 0], pontos[0, 1], pontos[0, 2], c='#d62728', marker='*', s=250, label='Origem (Drone)')
    caminho = [0] + list(best_ind) + [0]
    xs = [pontos[i, 0] for i in caminho]
    ys = [pontos[i, 1] for i in caminho]
    zs = [pontos[i, 2] for i in caminho]
    ax.plot(xs, ys, zs, c='#2ca02c', linewidth=2.5, label='Rota Otimizada')
    ax.set_title(f"Trajetória 3D Otimizada do Drone\nCusto Total (Distância): {best_cost:.2f} m", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Eixo X", fontsize=11)
    ax.set_ylabel("Eixo Y", fontsize=11)
    ax.set_zlabel("Eixo Z", fontsize=11)
    ax.legend(fontsize=11)
    output_path = OUTPUTS_DIR / "tsp_3d_route.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Trajetória 3D salva em: {output_path}")

def plot_elitism_comparison(hist_with, hist_without):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(hist_with, color='#2ca02c', linewidth=2.5, label='Com Elitismo (N_e = 5)')
    ax.plot(hist_without, color='#d62728', linewidth=2.5, label='Sem Elitismo')
    ax.set_title("Impacto do Elitismo na Convergência do GA (Custo do Melhor Caminho)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Gerações", fontsize=12)
    ax.set_ylabel("Distância Total (m)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=11)
    output_path = OUTPUTS_DIR / "tsp_ga_elitism_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico de comparação de elitismo salvo em: {output_path}")

def plot_convergence_histogram(gens_with, gens_without):
    fig, ax = plt.subplots(figsize=(10, 6))
    all_gens = gens_with + gens_without
    min_g, max_g = min(all_gens), max(all_gens)
    bins = np.arange(min_g - 5, max_g + 10, 10)
    ax.hist(gens_with, bins=bins, alpha=0.6, color='#2ca02c', edgecolor='black', label='Com Elitismo')
    ax.hist(gens_without, bins=bins, alpha=0.6, color='#d62728', edgecolor='black', label='Sem Elitismo')
    ax.set_title("Distribuição das Gerações para Convergência (Parada Antecipada)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Geração de Parada", fontsize=12)
    ax.set_ylabel("Frequência de Ocorrências", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=11)
    output_path = OUTPUTS_DIR / "tsp_ga_convergence_histogram.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Histograma de convergência salvo em: {output_path}")

def main():
    caminho_csv = Path(__file__).resolve().parent.parent / "data" / "CaixeiroGruposGA.csv"
    pontos = carregar_pontos_csv(caminho_csv, grupo_id=1)
    Npontos = len(pontos) - 1
    print(f"Iniciando Análise do Problema do Caixeiro Viajante 3D (Drone) com {Npontos} pontos da base...")
    dist_matrix = calcular_matriz_distancias(pontos)
    N_pop = 100
    max_gen = 500
    p_mut = 0.01
    N_e = 5
    early_stopping = 50
    n_simulations = 30
    
    costs_with = []
    gens_with = []
    histories_with = []
    best_ind_with = None
    best_cost_with = float('inf')
    
    costs_without = []
    gens_without = []
    histories_without = []
    
    print(f"Executando {n_simulations} simulações COM elitismo...")
    for i in range(n_simulations):
        ind, cost, hist, _, gen = executar_ga(dist_matrix, N_pop, max_gen, True, N_e, p_mut, early_stopping)
        costs_with.append(cost)
        gens_with.append(gen)
        full_hist = hist + [cost] * (max_gen - len(hist))
        histories_with.append(full_hist)
        if cost < best_cost_with:
            best_cost_with = cost
            best_ind_with = ind
            
    print(f"Executando {n_simulations} simulações SEM elitismo...")
    for i in range(n_simulations):
        ind, cost, hist, _, gen = executar_ga(dist_matrix, N_pop, max_gen, False, N_e, p_mut, early_stopping)
        costs_without.append(cost)
        gens_without.append(gen)
        full_hist = hist + [cost] * (max_gen - len(hist))
        histories_without.append(full_hist)

    moda_gen_with, count_moda_with = Counter(gens_with).most_common(1)[0]
    moda_gen_without, count_moda_without = Counter(gens_without).most_common(1)[0]
    mean_cost_with = np.mean(costs_with)
    mean_cost_without = np.mean(costs_without)
    
    print("\n" + "="*60)
    print("RESULTADOS E ANÁLISE COMPARATIVA (ALGORITMO GENÉTICO)")
    print("="*60)
    print(f"Configuração do GA: População={N_pop}, Max Gerações={max_gen}, Mutação={p_mut*100}%")
    print(f"Pontos tridimensionais (Drone): {Npontos}")
    print("-"*60)
    print("MÉTRICAS COM ELITISMO (Ne = 5):")
    print(f"  Custo Médio (Melhor Rota): {mean_cost_with:.2f} m")
    print(f"  Melhor Custo Absoluto:    {best_cost_with:.2f} m")
    print(f"  Moda de Gerações p/ Parada: {moda_gen_with} (ocorrido {count_moda_with} vezes)")
    print(f"  Média de Gerações p/ Parada: {np.mean(gens_with):.1f}")
    print("-"*60)
    print("MÉTRICAS SEM ELITISMO:")
    print(f"  Custo Médio (Melhor Rota): {mean_cost_without:.2f} m")
    print(f"  Melhor Custo Absoluto:    {min(costs_without):.2f} m")
    print(f"  Moda de Gerações p/ Parada: {moda_gen_without} (ocorrido {count_moda_without} vezes)")
    print(f"  Média de Gerações p/ Parada: {np.mean(gens_without):.1f}")
    print("="*60)
    
    avg_hist_with = np.mean(histories_with, axis=0)
    avg_hist_without = np.mean(histories_without, axis=0)
    
    plot_3d_route(pontos, best_ind_with, best_cost_with)
    plot_elitism_comparison(avg_hist_with, avg_hist_without)
    plot_convergence_histogram(gens_with, gens_without)

if __name__ == "__main__":
    main()
