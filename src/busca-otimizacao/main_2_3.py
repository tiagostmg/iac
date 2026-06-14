from pathlib import Path
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from models import TemperaSimulada

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

def f_rastrigin(x):
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(np.pi * x))

def sbx_crossover(p1, p2, eta=1.0, bounds=(-5.12, 5.12)):
    n = len(p1)
    c1 = np.empty(n)
    c2 = np.empty(n)
    for i in range(n):
        if random.random() < 0.9:
            u = random.random()
            if u <= 0.5:
                beta = (2.0 * u) ** (1.0 / (eta + 1.0))
            else:
                beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
            c1[i] = 0.5 * ((1.0 + beta) * p1[i] + (1.0 - beta) * p2[i])
            c2[i] = 0.5 * ((1.0 - beta) * p1[i] + (1.0 + beta) * p2[i])
        else:
            c1[i] = p1[i]
            c2[i] = p2[i]
    c1 = np.clip(c1, bounds[0], bounds[1])
    c2 = np.clip(c2, bounds[0], bounds[1])
    return c1, c2

def gaussian_mutation(x, p_mut=0.02, sigma=0.5, bounds=(-5.12, 5.12)):
    mutated = np.copy(x)
    for i in range(len(x)):
        if random.random() < p_mut:
            mutated[i] += random.gauss(0, sigma)
    mutated = np.clip(mutated, bounds[0], bounds[1])
    return mutated

def executar_ga_nao_canonico(pop_size=100, n_dims=50, max_gen=300, eta=1.0, bounds=(-5.12, 5.12)):
    pop = [np.random.uniform(bounds[0], bounds[1], n_dims) for _ in range(pop_size)]
    best_cost = float('inf')
    best_ind = None
    history = []
    for gen in range(max_gen):
        custos = [f_rastrigin(ind) for ind in pop]
        min_idx = np.argmin(custos)
        if custos[min_idx] < best_cost:
            best_cost = custos[min_idx]
            best_ind = np.copy(pop[min_idx])
        history.append(best_cost)
        nova_pop = []
        sorted_indices = np.argsort(custos)
        nova_pop.append(np.copy(pop[sorted_indices[0]]))
        nova_pop.append(np.copy(pop[sorted_indices[1]]))
        while len(nova_pop) < pop_size:
            idx_cand1 = random.sample(range(pop_size), 3)
            p1_idx = min(idx_cand1, key=lambda idx: custos[idx])
            idx_cand2 = random.sample(range(pop_size), 3)
            p2_idx = min(idx_cand2, key=lambda idx: custos[idx])
            p1, p2 = pop[p1_idx], pop[p2_idx]
            c1, c2 = sbx_crossover(p1, p2, eta, bounds)
            c1 = gaussian_mutation(c1, p_mut=0.02, sigma=0.5, bounds=bounds)
            c2 = gaussian_mutation(c2, p_mut=0.02, sigma=0.5, bounds=bounds)
            nova_pop.append(c1)
            if len(nova_pop) < pop_size:
                nova_pop.append(c2)
        pop = nova_pop
    return best_ind, best_cost, history

def plot_pop_comparison(results):
    fig, ax = plt.subplots(figsize=(10, 6))
    for pop_size, data in results.items():
        ax.plot(data['history'], label=f'População = {pop_size} (Custo Final: {data["best_cost"]:.2f})', linewidth=2)
    ax.set_title("Convergência do GA Não-Canônico por Tamanho da População", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Gerações", fontsize=12)
    ax.set_ylabel("Valor da Função f(x) (Rastrigin)", fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    output_path = OUTPUTS_DIR / "rastrigin_ga_pop_size_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico de comparação de população salvo em: {output_path}")

def plot_comparison_ga_sa(ga_history, sa_history):
    fig, ax = plt.subplots(figsize=(10, 6))
    ga_evals = np.arange(1, len(ga_history) + 1) * 100
    sa_evals = np.arange(1, len(sa_history) + 1)
    ax.plot(ga_evals, ga_history, color='#2ca02c', linewidth=2.5, label='Algoritmo Genético Não-Canônico (N = 100)')
    ax.plot(sa_evals, sa_history, color='#d62728', linewidth=2.5, label='Têmpera Simulada (Baixo Custo de Memória)')
    ax.set_title("Comparação de Métodos: GA Não-Canônico vs Têmpera Simulada", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Número de Avaliações da Função Objetivo", fontsize=12)
    ax.set_ylabel("Melhor Custo f(x) Encontrado", fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=11)
    output_path = OUTPUTS_DIR / "rastrigin_comparison_ga_sa.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico comparativo de métodos salvo em: {output_path}")

def main():
    n_dims = 50
    max_gen = 300
    eta = 1.0
    print("Iniciando Otimização da Função Rastrigin (n = 50)...")
    print("1. Análise de tamanho da população para o GA Não-Canônico (SBX + Mutação Gaussiana)")
    pop_sizes = [20, 50, 100, 150]
    pop_results = {}
    for pop_size in pop_sizes:
        print(f"  Avaliando população tamanho: {pop_size}...")
        start_time = time.time()
        run_costs = []
        run_histories = []
        best_ind = None
        best_cost = float('inf')
        for _ in range(5):
            ind, cost, hist = executar_ga_nao_canonico(pop_size, n_dims, max_gen, eta)
            run_costs.append(cost)
            run_histories.append(hist)
            if cost < best_cost:
                best_cost = cost
                best_ind = ind
        elapsed = time.time() - start_time
        mean_cost = np.mean(run_costs)
        mean_history = np.mean(run_histories, axis=0)
        pop_results[pop_size] = {
            'best_cost': best_cost,
            'mean_cost': mean_cost,
            'time': elapsed / 5.0,
            'history': mean_history
        }
        print(f"    Custo Médio: {mean_cost:.4f} | Tempo Médio: {elapsed / 5.0:.3f} s")
    plot_pop_comparison(pop_results)
    print("\n2. Executando Têmpera Simulada (Baixo Custo de Memória)...")
    start_time = time.time()
    sa_runs_costs = []
    sa_runs_histories = []
    best_sa_cost = float('inf')
    for _ in range(5):
        x_init = np.random.uniform(-5.12, 5.12, n_dims)
        perturbar_func = lambda x: np.clip(x + np.random.normal(0, 0.05, n_dims), -5.12, 5.12)
        ts = TemperaSimulada(T_init=50.0, alpha=0.9998, max_iter=30000, maximize=False)
        ts.fit(x_init, f_rastrigin, perturbar_func)
        cost = ts.f_best
        hist = ts.historico
        sa_runs_costs.append(cost)
        sa_runs_histories.append(hist)
        if cost < best_sa_cost:
            best_sa_cost = cost
    sa_elapsed = time.time() - start_time
    mean_sa_cost = np.mean(sa_runs_costs)
    mean_sa_history = np.mean(sa_runs_histories, axis=0)
    print(f"  Custo Médio (SA): {mean_sa_cost:.4f} | Tempo Médio (SA): {sa_elapsed / 5.0:.3f} s")
    plot_comparison_ga_sa(pop_results[100]['history'], mean_sa_history)
    print("\n" + "="*70)
    print("ANÁLISE COMPARATIVA FINAL (Rastrigin n=50)")
    print("="*70)
    print(f"{'Algoritmo':<35} | {'Custo Médio':<12} | {'Tempo Médio (s)':<15}")
    print("-"*70)
    for pop_size, data in pop_results.items():
        print(f"GA Não-Canônico (População={pop_size:<3d}) | {data['mean_cost']:<12.4f} | {data['time']:<15.3f}")
    print(f"Têmpera Simulada (SA Contínua)       | {mean_sa_cost:<12.4f} | {sa_elapsed / 5.0:<15.3f}")
    print("="*70)

if __name__ == "__main__":
    main()
