from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
from models import TemperaSimulada

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

def h(x):
    attacks = 0
    for i in range(8):
        for j in range(i + 1, 8):
            if x[i] == x[j] or abs(x[i] - x[j]) == (j - i):
                attacks += 1
    return attacks

def f(x):
    return 28 - h(x)

def perturbar(x):
    x_new = list(x)
    col = np.random.randint(0, 8)
    current_val = x_new[col]
    choices = [v for v in range(1, 9) if v != current_val]
    x_new[col] = np.random.choice(choices)
    return tuple(x_new)

def plot_progress(runs_history, unique_found_history):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(runs_history, unique_found_history, color='#1f77b4', linewidth=2.5, label='Soluções Únicas Descobertas')
    ax.axhline(92, color='#d62728', linestyle='--', alpha=0.8, label='Total de Soluções Possíveis (92)')
    ax.set_title("Progresso de Descoberta das 92 Soluções Únicas (8 Rainhas)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Número de Execuções (Runs)", fontsize=12)
    ax.set_ylabel("Soluções Únicas Encontradas", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc='lower right', fontsize=11)
    output_path = OUTPUTS_DIR / "8_queens_discovery_progress.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico de progresso salva em: {output_path}")

def plot_chessboard(sol):
    fig, ax = plt.subplots(figsize=(8, 8))
    board = np.zeros((8, 8))
    board[1::2, ::2] = 1
    board[::2, 1::2] = 1
    ax.imshow(board, cmap=plt.colormaps['Oranges'], alpha=0.3)
    for col, row_1indexed in enumerate(sol):
        row = 8 - row_1indexed
        ax.text(col, row, '♛', fontsize=40, ha='center', va='center', color='#8b0000')
    ax.set_xticks(range(8))
    ax.set_xticklabels(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], fontsize=12, fontweight='bold')
    ax.set_yticks(range(8))
    ax.set_yticklabels([str(i) for i in range(8, 0, -1)], fontsize=12, fontweight='bold')
    ax.set_title(f"Exemplo de Solução: {list(sol)}", fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    ax.tick_params(which='both', length=0)
    output_path = OUTPUTS_DIR / "8_queens_chessboard.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualização do tabuleiro salva em: {output_path}")

def main():
    print("Iniciando a busca pelas 92 soluções únicas do problema das 8 Rainhas...")
    print("Utilizando Têmpera Simulada com T_init=10.0, alpha=0.95, max_iter=1000")
    
    solutions = set()
    total_runs = 0
    total_iterations = 0
    runs_history = []
    unique_found_history = []
    start_time = time.time()
    
    while len(solutions) < 92:
        total_runs += 1
        x_init = tuple(np.random.randint(1, 9, size=8))
        ts = TemperaSimulada(T_init=10.0, alpha=0.95, max_iter=1000, maximize=True)
        ts.fit(x_init, f, perturbar, early_stopping_val=28)
        x_best, f_best, iters = ts.x_best, ts.f_best, ts.iterations
        total_iterations += iters
        
        if f_best == 28:
            solutions.add(x_best)
            
        runs_history.append(total_runs)
        unique_found_history.append(len(solutions))
        
        if total_runs % 100 == 0 or len(solutions) == 92:
            print(f"Execução {total_runs:4d} | Soluções Únicas Encontradas: {len(solutions)}/92")
            
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "="*50)
    print("SUCESSO: Todas as 92 soluções distintas foram encontradas!")
    print("="*50)
    print(f"Tempo total gasto: {elapsed_time:.4f} segundos")
    print(f"Total de execuções (runs): {total_runs}")
    print(f"Total de iterações: {total_iterations}")
    print(f"Média de iterações por execução: {total_iterations / total_runs:.1f}")
    print(f"Custo computacional médio por solução única: {elapsed_time / 92 * 1000:.2f} ms")
    print("="*50)
    
    ordered_solutions = sorted(list(solutions))
    print("\nExemplo das 5 primeiras soluções encontradas:")
    for idx, sol in enumerate(ordered_solutions[:5]):
        print(f"  Solução {idx+1:02d}: {list(sol)}")
        
    plot_progress(runs_history, unique_found_history)
    plot_chessboard(ordered_solutions[0])

if __name__ == "__main__":
    main()