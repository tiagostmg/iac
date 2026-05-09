from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Tentativa de importação das classes das redes neurais
try:
    from .models.adaline import Adaline
    from .models.mlp import MultilayerPerceptron
    from .models.perceptron import Perceptron
    from .models.radialbasisfunction import RadialBasisFunction
except ImportError:
    from models.adaline import Adaline
    from models.mlp import MultilayerPerceptron
    from models.perceptron import Perceptron
    from models.radialbasisfunction import RadialBasisFunction

# Configurações de Caminhos
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parents[0] / "data"
OUTPUTS_DIR = BASE_DIR / "outputs pt1 edit"
OUTPUTS_DIR.mkdir(exist_ok=True)

# --- Funções Auxiliares de Utilidade ---

def save_figure(fig, filename):
    output_path = OUTPUTS_DIR / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def summarize(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
    }

def train_test_split_columns(X, Y, train_size=0.8, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n_samples = X.shape[1]
    indices = rng.permutation(n_samples)
    split_idx = int(train_size * n_samples)
    return (X[:, indices[:split_idx]], Y[:, indices[:split_idx]], 
            X[:, indices[split_idx:]], Y[:, indices[split_idx:]], 
            indices[:split_idx], indices[split_idx:])

def normalize_bipolar(X_train, X_test):
    xmin = X_train.min(axis=1, keepdims=True)
    xmax = X_train.max(axis=1, keepdims=True)
    scale = np.where(xmax - xmin == 0, 1, xmax - xmin)
    return (2 * (X_train - xmin) / scale - 1, 2 * (X_test - xmin) / scale - 1)

def labels_to_bipolar_row(labels):
    labels = labels.astype(int)
    if np.min(labels) >= 1: labels = labels - 1
    return np.where(labels > 0, 1, -1).reshape(1, -1)

def decode_binary_targets(Y): 
    return np.where(Y[0] >= 0, 1, -1)

def decode_binary_outputs(outputs): 
    return np.where(outputs[0] >= 0, 1, -1)

# --- Métricas e Plots ---

def binary_confusion_matrix(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == -1) & (y_pred == -1)))
    fp = int(np.sum((y_true == -1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == -1)))
    return np.array([[tn, fp], [fn, tp]])

def binary_metrics(matrix):
    tn, fp, fn, tp = matrix.ravel()
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-12)
    return {"accuracy": accuracy, "sensitivity": sens, "specificity": spec, "precision": prec, "f1_score": f1}

def plot_learning_curve(history, title, filename):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history, color="#C44E52")
    ax.set_title(title); ax.set_xlabel("Epocas"); ax.set_ylabel("Erro EQM"); ax.grid(True, alpha=0.3)
    save_figure(fig, filename)

def plot_confusion_matrix(matrix, classes, title, filename):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title); ax.set_xticks(np.arange(len(classes))); ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45); ax.set_yticklabels(classes)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max()/2 else "black")
    save_figure(fig, filename)

def plot_metric_boxplot(results_by_model, metric_name, filename):
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = list(results_by_model.keys())
    data = [results_by_model[l][metric_name]["values"] for l in labels]
    ax.boxplot(data, labels=labels)
    ax.set_title(f"Distribuicao de {metric_name}"); ax.grid(True, alpha=0.3)
    save_figure(fig, filename)

def plot_spiral_scatter(X, y, filename="spiral_scatter.png"):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X[0, y == 1], X[1, y == 1], c="pink", edgecolors="k", alpha=0.75, label="Classe +1")
    ax.scatter(X[0, y == -1], X[1, y == -1], c="purple", edgecolors="k", alpha=0.75, label="Classe -1")
    ax.set_title("Conjunto spiral_d")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.legend(loc="best"); ax.grid(True, linestyle="--", alpha=0.3)
    save_figure(fig, filename)

def print_metric_summary(results_by_model, metric_names):
    for model_name, metrics_dict in results_by_model.items():
        print(f"\n[{model_name}] Resumo das Métricas:")
        for metric_name in metric_names:
            summary = metrics_dict[metric_name]["summary"]
            print(f"  -> {metric_name.ljust(15)} | Média: {summary['mean']:.4f} | Desvio: {summary['std']:.4f} | Max: {summary['max']:.4f} | Min: {summary['min']:.4f}")

# --- Carregamento de Dados ---

def read_spiral_data():
    data = np.loadtxt(DATA_DIR / "spiral_d (1).csv", delimiter=",", skiprows=1)
    return data[:, :2].T, labels_to_bipolar_row(data[:, 2])

# --- Treinamento e Avaliação ---

def build_model(name, X, Y, cfg):
    if name == "Perceptron": return Perceptron(X, Y, cfg["learning_rate"], cfg["max_epochs"])
    if name == "Adaline": return Adaline(X, Y, cfg["learning_rate"], cfg["precision"], cfg["max_epochs"])
    if name == "MLP": return MultilayerPerceptron(cfg["topology"], X, Y, cfg["learning_rate"], cfg["max_epochs"], cfg["precision"])
    if name == "RBF": return RadialBasisFunction(cfg["n_neurons"], X, Y, cfg["learning_rate"], cfg["precision"], cfg["max_epochs"])

def evaluate_binary(name, cfg, Xtr, Ytr, Xte, Yte):
    m = build_model(name, Xtr, Ytr, cfg); m.fit()
    y_tr_p = decode_binary_outputs(m.decision_function(Xtr))
    y_te_p = decode_binary_outputs(m.decision_function(Xte))
    y_tr_t, y_te_t = decode_binary_targets(Ytr), decode_binary_targets(Yte)
    conf = binary_confusion_matrix(y_te_t, y_te_p)
    return {
        "history": list(m.error_history), 
        "confusion_matrix": conf, 
        "metrics": binary_metrics(conf), 
        "train_metrics": binary_metrics(binary_confusion_matrix(y_tr_t, y_tr_p))
    }

# --- Experimento Principal (Etapa 1) ---

def etapa_1_spiral():
    print("\n" + "="*60 + "\nETAPA 1: CLASSIFICACAO SPIRAL (500 RODADAS MONTE CARLO)\n" + "="*60)
    
    # 1. Leitura e plotagem inicial
    X, Y = read_spiral_data()
    y_plot = decode_binary_targets(Y)
    print("[Status] Salvando visualização inicial dos dados...")
    plot_spiral_scatter(X, y_plot)
    
    # 2. Configurações base para o Monte Carlo
    configs = {
        "Perceptron": {"learning_rate": 0.01, "max_epochs": 300},
        "Adaline": {"learning_rate": 0.01, "precision": 1e-4, "max_epochs": 300},
        "MLP": {"topology": [10, 6], "learning_rate": 0.02, "precision": 1e-4, "max_epochs": 400},
        "RBF": {"n_neurons": 15, "learning_rate": 0.01, "precision": 1e-4, "max_epochs": 300},
    }

    # 3. Configurações para estudo de complexidade
    mlp_study = {
        "underfitting": {"topology": [2], "learning_rate": 0.02, "precision": 1e-4, "max_epochs": 200},
        "ajuste_intermediario": {"topology": [10, 6], "learning_rate": 0.02, "precision": 1e-4, "max_epochs": 400},
        "overfitting": {"topology": [24, 18, 12], "learning_rate": 0.01, "precision": 1e-4, "max_epochs": 500},
    }
    rbf_study = {
        "underfitting": {"n_neurons": 2, "learning_rate": 0.02, "precision": 1e-4, "max_epochs": 200},
        "ajuste_intermediario": {"n_neurons": 15, "learning_rate": 0.02, "precision": 1e-4, "max_epochs": 400},
        "overfitting": {"n_neurons": 70, "learning_rate": 0.01, "precision": 1e-4, "max_epochs": 500},
    }

    rng = np.random.default_rng(42)
    Xtr, Ytr, Xte, Yte, _, _ = train_test_split_columns(X, Y, rng=rng)
    Xtr, Xte = normalize_bipolar(Xtr, Xte)

    # --- ESTUDOS DE COMPLEXIDADE (Overfitting/Underfitting) ---
    print("\n[Status] Realizando estudos de complexidade (MLP e RBF)...")
    
    for study_name, config in mlp_study.items():
        print(f"  -> Treinando MLP ({study_name})...")
        res = evaluate_binary("MLP", config, Xtr, Ytr, Xte, Yte)
        plot_learning_curve(res["history"], f"MLP {study_name}", f"etapa1_mlp_{study_name}_curva.png")
        plot_confusion_matrix(res["confusion_matrix"], ["-1", "+1"], f"Matriz MLP {study_name}", f"etapa1_mlp_{study_name}_matriz.png")
        
    for study_name, config in rbf_study.items():
        print(f"  -> Treinando RBF ({study_name})...")
        res = evaluate_binary("RBF", config, Xtr, Ytr, Xte, Yte)
        plot_learning_curve(res["history"], f"RBF {study_name}", f"etapa1_rbf_{study_name}_curva.png")
        plot_confusion_matrix(res["confusion_matrix"], ["-1", "+1"], f"Matriz RBF {study_name}", f"etapa1_rbf_{study_name}_matriz.png")
    
    # --- MONTE CARLO (500 Rodadas) ---
    rodadas = 500
    metrics = ["accuracy", "sensitivity", "specificity", "precision", "f1_score"]
    mc_results = {m_n: {met: {"values": [], "rounds": []} for met in metrics} for m_n in configs}

    print(f"\n[Status] Iniciando Monte Carlo ({rodadas} rodadas)...")
    for r in range(rodadas):
        print(f"  -> Rodada {r+1}/{rodadas}", end="\r")
        Xtr, Ytr, Xte, Yte, _, _ = train_test_split_columns(X, Y, rng=rng)
        Xtr, Xte = normalize_bipolar(Xtr, Xte)
        
        for m_n, cfg in configs.items():
            res = evaluate_binary(m_n, cfg, Xtr, Ytr, Xte, Yte)
            for met in metrics:
                mc_results[m_n][met]["values"].append(res["metrics"][met])
                # Salva o resultado da rodada para acharmos o melhor/pior depois
                mc_results[m_n][met]["rounds"].append({
                    "value": res["metrics"][met], 
                    "history": res["history"], 
                    "confusion_matrix": res["confusion_matrix"]
                })

    print("\n\n[Status] Gerando Relatórios, Boxplots e Matrizes Finais...")
    
    # 1. Calcula os resumos estatísticos e imprime no terminal
    for m_n in mc_results:
        for met in metrics:
            mc_results[m_n][met]["summary"] = summarize(mc_results[m_n][met]["values"])
    print_metric_summary(mc_results, metrics)

    # 2. Plota os Boxplots para cada métrica
    for met in metrics: 
        plot_metric_boxplot(mc_results, met, f"etapa1_boxplot_{met}.png")

    # 3. Salva a melhor e a pior rodada (baseada na Acurácia) de cada modelo
    for m_n in configs:
        rounds_data = mc_results[m_n]["accuracy"]["rounds"]
        best_round = max(rounds_data, key=lambda item: item["value"])
        worst_round = min(rounds_data, key=lambda item: item["value"])

        plot_confusion_matrix(best_round["confusion_matrix"], ["-1", "+1"], f"{m_n} - Maior Acurácia", f"etapa1_{m_n.lower()}_melhor_matriz.png")
        plot_learning_curve(best_round["history"], f"{m_n} - Maior Acurácia", f"etapa1_{m_n.lower()}_melhor_curva.png")
        
        plot_confusion_matrix(worst_round["confusion_matrix"], ["-1", "+1"], f"{m_n} - Menor Acurácia", f"etapa1_{m_n.lower()}_pior_matriz.png")
        plot_learning_curve(worst_round["history"], f"{m_n} - Menor Acurácia", f"etapa1_{m_n.lower()}_pior_curva.png")

if __name__ == "__main__":
    # Executa unicamente a Etapa 1
    etapa_1_spiral()
    print("\nProcessamento da Etapa 1 concluído. Verifique a pasta 'outputs pt1 edit'.") 