import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Importação dos modelos da Etapa 1
try:
    from models.adaline import Adaline
    from models.mlp import MultilayerPerceptron
    from models.radialbasisfunction import RadialBasisFunction
except ImportError:
    print("[Aviso] Certifique-se de que a pasta 'models' está no mesmo diretório.")

# ==============================================================================
# CONFIGURAÇÕES E CAMINHOS
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent

def resolve_dataset_path() -> Path:
    path = BASE_DIR.parent / "data" / "RecFac"
    if not path.exists():
        path = BASE_DIR / "recfac"
        if not path.exists():
            raise FileNotFoundError(f"O caminho para o dataset não foi encontrado.")
    return path

DATASET_PATH = resolve_dataset_path()
OUTPUTS_DIR = BASE_DIR / "outputs_pt2"
OUTPUTS_DIR.mkdir(exist_ok=True)

IMG_SIZE = (30, 30)

# ==============================================================================
# PROCESSAMENTO DE DADOS (TÓPICOS 1, 2 e 3)
# ==============================================================================

def get_class_names(dataset_path):
    classes = [f.name for f in dataset_path.iterdir() if f.is_dir()]
    return sorted(classes)

def load_and_preprocess_faces(dataset_path, img_size):
    classes = get_class_names(dataset_path)
    X_list, Y_list = [], []
    sample_saved = False
    
    print(f"[Status] Classes detectadas: {classes}")

    for idx, class_name in enumerate(classes):
        class_folder = dataset_path / class_name
        img_files = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png"))
        
        for img_path in img_files:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, img_size)
                
                if not sample_saved:
                    sample_path = OUTPUTS_DIR / "amostra_redimensionada.png"
                    cv2.imwrite(str(sample_path), img_resized)
                    sample_saved = True
                
                vector_normalized = img_resized.flatten() / 255.0
                X_list.append(vector_normalized)
                Y_list.append(idx)

    return np.array(X_list).T, np.array(Y_list), classes

def split_data_80_20(X, Y):
    n_samples = X.shape[1]
    indices = np.random.permutation(n_samples)
    split_point = int(0.8 * n_samples)
    train_idx, test_idx = indices[:split_point], indices[split_point:]
    return X[:, train_idx], Y[train_idx], X[:, test_idx], Y[test_idx]

def to_one_hot(Y, num_classes):
    one_hot = np.full((num_classes, Y.shape[0]), -1.0) 
    for i, y in enumerate(Y):
        one_hot[y, i] = 1.0 
    return one_hot

# ==============================================================================
# FUNÇÕES DE GRÁFICOS E TABELAS
# ==============================================================================

def get_model_history(model):
    if hasattr(model, 'error_history'):
        history = getattr(model, 'error_history')
        return list(history) if history else []
    return []

def plot_confusion_matrix(cm, class_names, title, filename):
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_title(title)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_learning_curve(history, title, filename):
    plt.figure(figsize=(8, 6))
    if history and len(history) > 0:
        plt.plot(range(1, len(history) + 1), history, color="red", linewidth=2)
        plt.yscale('log')
    else:
        plt.text(0.5, 0.5, "Histórico vazio", ha='center')
    plt.title(title)
    plt.xlabel("Épocas")
    plt.ylabel("Erro (EQM)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(OUTPUTS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()

def plot_boxplot(results_dict, filename, mc_rounds):
    fig, ax = plt.subplots(figsize=(8, 6))
    data = [results_dict["Adaline"], results_dict["MLP"], results_dict["RBF"]]
    ax.boxplot(data, labels=["Adaline", "MLP", "RBF"], patch_artist=True, boxprops=dict(facecolor="lightblue"))
    ax.set_title(f"Acurácia no Reconhecimento Facial ({mc_rounds} Rodadas)")
    ax.set_ylabel("Taxa de Acerto")
    ax.set_ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(OUTPUTS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()

# ==============================================================================
# EXECUÇÃO DO TÓPICO 1 AO 6 (10 Rodadas para Matrizes e Curvas)
# ==============================================================================

def executar_topicos_1_a_6(X, Y, class_names):
    print("\n" + "="*60)
    print("EXECUTANDO TÓPICOS 1 A 6 (Curvas e Matrizes de Confusão)")
    print("="*60)
    
    num_classes = len(class_names)
    mc_rounds = 10  # Apenas 10 rodadas são suficientes para achar um bom e um mau caso
    results = {"Adaline": [], "MLP": [], "RBF": []}
    
    ada_config = {"lr": 0.01, "ep": 500, "pr": 1e-5}
    mlp_config = {"topo": [50], "lr": 0.1, "ep": 500, "pr": 1e-5}
    rbf_config = {"neu": 30, "lr": 0.1, "ep": 300, "pr": 1e-5}

    for r in range(mc_rounds):
        print(f"  -> Rodada {r+1:02d}/{mc_rounds}...", end="\r", flush=True)
        start_time = time.time()
        
        X_train, Y_train, X_test, Y_test = split_data_80_20(X, Y)
        Y_train_oh = to_one_hot(Y_train, num_classes)
        
        # Adaline
        ada = Adaline(X_train, Y_train_oh, ada_config["lr"], ada_config["pr"], ada_config["ep"])
        ada.fit()
        X_test_bias = np.vstack((-np.ones((1, X_test.shape[1])), X_test))
        ada_preds = np.argmax(ada.w @ X_test_bias, axis=0)
        results["Adaline"].append({"accuracy": np.mean(ada_preds == Y_test), "history": get_model_history(ada), "preds": ada_preds, "y_true": Y_test})

        # MLP
        mlp = MultilayerPerceptron(mlp_config["topo"], X_train, Y_train_oh, mlp_config["lr"], mlp_config["ep"], mlp_config["pr"])
        mlp.fit()
        mlp_preds = np.argmax(mlp.decision_function(X_test), axis=0)
        results["MLP"].append({"accuracy": np.mean(mlp_preds == Y_test), "history": get_model_history(mlp), "preds": mlp_preds, "y_true": Y_test})

        # RBF
        rbf = RadialBasisFunction(rbf_config["neu"], X_train, Y_train_oh, rbf_config["lr"], rbf_config["pr"], rbf_config["ep"])
        rbf.fit()
        rbf_preds = np.argmax(rbf.decision_function(X_test), axis=0)
        results["RBF"].append({"accuracy": np.mean(rbf_preds == Y_test), "history": get_model_history(rbf), "preds": rbf_preds, "y_true": Y_test})

        elapsed = time.time() - start_time
        print(f"  -> Rodada {r+1:02d}/{mc_rounds} concluída! | Tempo: {elapsed:.1f}s")

    print("\n[Status] Gerando Gráficos (Matrizes e Curvas)...")
    for model in ["Adaline", "MLP", "RBF"]:
        best = max(results[model], key=lambda x: x["accuracy"])
        worst = min(results[model], key=lambda x: x["accuracy"])
        
        def build_cm(d):
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for t, p in zip(d["y_true"], d["preds"]): cm[t, p] += 1
            return cm

        plot_confusion_matrix(build_cm(best), class_names, f"{model} Melhor Caso", f"etapa2_{model.lower()}_melhor_matriz.png")
        plot_learning_curve(best["history"], f"{model} Curva (Melhor)", f"etapa2_{model.lower()}_melhor_curva.png")
        plot_confusion_matrix(build_cm(worst), class_names, f"{model} Pior Caso", f"etapa2_{model.lower()}_pior_matriz.png")
        plot_learning_curve(worst["history"], f"{model} Curva (Pior)", f"etapa2_{model.lower()}_pior_curva.png")
    
    print("[Sucesso] Tópicos 1 a 6 finalizados. Gráficos salvos em 'outputs_pt2'.\n")

# ==============================================================================
# EXECUÇÃO DO TÓPICO 7 (100 Rodadas para Tabela Estatística)
# ==============================================================================

def executar_topico_7(X, Y, class_names):
    print("="*60)
    print("EXECUTANDO TÓPICO 7 (100 Rodadas - Estatísticas e Boxplot)")
    print("="*60)
    
    num_classes = len(class_names)
    mc_rounds = 100
    results_acc = {"Adaline": [], "MLP": [], "RBF": []}
    
    ada_config = {"lr": 0.01, "ep": 500, "pr": 1e-5}
    mlp_config = {"topo": [50], "lr": 0.1, "ep": 500, "pr": 1e-5}
    rbf_config = {"neu": 30, "lr": 0.1, "ep": 300, "pr": 1e-5}

    print(f"[Info] Iniciando {mc_rounds} rodadas. Isso pode demorar...")

    for r in range(mc_rounds):
        print(f"  -> Rodada {r+1:03d}/{mc_rounds}...", end="\r", flush=True)
        start_time = time.time()
        
        X_train, Y_train, X_test, Y_test = split_data_80_20(X, Y)
        Y_train_oh = to_one_hot(Y_train, num_classes)
        
        # Adaline
        ada = Adaline(X_train, Y_train_oh, ada_config["lr"], ada_config["pr"], ada_config["ep"])
        ada.fit()
        X_test_bias = np.vstack((-np.ones((1, X_test.shape[1])), X_test))
        acc_ada = np.mean(np.argmax(ada.w @ X_test_bias, axis=0) == Y_test)
        results_acc["Adaline"].append(acc_ada)

        # MLP
        mlp = MultilayerPerceptron(mlp_config["topo"], X_train, Y_train_oh, mlp_config["lr"], mlp_config["ep"], mlp_config["pr"])
        mlp.fit()
        acc_mlp = np.mean(np.argmax(mlp.decision_function(X_test), axis=0) == Y_test)
        results_acc["MLP"].append(acc_mlp)

        # RBF
        rbf = RadialBasisFunction(rbf_config["neu"], X_train, Y_train_oh, rbf_config["lr"], rbf_config["pr"], rbf_config["ep"])
        rbf.fit()
        acc_rbf = np.mean(np.argmax(rbf.decision_function(X_test), axis=0) == Y_test)
        results_acc["RBF"].append(acc_rbf)

        elapsed = time.time() - start_time
        print(f"  -> Rodada {r+1:03d}/{mc_rounds} concluída! | Ada: {acc_ada:.2%} | MLP: {acc_mlp:.2%} | RBF: {acc_rbf:.2%} | Tempo: {elapsed:.1f}s")

    print("\n[Status] Gerando Tabela e Boxplot (Tópico 7)...")
    
    tabela_texto = "Modelos\t\t\t\tMédia\t\tDesvio-Padrão\tMaior Valor\tMenor Valor\n"
    tabela_texto += "-"*85 + "\n"

    for model in ["Adaline", "MLP", "RBF"]:
        media = np.mean(results_acc[model])
        std = np.std(results_acc[model])
        maior = np.max(results_acc[model])
        menor = np.min(results_acc[model])
        
        nome_formatado = "ADAptive LINear Element." if model == "Adaline" else ("Perceptron de Múltiplas Camadas" if model == "MLP" else "Rede RBF")
        tabela_texto += f"{nome_formatado:<30}\t{media:.4f}\t\t{std:.4f}\t\t{maior:.4f}\t\t{menor:.4f}\n"

    print("\n" + tabela_texto)
    
    with open(OUTPUTS_DIR / "tabela_topico7.txt", "w", encoding="utf-8") as f:
        f.write(tabela_texto)

    plot_boxplot(results_acc, "etapa2_boxplot_topico7.png", mc_rounds)
    print(f"[Sucesso] Tabela salva em: {OUTPUTS_DIR / 'tabela_topico7.txt'}")
    print("[Sucesso] Tópico 7 finalizado.\n")

# ==============================================================================
# MAIN
# ==============================================================================

def main_pt2():
    print("\n" + "*"*60)
    print(" INICIANDO ETAPA 2: RECONHECIMENTO FACIAL")
    print("*"*60 + "\n")

    # Lê as imagens apenas uma vez!
    X, Y, class_names = load_and_preprocess_faces(DATASET_PATH, IMG_SIZE)
    
    # Executa a primeira parte do trabalho (10 rodadas)
    executar_topicos_1_a_6(X, Y, class_names)
    
    # Executa a segunda parte do trabalho (100 rodadas focadas em estatística)
    executar_topico_7(X, Y, class_names)

if __name__ == "__main__":
    main_pt2()