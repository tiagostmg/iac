from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from .models import Adaline, MultilayerPerceptron, Perceptron
except ImportError:
    from models import Adaline, MultilayerPerceptron, Perceptron


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parents[0] / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


def save_figure(fig, filename):
    output_path = OUTPUTS_DIR / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot salvo em: {output_path}")


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

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    return (
        X[:, train_idx],
        Y[:, train_idx],
        X[:, test_idx],
        Y[:, test_idx],
        train_idx,
        test_idx,
    )


def normalize_bipolar(X_train, X_test):
    xmin = X_train.min(axis=1, keepdims=True)
    xmax = X_train.max(axis=1, keepdims=True)
    scale = np.where(xmax - xmin == 0, 1, xmax - xmin)

    X_train_norm = 2 * (X_train - xmin) / scale - 1
    X_test_norm = 2 * (X_test - xmin) / scale - 1
    return X_train_norm, X_test_norm


def labels_to_bipolar_row(labels):
    labels = labels.astype(int)
    if np.min(labels) >= 1:
        labels = labels - 1
    labels = np.where(labels > 0, 1, -1)
    return labels.reshape(1, -1)


def decode_binary_targets(Y):
    return np.where(Y[0] >= 0, 1, -1)


def decode_binary_outputs(outputs):
    return np.where(outputs[0] >= 0, 1, -1)


def one_hot_bipolar(labels):
    labels = np.asarray(labels, dtype=int)
    classes = np.unique(labels)
    class_to_index = {label: index for index, label in enumerate(classes)}
    Y = -np.ones((classes.size, labels.size))

    for idx, label in enumerate(labels):
        Y[class_to_index[label], idx] = 1

    return Y, classes


def decode_multiclass_targets(Y, classes):
    indices = np.argmax(Y, axis=0)
    return classes[indices]


def decode_multiclass_outputs(outputs, classes):
    indices = np.argmax(outputs, axis=0)
    return classes[indices]


def binary_confusion_matrix(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == -1) & (y_pred == -1)))
    fp = int(np.sum((y_true == -1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == -1)))

    return np.array([
        [tn, fp],
        [fn, tp],
    ])


def binary_metrics(confusion_matrix):
    tn, fp = confusion_matrix[0]
    fn, tp = confusion_matrix[1]

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1_score = 2 * precision * sensitivity / max(precision + sensitivity, 1e-12)

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1_score,
    }


def multiclass_confusion_matrix(y_true, y_pred, classes):
    n_classes = len(classes)
    class_to_index = {label: index for index, label in enumerate(classes)}
    matrix = np.zeros((n_classes, n_classes), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        matrix[class_to_index[true_label], class_to_index[pred_label]] += 1

    return matrix


def accuracy_score(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def plot_learning_curve(history, title, filename, ylabel="Erro"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history, color="#C44E52", linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Epocas")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)
    save_figure(fig, filename)


def plot_confusion_matrix(matrix, classes, title, filename):
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicao")
    ax.set_ylabel("Classe real")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    threshold = matrix.max() / 2 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > threshold else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color)

    save_figure(fig, filename)


def plot_metric_boxplot(results_by_model, metric_name, filename):
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = list(results_by_model.keys())
    data = [results_by_model[label][metric_name]["values"] for label in labels]

    ax.boxplot(data, labels=labels)
    ax.set_title(f"Distribuicao de {metric_name}")
    ax.set_ylabel(metric_name)
    ax.grid(True, linestyle="--", alpha=0.3)
    save_figure(fig, filename)


def plot_spiral_scatter(X, y, filename="spiral_scatter.png"):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X[0, y == 1], X[1, y == 1], c="pink", edgecolors="k", alpha=0.75, label="Classe +1")
    ax.scatter(X[0, y == -1], X[1, y == -1], c="purple", edgecolors="k", alpha=0.75, label="Classe -1")
    ax.set_title("Conjunto spiral_d")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)
    save_figure(fig, filename)


def resize_image_nearest(image, new_shape):
    row_idx = np.linspace(0, image.shape[0] - 1, new_shape[0]).astype(int)
    col_idx = np.linspace(0, image.shape[1] - 1, new_shape[1]).astype(int)
    return image[row_idx][:, col_idx]


def read_spiral_data():
    file_path = DATA_DIR / "spiral_d (1).csv"
    data = np.loadtxt(file_path, delimiter=",", skiprows=1)
    X = data[:, :2].T
    Y = labels_to_bipolar_row(data[:, 2])
    return X, Y


def read_recfac_data(image_shape=(30, 30)):
    root_dir = DATA_DIR / "RecFac"
    class_dirs = sorted(path for path in root_dir.iterdir() if path.is_dir())

    features = []
    labels = []
    class_names = []

    for class_index, class_dir in enumerate(class_dirs):
        class_names.append(class_dir.name)
        for image_path in sorted(class_dir.glob("*.png")):
            image = plt.imread(image_path)
            if image.ndim == 3:
                image = image[..., :3].mean(axis=2)
            image_resized = resize_image_nearest(image, image_shape)
            features.append(image_resized.reshape(-1))
            labels.append(class_index)

    X = np.asarray(features, dtype=float).T
    labels = np.asarray(labels, dtype=int)
    Y, classes = one_hot_bipolar(labels)
    return X, Y, classes, class_names


def build_model(model_name, X_train, Y_train, config):
    if model_name == "Perceptron":
        return Perceptron(
            X_train,
            Y_train,
            learning_rate=config["learning_rate"],
            max_epochs=config["max_epochs"],
        )

    if model_name == "Adaline":
        return Adaline(
            X_train,
            Y_train,
            learning_rate=config["learning_rate"],
            precision=config["precision"],
            max_epochs=config["max_epochs"],
        )

    if model_name == "MLP":
        return MultilayerPerceptron(
            topology=config["topology"],
            X_train=X_train,
            Y_train=Y_train,
            learning_rate=config["learning_rate"],
            max_epochs=config["max_epochs"],
            precision=config["precision"],
        )

    raise ValueError(f"Modelo desconhecido: {model_name}")


def evaluate_binary_model(model_name, config, X_train, Y_train, X_test, Y_test):
    model = build_model(model_name, X_train, Y_train, config)
    model.fit()

    train_outputs = model.decision_function(X_train)
    test_outputs = model.decision_function(X_test)
    y_train_true = decode_binary_targets(Y_train)
    y_test_true = decode_binary_targets(Y_test)
    y_train_pred = decode_binary_outputs(train_outputs)
    y_test_pred = decode_binary_outputs(test_outputs)

    confusion = binary_confusion_matrix(y_test_true, y_test_pred)
    metrics = binary_metrics(confusion)
    train_metrics = binary_metrics(binary_confusion_matrix(y_train_true, y_train_pred))

    return {
        "model": model,
        "history": list(model.error_history),
        "confusion_matrix": confusion,
        "metrics": metrics,
        "train_metrics": train_metrics,
        "y_test_true": y_test_true,
        "y_test_pred": y_test_pred,
    }


def evaluate_multiclass_model(model_name, config, X_train, Y_train, X_test, Y_test, classes):
    model = build_model(model_name, X_train, Y_train, config)
    model.fit()

    train_outputs = model.decision_function(X_train)
    test_outputs = model.decision_function(X_test)
    y_train_true = decode_multiclass_targets(Y_train, classes)
    y_test_true = decode_multiclass_targets(Y_test, classes)
    y_train_pred = decode_multiclass_outputs(train_outputs, classes)
    y_test_pred = decode_multiclass_outputs(test_outputs, classes)

    return {
        "model": model,
        "history": list(model.error_history),
        "confusion_matrix": multiclass_confusion_matrix(y_test_true, y_test_pred, classes),
        "train_accuracy": accuracy_score(y_train_true, y_train_pred),
        "test_accuracy": accuracy_score(y_test_true, y_test_pred),
        "y_test_true": y_test_true,
        "y_test_pred": y_test_pred,
    }


def print_metric_summary(results_by_model, metric_names):
    for model_name, metrics_dict in results_by_model.items():
        print(f"\n{model_name}")
        for metric_name in metric_names:
            summary = metrics_dict[metric_name]["summary"]
            print(
                f"{metric_name} -> "
                f"mean: {summary['mean']:.4f}, "
                f"std: {summary['std']:.4f}, "
                f"max: {summary['max']:.4f}, "
                f"min: {summary['min']:.4f}"
            )


def etapa_1_spiral(modo_rapido=False):
    print("\n--- Etapa 1: Classificacao bidimensional com spiral_d ---")

    X, Y = read_spiral_data()
    y = decode_binary_targets(Y)
    plot_spiral_scatter(X, y)

    model_configs = {
        "Perceptron": {
            "learning_rate": 0.01,
            "max_epochs": 300,
        },
        "Adaline": {
            "learning_rate": 0.01,
            "precision": 1e-4,
            "max_epochs": 300,
        },
        "MLP": {
            "topology": [10, 6],
            "learning_rate": 0.02,
            "precision": 1e-4,
            "max_epochs": 400,
        },
    }

    mlp_study = {
        "underfitting": {
            "topology": [2],
            "learning_rate": 0.02,
            "precision": 1e-4,
            "max_epochs": 200,
        },
        "ajuste_intermediario": {
            "topology": [10, 6],
            "learning_rate": 0.02,
            "precision": 1e-4,
            "max_epochs": 400,
        },
        "overfitting": {
            "topology": [24, 18, 12],
            "learning_rate": 0.01,
            "precision": 1e-4,
            "max_epochs": 500,
        },
    }

    rng = np.random.default_rng(42)
    X_train, Y_train, X_test, Y_test, _, _ = train_test_split_columns(X, Y, rng=rng)
    X_train, X_test = normalize_bipolar(X_train, X_test)

    for study_name, config in mlp_study.items():
        result = evaluate_binary_model("MLP", config, X_train, Y_train, X_test, Y_test)
        print(
            f"MLP {study_name}: "
            f"treino={result['train_metrics']['accuracy']:.4f}, "
            f"teste={result['metrics']['accuracy']:.4f}"
        )
        plot_learning_curve(
            result["history"],
            f"Curva de aprendizado - MLP {study_name}",
            f"etapa1_mlp_{study_name}_curva.png",
            ylabel="EQM",
        )
        plot_confusion_matrix(
            result["confusion_matrix"],
            ["-1", "+1"],
            f"Matriz de confusao - MLP {study_name}",
            f"etapa1_mlp_{study_name}_matriz_confusao.png",
        )

    rodadas = 20 if modo_rapido else 500
    metric_names = ["accuracy", "sensitivity", "specificity", "precision", "f1_score"]
    monte_carlo_results = {
        model_name: {
            metric_name: {"values": [], "rounds": []}
            for metric_name in metric_names
        }
        for model_name in model_configs
    }

    for rodada in range(rodadas):
        X_train, Y_train, X_test, Y_test, _, _ = train_test_split_columns(X, Y, rng=rng)
        X_train, X_test = normalize_bipolar(X_train, X_test)

        for model_name, config in model_configs.items():
            result = evaluate_binary_model(model_name, config, X_train, Y_train, X_test, Y_test)

            for metric_name in metric_names:
                metric_value = result["metrics"][metric_name]
                monte_carlo_results[model_name][metric_name]["values"].append(metric_value)
                monte_carlo_results[model_name][metric_name]["rounds"].append({
                    "round": rodada,
                    "value": metric_value,
                    "confusion_matrix": result["confusion_matrix"],
                    "history": result["history"],
                })

    for model_name in monte_carlo_results:
        for metric_name in metric_names:
            values = monte_carlo_results[model_name][metric_name]["values"]
            monte_carlo_results[model_name][metric_name]["summary"] = summarize(values)

    print_metric_summary(monte_carlo_results, metric_names)

    for metric_name in metric_names:
        plot_metric_boxplot(
            monte_carlo_results,
            metric_name,
            f"etapa1_boxplot_{metric_name}.png",
        )

    for model_name in monte_carlo_results:
        for metric_name in metric_names:
            rounds_data = monte_carlo_results[model_name][metric_name]["rounds"]
            best_round = max(rounds_data, key=lambda item: item["value"])
            worst_round = min(rounds_data, key=lambda item: item["value"])

            plot_confusion_matrix(
                best_round["confusion_matrix"],
                ["-1", "+1"],
                f"{model_name} - melhor {metric_name}",
                f"etapa1_{model_name.lower()}_{metric_name}_melhor_matriz.png",
            )
            plot_learning_curve(
                best_round["history"],
                f"{model_name} - melhor {metric_name}",
                f"etapa1_{model_name.lower()}_{metric_name}_melhor_curva.png",
                ylabel="Erro",
            )

            plot_confusion_matrix(
                worst_round["confusion_matrix"],
                ["-1", "+1"],
                f"{model_name} - pior {metric_name}",
                f"etapa1_{model_name.lower()}_{metric_name}_pior_matriz.png",
            )
            plot_learning_curve(
                worst_round["history"],
                f"{model_name} - pior {metric_name}",
                f"etapa1_{model_name.lower()}_{metric_name}_pior_curva.png",
                ylabel="Erro",
            )

    return monte_carlo_results


def etapa_2_recfac(modo_rapido=False):
    print("\n--- Etapa 2: Reconhecimento facial com RecFac ---")

    X, Y, classes, class_names = read_recfac_data(image_shape=(30, 30))

    model_configs = {
        "Perceptron": {
            "learning_rate": 0.01,
            "max_epochs": 150,
        },
        "Adaline": {
            "learning_rate": 0.005,
            "precision": 1e-4,
            "max_epochs": 200,
        },
        "MLP": {
            "topology": [32],
            "learning_rate": 0.01,
            "precision": 1e-3,
            "max_epochs": 150,
        },
    }

    rodadas = 2 if modo_rapido else 10
    rng = np.random.default_rng(7)
    monte_carlo_results = {
        model_name: {
            "accuracy": {"values": [], "rounds": []}
        }
        for model_name in model_configs
    }

    for rodada in range(rodadas):
        X_train, Y_train, X_test, Y_test, _, _ = train_test_split_columns(X, Y, rng=rng)
        X_train, X_test = normalize_bipolar(X_train, X_test)

        for model_name, config in model_configs.items():
            result = evaluate_multiclass_model(
                model_name,
                config,
                X_train,
                Y_train,
                X_test,
                Y_test,
                classes,
            )

            monte_carlo_results[model_name]["accuracy"]["values"].append(result["test_accuracy"])
            monte_carlo_results[model_name]["accuracy"]["rounds"].append({
                "round": rodada,
                "value": result["test_accuracy"],
                "confusion_matrix": result["confusion_matrix"],
                "history": result["history"],
            })

    for model_name in monte_carlo_results:
        values = monte_carlo_results[model_name]["accuracy"]["values"]
        monte_carlo_results[model_name]["accuracy"]["summary"] = summarize(values)

    print_metric_summary(monte_carlo_results, ["accuracy"])
    plot_metric_boxplot(monte_carlo_results, "accuracy", "etapa2_boxplot_accuracy.png")

    for model_name in monte_carlo_results:
        rounds_data = monte_carlo_results[model_name]["accuracy"]["rounds"]
        best_round = max(rounds_data, key=lambda item: item["value"])
        worst_round = min(rounds_data, key=lambda item: item["value"])

        plot_confusion_matrix(
            best_round["confusion_matrix"],
            class_names,
            f"{model_name} - maior acuracia",
            f"etapa2_{model_name.lower()}_melhor_matriz.png",
        )
        plot_learning_curve(
            best_round["history"],
            f"{model_name} - maior acuracia",
            f"etapa2_{model_name.lower()}_melhor_curva.png",
            ylabel="Erro",
        )

        plot_confusion_matrix(
            worst_round["confusion_matrix"],
            class_names,
            f"{model_name} - menor acuracia",
            f"etapa2_{model_name.lower()}_pior_matriz.png",
        )
        plot_learning_curve(
            worst_round["history"],
            f"{model_name} - menor acuracia",
            f"etapa2_{model_name.lower()}_pior_curva.png",
            ylabel="Erro",
        )

    return monte_carlo_results


if __name__ == "__main__":
    PARTE_EXECUTAR = 0
    MODO_RAPIDO = True

    if PARTE_EXECUTAR == 1:
        etapa_1_spiral(modo_rapido=MODO_RAPIDO)
    elif PARTE_EXECUTAR == 2:
        etapa_2_recfac(modo_rapido=MODO_RAPIDO)
    elif PARTE_EXECUTAR == 3:
        etapa_1_spiral(modo_rapido=MODO_RAPIDO)
        etapa_2_recfac(modo_rapido=MODO_RAPIDO)
    else:
        print("Defina PARTE_EXECUTAR como 1, 2 ou 3 para executar os experimentos.")
