from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from models.mqo import MQO
from models.media_y import MediaY

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

def read_data():
    file_path = Path(__file__).resolve().parents[1] / "data" / "aerogerador.dat"

    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {file_path}")

    data = np.loadtxt(file_path, delimiter="\t", dtype=float)


    X = data[:, :-1]
    y = data[:, -1]
    
    return (X, y)

X, y = read_data()


def save_figure(fig, filename):
    output_path = OUTPUTS_DIR / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot salvo em: {output_path}")


def plot(X, y, xlabel="Velocidade do vento", ylabel="Potência gerada", plot=True):
    fig, ax = plt.subplots()
    ax.scatter(X, y, alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)

    save_figure(fig, "dispersao_aerogerador.png")

    if plot:
        plt.show()
    else:
        plt.close(fig)
    
plot(X, y)

def train_test_split(X, y, train_size=0.8):
    n_samples = X.shape[0]
    idx = np.random.permutation(n_samples)
    split_idx = int(n_samples * train_size)

    train_idx = idx[:split_idx]
    test_idx = idx[split_idx:]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    return X_train, y_train, X_test, y_test


def random_subsampling_validation(model_factory, X, y, lbd=0.0, R=500):
    mse_scores = []
    r2_scores = []

    for _ in range(R):
        X_train, y_train, X_test, y_test = train_test_split(X, y)

        model = model_factory()
        model.fit(X_train, y_train, lbd)

        mse_scores.append(model.mse(X_test, y_test))
        r2_scores.append(model.r_squared(X_test, y_test))

    return mse_scores, r2_scores



models = [
    ("MQO tradicional", MQO,    0.0),
    ("MQO λ=0.25",      MQO,    0.25),
    ("MQO λ=0.5",       MQO,    0.5),
    ("MQO λ=0.75",      MQO,    0.75),
    ("MQO λ=1",         MQO,    1.0),
    ("Média da variável dependente", MediaY, 0.0),
]

def summarize(values):
    values = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
    }


results_by_model = {}

for nome, model_factory, lbd in models:
    mse_list, r2_list = random_subsampling_validation(model_factory, X, y, lbd)
    results_by_model[nome] = {
        "mse": {
            "values": mse_list,
            "summary": summarize(mse_list),
        },
        "r2": {
            "values": r2_list,
            "summary": summarize(r2_list),
        },
    }


for nome, model_results in results_by_model.items():
    mse_summary = model_results["mse"]["summary"]
    r2_summary = model_results["r2"]["summary"]

    print(f"\n{nome}")
    print(f"MSE -> mean: {mse_summary['mean']:.6f}, std: {mse_summary['std']:.6f}, max: {mse_summary['max']:.6f}, min: {mse_summary['min']:.6f}")
    print(f"R2  -> mean: {r2_summary['mean']:.6f}, std: {r2_summary['std']:.6f}, max: {r2_summary['max']:.6f}, min: {r2_summary['min']:.6f}")


def plot_summary(results, include_mean_model=True):
    model_names = list(results.keys())
    if not include_mean_model:
        model_names = [name for name in model_names if name != "Média da variável dependente"]

    mse_means = [results[name]["mse"]["summary"]["mean"] for name in model_names]
    mse_stds = [results[name]["mse"]["summary"]["std"] for name in model_names]
    r2_means = [results[name]["r2"]["summary"]["mean"] for name in model_names]
    r2_stds = [results[name]["r2"]["summary"]["std"] for name in model_names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(model_names, mse_means, yerr=mse_stds, capsize=4, color="#4C78A8")
    if include_mean_model:
        axes[0].set_title("MSE medio por modelo (com media)")
    else:
        axes[0].set_title("MSE medio por modelo (sem media)")
    axes[0].set_ylabel("MSE")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(model_names, r2_means, yerr=r2_stds, capsize=4, color="#F58518")
    if include_mean_model:
        axes[1].set_title("R2 medio por modelo (com media)")
    else:
        axes[1].set_title("R2 medio por modelo (sem media)")
    axes[1].set_ylabel("R2")
    axes[1].tick_params(axis="x", rotation=25)

    plt.tight_layout()
    filename = "resumo_modelos_com_media.png" if include_mean_model else "resumo_modelos_sem_media.png"
    save_figure(fig, filename)
    plt.show()


plot_summary(results_by_model, include_mean_model=True)
plot_summary(results_by_model, include_mean_model=False)
