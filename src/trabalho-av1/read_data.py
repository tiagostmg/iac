from pathlib import Path

import numpy as np


def read_aerogerador_data(path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    file_path = path or Path(__file__).resolve().parent.parent / "data" / "aerogerador.dat"

    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {file_path}")

    data = np.loadtxt(file_path, delimiter="\t", dtype=float)
    velocidade_do_vento = data[:, 0]
    potencia_gerada = data[:, 1]
    return velocidade_do_vento, potencia_gerada


if __name__ == "__main__":
    velocidade_do_vento, potencia_gerada = read_aerogerador_data()
    print(velocidade_do_vento)
    print(potencia_gerada)
