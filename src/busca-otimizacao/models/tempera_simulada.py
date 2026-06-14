from typing import Any, Callable, List, TypeVar, Union
import numpy as np

State = TypeVar('State')

class TemperaSimulada:
    x_best: Any
    f_best: float
    historico: List[float]
    iterations: int

    def __init__(self, T_init: float = 100.0, alpha: float = 0.99, max_iter: int = 1000, maximize: bool = False):
        self.T_init = T_init
        self.alpha = alpha
        self.max_iter = max_iter
        self.maximize = maximize
        self.x_best = None
        self.f_best = 0.0
        self.historico = []
        self.iterations = 0

    def fit(self, x_init: State, f_func: Callable[[State], float], perturbar_func: Callable[[State], State], early_stopping_val: Union[float, int, None] = None) -> 'TemperaSimulada':
        x = x_init
        f_x = f_func(x)
        x_best = np.copy(x) if isinstance(x, np.ndarray) else x
        f_best = f_x
        self.historico = [f_best]
        T = self.T_init
        self.iterations = 0
        
        for i in range(self.max_iter):
            self.iterations += 1
            if early_stopping_val is not None and f_best == early_stopping_val:
                break
            x_cand = perturbar_func(x)
            f_cand = f_func(x_cand)
            delta = f_cand - f_x
            if self.maximize:
                if delta > 0:
                    accepted = True
                else:
                    P = np.exp(delta / T)
                    accepted = P >= np.random.uniform()
            else:
                if delta < 0:
                    accepted = True
                else:
                    P = np.exp(-delta / T)
                    accepted = P >= np.random.uniform()
            if accepted:
                x = x_cand
                f_x = f_cand
                if self.maximize:
                    if f_x > f_best:
                        x_best = np.copy(x) if isinstance(x, np.ndarray) else x
                        f_best = f_x
                else:
                    if f_x < f_best:
                        x_best = np.copy(x) if isinstance(x, np.ndarray) else x
                        f_best = f_x
            self.historico.append(f_best)
            T *= self.alpha
        self.x_best = x_best
        self.f_best = f_best
        return self
