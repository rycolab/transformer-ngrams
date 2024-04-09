from typing import List

import numpy as np

from ngrams.base.F import ReLU


class Layer:
    def __init__(self) -> None:
        self.W = None
        self.b = None
        self.Wʹ = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.dot(self.Wʹ, ReLU(np.dot(self.W, x) + self.b))


class MLP:
    def __init__(self, layers: List[Layer]):
        self.layers = layers

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x
