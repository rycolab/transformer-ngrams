from typing import Callable, List, Sequence

import numpy as np

from ngrams.ngram.symbol import BOS, EOS, Sym


class AttentionHead:
    """A class implementing the attention mechanism."""

    def __init__(
        self,
        Q: Callable[[np.ndarray], np.ndarray],
        K: Callable[[np.ndarray], np.ndarray],
        V: Callable[[np.ndarray], np.ndarray],
        f: Callable[[np.ndarray], np.ndarray],
        projection: Callable[[np.ndarray], np.ndarray],
        O: Callable[[np.ndarray], np.ndarray],  # noqa: E741, E743
        residual: bool = False,
    ):
        self.Q = Q
        self.K = K
        self.V = V

        self.f = f
        self.projection = projection

        self.O = O  # noqa: E741, E743

        self.residual = residual

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Applies the attention mechanism to the input X, where the query is
        simply based on the last entry in X.

        Args:
            X (np.ndarray): The input to the attention mechanism.

        Returns:
            np.ndarray: The output of the attention mechanism.
        """

        Z = []

        for t in range(X.shape[0]):
            q = self.Q(X[t, :])
            K = self.K(X)
            V = self.V(X)

            T = X.shape[0]
            r = np.asarray([self.f(q, K[j, :]) for j in range(T)])
            s = self.projection(r)
            # print(f"r = {r}")
            # print(f"s = {s}")
            # print(f"X = {X}")
            # print(f"q = {q}")
            # print(f"K = {K}")
            # print(f"V = {V}")

            if self.residual:
                a = np.dot(s, V) + X[t, :]
            else:
                a = np.dot(s, V)
            # print(f"a = {a}")
            # print(f"a.shape = {a.shape}")

            if self.residual:
                z = self.O(a) + a
            else:
                z = self.O(a)

            # print(f"z = {z}")
            # print(f"z.shape = {z.shape}")
            # print()

            Z.append(z)

        return np.vstack(Z)


class MultiHeadAttentionLayer:
    """A class implementing a single layer of the Transformer network based on the
    AttentionHead mechanism class.
    """

    def __init__(
        self,
        heads: List[AttentionHead],
        fH: Callable[[np.ndarray], np.ndarray],
    ):
        self.heads = heads
        self.fH = fH

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Applies the Transformer layer to the input X.

        Args:
            X (np.ndarray): The input to the Transformer layer.

        Returns:
            np.ndarray: The output of the Transformer layer.
        """

        Zs = []
        for h, H in enumerate(self.heads):  # Iterate over the heads
            Zs.append(H(X))

        Z = self.fH(np.hstack(Zs))

        return Z


class Transformer:
    """A class implementing the Transformer network."""

    def __init__(
        self,
        layers: Sequence[MultiHeadAttentionLayer],
        F: Callable[[np.ndarray], np.ndarray],
        encoding: Callable[[str], np.ndarray],
        X0: Callable[[str], np.ndarray],
    ):
        self.layers = layers
        self.F = F
        self.encoding = encoding
        self.X0 = X0

    def __call__(self, y: str) -> np.ndarray:
        """Applies the Transformer to the input string y.

        Args:
            y (str): The input string.

        Returns:
            np.ndarray: The output of the Transformer layer.  # TODO
        """

        X = np.vstack([self.X0(yt, t + 1) for t, yt in enumerate(y)])

        print(X)

        for ll, layer in enumerate(self.layers):
            X = layer(X)

        return self.F(X).T


class TransfomerLM:
    def __init__(self, T: Transformer, E: np.ndarray, n: int):
        self.T = T
        self.E = E
        self.n = n

    def __call__(self, y: str) -> float:

        y = [BOS] * (self.n - 1) + [Sym(yt) for yt in y]

        logp = 0
        for t, yt in enumerate(y[self.n - 1 :]):
            zt = self.T(y[: self.n - 1 + t])
            # print(f"zt = {zt}")
            logpt = (self.E[:, zt.argmax()])[self.T.encoding(yt).argmax()]
            # print(f"p = {np.exp(logpt)}")
            # print()
            # print()
            logp += logpt

        zt = self.T(y)
        # print(f"zt = {zt}")
        logpEOS = (self.E[:, zt.argmax()])[self.T.encoding(EOS).argmax()]
        # print(f"pEOS = {np.exp(logpEOS)}")

        # print()
        # print()

        logp += logpEOS

        return logp
