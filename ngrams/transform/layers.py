from itertools import product
from typing import Union

import numpy as np

from ngrams.base.F import ProjectionFunctions, ReLU
from ngrams.base.modules import construct_and
from ngrams.base.transformer import (
    AttentionHead,
    MultiHeadAttentionLayer,
    TransfomerLM,
    Transformer,
)
from ngrams.ngram.fsa import FSA
from ngrams.ngram.state import State
from ngrams.ngram.symbol import BOS, EOS, Sym


class NgramTransform:
    def __init__(self, A: FSA, n: int) -> None:
        """The class that constructs a transformer network simulating a probabilistic
        finite state automaton.

        Args:
            A (FSA): The PFSA to be transformed.
        """
        assert A.probabilistic, "The FSA must be probabilistic."

        self.A = A
        self.q0 = list(self.A.I)[0][0]
        self.Sigma = list(self.A.Sigma)
        self.SigmaBOSEOS = self.Sigma + [BOS, EOS]
        self.M = len(self.SigmaBOSEOS)
        self.n = n

        self.D1 = (self.n - 1) * self.M
        self.D2 = 2 + 2

        self.C1 = self.D1
        self.C2 = self.D1 + self.D2

        self.D = self.D1 + self.D2

        # The hidden states have the organization:
        # [
        #   one-hot(yt)                 # |Sigma| + 2
        #   ...
        #   one-hot(yt)                 # |Sigma| + 2
        #   positional encoding,        # 2 + 2
        # ]

        self.construct()

    def display_hidden_state(self, X: np.ndarray) -> None:
        print()
        for i, x in enumerate(X):
            for j in range(self.M):
                if x[j] == 1:
                    print(f"y_{i}: {self.SigmaBOSEOS[j].value}")
                    break
            # print(f"p_{i}: {int(x[self.C1 + 1])}")

    def set_up_orderings(self):

        # Ordering of Σ
        self.m = {a: i for i, a in enumerate(self.SigmaBOSEOS)}
        self.m_inv = {i: a for i, a in self.m.items()}

        # Ordering of Q
        self.s = {q: i for i, q in enumerate(self.A.Q)}
        self.s_inv = {i: q for i, q in enumerate(self.A.Q)}

    def one_hot(self, x: Union[State, Sym]) -> np.ndarray:
        if isinstance(x, Sym):
            y = np.zeros(self.M)
            y[self.m[x]] = 1
            return y
        elif isinstance(x, State):
            y = np.zeros((self.A.num_states))
            y[self.s[x]] = 1
            return y
        else:
            raise TypeError

    def ey2y(self, x: np.ndarray) -> str:
        return self.m_inv[np.argmax(x[: self.C1])]

    def static_encoding(self, y: str, t: int) -> np.ndarray:

        pos = np.hstack(
            [
                [1 / np.sqrt(t), np.sqrt(1 - 1 / t)],
                [1 / np.sqrt(t + 1), np.sqrt(1 - 1 / (t + 1))],
            ]
        )

        X0 = np.concatenate([self.one_hot(y), np.zeros(self.M * (self.n - 2)), pos])

        return X0

    def construct_head(self, ll: int) -> AttentionHead:
        Wq = np.zeros((2, self.D))
        Wq[:, self.C1 : self.C1 + 2] = np.eye(2)

        Wk = np.zeros((2, self.D))
        Wk[:, self.C1 + 2 : self.C1 + 4] = np.eye(2)

        Wv = np.zeros((self.D, self.D))
        Wv[ll * self.M : (ll + 1) * self.M, (ll - 1) * self.M : ll * self.M] = np.eye(
            self.M
        )

        def Q(X):
            return (Wq @ X.T).T

        def K(X):
            return (Wk @ X.T).T

        def V(X):
            return (Wv @ X.T).T

        def f(q, k):
            return np.dot(q, k.T)

        Wo = np.zeros((self.D, self.D))

        def O(X):  # noqa: E741, E743
            return (Wo @ X.T).T

        return AttentionHead(
            Q=Q,
            K=K,
            V=V,
            f=f,
            projection=ProjectionFunctions.unique_hard,
            O=O,
            residual=True,
        )

    def construct_layer(self, ll: int):
        """Construct the parameters of the transformer block."""

        H = self.construct_head(ll)

        def fH(Z):
            return Z

        return MultiHeadAttentionLayer(heads=[H], fH=fH)

    def construct_output_matrix(self):
        E = -np.inf * np.ones((self.M, self.A.num_states))

        for q in self.A.Q:
            for a, _, w in self.A.arcs(q):
                E[self.m[a], self.s[q]] = np.log(w.value)

        for q, w in self.A.F:
            E[self.m[EOS], self.s[q]] = np.log(w.value)

        return E

    def construct(self):
        self.set_up_orderings()

        # Set up layer:
        MAH = [self.construct_layer(ll) for ll in range(1, self.n - 1)]
        # MAH = [self.construct_layer(ll) for ll in range(self.n - 1, 0, -1)]

        # Set up the output matrix
        E = self.construct_output_matrix()

        Dout = self.A.num_states
        W = np.zeros((Dout, self.D))
        b = np.zeros(Dout)

        entries = [j * self.M + self.m[a] for j, a in enumerate((BOS,) * (self.n - 1))]
        _w, _b = construct_and(self.D, entries)
        # print(f"ngr = {(BOS,) * (self.n - 1)}")
        # print(f"entries = {entries}")
        # print(f"_w = {_w[:, :-4].flatten()}")
        # print(f"_b = {_b}")
        W[self.s[State((BOS,) * (self.n - 1))], :] = _w
        b[self.s[State((BOS,) * (self.n - 1))]] = _b

        for ll in range(self.n - 2, 0, -1):
            # for ll in range(self.n - 1):
            for ngr in product(self.Sigma, repeat=ll):
                ngr = (BOS,) * (self.n - ll - 1) + ngr
                entries = [j * self.M + self.m[a] for j, a in enumerate(reversed(ngr))]
                _w, _b = construct_and(self.D, entries)
                # print(f"ngr = {ngr}")
                # print(f"entries = {entries}")
                # print(f"_w = {_w[:, :-4].flatten()}")
                # print(f"_b = {_b}")
                W[self.s[State(ngr)], :] = _w
                b[self.s[State(ngr)]] = _b

        for ngr in product(self.Sigma, repeat=self.n - 1):
            entries = [j * self.M + self.m[a] for j, a in enumerate(reversed(ngr))]
            _w, _b = construct_and(self.D, entries)
            # print(f"ngr = {ngr}")
            # print(f"entries = {entries}")
            # print(f"_w = {_w[:, :-4].flatten()}")
            # print(f"_b = {_b}")
            W[self.s[State(ngr)], :] = _w
            b[self.s[State(ngr)]] = _b

        # print()
        # print()
        # print()

        def F(Z):
            # for i in range(self.n - 1):
            # print(Z[-1, i * self.M : (i + 1) * self.M])
            # print()
            # print(Z[-1, :-4])
            return ReLU(Z[-1, :] @ W.T + b)

        T = Transformer(
            layers=MAH,
            F=F,
            encoding=self.one_hot,
            X0=self.static_encoding,
        )

        self.lm = TransfomerLM(T, E, self.n)
