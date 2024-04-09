from itertools import product
from typing import Tuple, Union

import numpy as np

from ngrams.base.F import ProjectionFunctions, ReLU
from ngrams.base.mlp import MLP, Layer
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
        # self.Sigma = [a for a in self.A.Sigma]
        self.Sigma = list(self.A.Sigma)
        self.SigmaBOSEOS = self.Sigma + [BOS, EOS]
        self.M = len(self.SigmaBOSEOS)
        self.n = n

        self.D1 = self.M
        self.D2 = 2

        self.C1 = self.D1
        self.C2 = self.D1 + self.D2

        self.D = self.D1 + self.D2

        # The hidden states have the organization:
        # [
        #   one-hot(yt)                 # |Sigma| + 2
        #   positional encoding,        # 2
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
        self.m_inv = {i: a for a, i in self.m.items()}

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
        return np.concatenate(
            (10 ** (-(t + 1)) * self.one_hot(y), np.asarray([1, t + 1]))
        )

    def construct_head(self) -> AttentionHead:
        Wq = np.zeros((2, self.D))
        Wq[:, self.C1 : self.C2] = np.eye(2)
        bq = np.asarray([0, -(self.n - 2)])

        Wk = np.zeros((2, self.D))
        P = np.zeros((2, 2))
        P[0, 1] = -1
        P[1, 0] = 1
        Wk[:, self.C1 : self.C2] = P

        Wv = np.zeros((self.D1, self.D))
        Wv[: self.D1, : self.D1] = np.eye(self.D1)

        def Q(X):
            return (Wq @ X.T).T + bq

        def K(X):
            return (Wk @ X.T).T

        def V(X):
            return (Wv @ X.T).T

        def f(q, k):
            return -ReLU(np.dot(q, k.T))

        def O(X):  # noqa: E741, E743
            return X

        return AttentionHead(
            Q=Q,
            K=K,
            V=V,
            f=f,
            projection=ProjectionFunctions.averaging_hard,
            O=O,
        )

    def construct_layer(self):
        """Construct the parameters of the transformer block."""

        H = self.construct_head()

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

    def construct_ngram_onehot(self) -> Tuple[np.ndarray, np.ndarray]:

        Dout = self.A.num_states
        W = np.zeros((Dout, (self.n - 1) * self.M))
        b = np.zeros(Dout)

        entries = [
            self.m[a] * (self.n - 1) + j for j, a in enumerate((BOS,) * (self.n - 1))
        ]
        _w, _b = construct_and((self.n - 1) * self.M, entries)
        W[self.s[State((BOS,) * (self.n - 1))], :] = _w
        b[self.s[State((BOS,) * (self.n - 1))]] = _b

        for ll in range(self.n - 2, 0, -1):
            for ngr in product(self.Sigma, repeat=ll):
                ngr = (BOS,) * (self.n - ll - 1) + ngr
                entries = [self.m[a] * (self.n - 1) + j for j, a in enumerate(ngr)]
                _w, _b = construct_and((self.n - 1) * self.M, entries)
                W[self.s[State(ngr)], :] = _w
                b[self.s[State(ngr)]] = _b

        for ngr in product(self.Sigma, repeat=self.n - 1):
            entries = [self.m[a] * (self.n - 1) + j for j, a in enumerate(ngr)]
            _w, _b = construct_and((self.n - 1) * self.M, entries)
            W[self.s[State(ngr)], :] = _w
            b[self.s[State(ngr)]] = _b

        return W, b

    def construct_ngram_mlp(self) -> MLP:  # noqa: C901

        layers = []
        for ll in range(self.n - 1):

            W_ = np.zeros((self.n - 1 + 1, self.n - 1))
            for j in range(ll):
                W_[j, j] = 1
            for j in range(ll + 2, self.n):
                W_[j, j - 1] = 1

            w = np.zeros((self.n - 1, 1))
            for j in range(ll):
                w[j] = -(10 ** (ll + 1 - j - 1))
            w[ll] = 10 ** (ll + 1)
            W_[ll, :] = W_[ll + 1, :] = w.T

            b_ = np.zeros((self.n - 1 + 1, 1))
            b_[ll] = -1 + 10 ** (-self.n)
            b_[ll + 1] = -1

            W = np.zeros((self.M * (self.n - 1 + 1), self.M * (self.n - 1)))
            for ii in range(self.M):
                W[
                    ii * self.n : (ii + 1) * self.n,
                    ii * (self.n - 1) : (ii + 1) * (self.n - 1),
                ] = W_

            b = np.vstack([b_] * self.M).flatten()

            W_ʹ = np.zeros((self.n - 1, self.n - 1 + 1))
            for j in range(ll):
                W_ʹ[j, j] = 1
            for j in range(ll + 1, self.n - 1):
                W_ʹ[j, j + 1] = 1
            W_ʹ[ll, ll] = 10**self.n
            W_ʹ[ll, ll + 1] = -(10**self.n)
            Wʹ = np.zeros((self.M * (self.n - 1), self.M * (self.n - 1 + 1)))
            for ii in range(self.M):
                Wʹ[
                    ii * (self.n - 1) : (ii + 1) * (self.n - 1),
                    ii * self.n : (ii + 1) * self.n,
                ] = W_ʹ

            layer = Layer()
            layer.W = W
            layer.b = b
            layer.Wʹ = Wʹ

            layers.append(layer)

        return MLP(layers)

    def construct(self):
        self.set_up_orderings()

        # Set up layer:
        MAH = self.construct_layer()

        # Set up the output matrix
        E = self.construct_output_matrix()

        FZ = 0.1 * (1 - (1 / 10) ** (self.n - 1)) / (1 - 1 / 10)

        W1 = np.zeros((self.M * (self.n - 1), self.M))
        for ii in range(self.M):
            W1[ii * (self.n - 1) : (ii + 1) * (self.n - 1), ii] = 1

        Fmlp = self.construct_ngram_mlp()

        W, b = self.construct_ngram_onehot()

        def F(Z):

            a = Z[-1, :]

            # Normalize by the L1 norm and the constant factor
            a = FZ / np.sum(np.abs(a)) * a

            # Replicate the symbol positions
            a = (W1 @ a.T).T

            # Extract the digits
            a = Fmlp(a)

            # One-hot encode the n-gram
            a = ReLU((W @ a.T + b))

            return a

        T = Transformer(
            layers=[MAH],
            F=F,
            encoding=self.one_hot,
            X0=self.static_encoding,
        )

        self.lm = TransfomerLM(T, E, self.n)
