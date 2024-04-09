import random

import numpy as np

from ngrams.ngram.random import random_ngram
from ngrams.transform.layers import NgramTransform


def test_dfsa():
    n = 4
    for _ in range(10):
        A = random_ngram(Sigma="abcdef", n=n)

        T = NgramTransform(A, n=n)

        for _ in range(20):
            length = random.randint(1, 15)
            y = "".join(random.choices("abcdef", k=length))

            pA = A(y).value
            if pA == 0:
                continue

            logpA = np.log(A(y).value)

            logpT = T.lm(y)

            assert np.isclose(logpA, logpT, atol=1e-5)
