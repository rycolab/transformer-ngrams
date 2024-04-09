from random import sample
from typing import List

from ngrams.ngram.semiring import Real
from ngrams.ngram.symbol import EOS, Sym, to_sym


def to_compatible_string(s: str) -> List[Sym]:
    """Converts a string to a list of symbols and appends EOS.
    Due to the way the Elman network is defined, the input string
    must not contain the BOS symbol - it is assumed that this symbol has been
    read when reading the initial hidden state h0.
    Args:
        s (str): The input string

    Returns:
        List[Sym]: List of symbols.
    """
    return [to_sym(a) for a in s] + [EOS]


def sample_string(A, max_len=20) -> str:
    q, _ = list(A.I)[0]
    for i in range(10):
        s = ""
        while A.ρ[q] == Real(0):
            a, q, _ = sample([arc for arc in A.arcs(q)], 1)[0]
            s += a.value
        if len(s) < max_len:
            return s
    return None
