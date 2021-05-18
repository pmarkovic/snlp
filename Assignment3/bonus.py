from typing import List
from collections import Counter
from operator import itemgetter
import numpy as np

def find_ngram_probs(text):
    tokens = text.split()
    uni_counts = Counter()
    for tok in tokens:
        uni_counts[tok] += 1
    uni_prob = {k: uni_counts[k] / len(tokens) for k in uni_counts}
    uni_dist = sorted(uni_prob.items(), key=itemgetter(1), reverse=True)
    uni_dist = [tup[1] for tup in uni_dist]
    return uni_dist

def dkl(P:List[float], Q:List[float]) -> float:
    """
    Calculates the Kullback-Leibler-Divergence of two probability distributions
    :param P: the ground truth distribution
    :param Q: the estimated distribution
    :return: the DKL of the two distribution, in bits
    """
    dkl = 0
    for p_prob, q_prob in zip(P, Q):
        dkl += p_prob * np.log2(p_prob/q_prob)

    return dkl
