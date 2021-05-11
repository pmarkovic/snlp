from typing import List


def dkl(P:List[float], Q:List[float]) -> float:
    """
    Calculates the Kullback-Leibler-Divergence of two probability distributions
    :param P: the ground truth distribution
    :param Q: the estimated distribution
    :return: the DKL of the two distribution, in bits
    """