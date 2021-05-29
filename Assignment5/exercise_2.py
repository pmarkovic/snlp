from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_pp(pps: Dict):
    """ Plots perplexity vs n for all languages in the corpus

    :param pps: dictionary with langs as keys and lists of perplexity scores as values
    """
    fig = plt.figure(figsize=(4, 2))
    fig, ax = plt.subplots()
    for k, v in pps.items():
        plt.loglog(range(len(v)), v, label=k)
        ax.set_xlabel("Ngrams")
        ax.set_ylabel("Perplexity")
    plt.legend()
    plt.show()


def plot_pp_vs_alpha(pps: List[float], alphas: List[float]):
    """ Plots n-gram perplexity vs alpha
    :param pps: list of perplexity scores
    :param alphas: list of alphas
    """
    plt.plot(alphas, pps)
    plt.xlabel('Alpha values')
    plt.ylabel('Perplexity')
    plt.show()