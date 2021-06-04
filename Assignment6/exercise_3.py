from typing import List, Union, Dict
import re
import matplotlib.pyplot as plt


def preprocess(text) -> List:
    # remove all non-alphabetical characters
    removed_punctations = re.sub(r'[^\w\s]', '', text)
    return removed_punctations.lower().split()


def train_test_split_data(text: List[str], test_size=0.2):
    """ Splits the input corpus in a train and a test set
    :param text: input corpus
    :param test_size: size of the test set, in fractions of the original corpus
    :return: train and test set
    """
    k = int(len(text) * (1 - test_size))
    return text[:k], text[k:]


def k_validation_folds(text: List[str], k_folds=10):
    """ Splits a corpus into k_folds cross-validation folds
    :param text: input corpus
    :param k_folds: number of cross-validation folds
    :return: the cross-validation folds
    """
    size = int(len(text) / k_folds)
    cv_folds = []
    fold_len = []
    for i in range(k_folds):
        ind = i*size
        cv_folds.append(text[ind:ind+size])
        fold_len.append(len(text[ind:ind+size]))
    assert len(set(fold_len)) == 1
    return cv_folds


def plot_pp_vs_alpha(pps: Union[List[float], Dict] , alphas: List[float]):
    """ Plots n-gram perplexity vs alpha
    :param pps: list of perplexity scores
    :param alphas: list of alphas
    """
    fig, ax = plt.subplots()
    if isinstance(pps, List):
        plt.plot(alphas, pps)
        plt.xlabel('Alpha values')
        plt.ylabel('Perplexity')
    else:
        for k, v in pps.items():
            plt.plot(alphas, v, label=k)
            ax.set_xlabel('Alpha values')
            ax.set_ylabel('Perplexity')
        plt.legend()
    plt.show()