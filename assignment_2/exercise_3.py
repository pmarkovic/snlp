import numpy as np
from typing import Dict, List


def train_test_split(corpus:List, test_size:float) -> (List, List):
    """
    Should split a corpus into a train set of size len(corpus)*(1-test_size)
    and a test set of size len(corpus)*test_size.
    :param text: the corpus, i. e. a list of strings
    :param test_size: the size of the training corpus
    :return: the train and test set of the corpus
    """
    return [], []


def relative_frequencies(tokens:List, model='unigram') -> dict:
    """
    Should compute the relative n-gram frequencies of the test set of the corpus.
    :param tokens: the tokenized test set
    :param model: the identifier of the model ('unigram, 'bigram', 'trigram')
    :return: a dictionary with the ngrams as keys and the relative frequencies as values
    """
    return {}


def pp(lm:Dict, rfs:Dict) -> float:
    """
    Should calculate the perplexity score of a language model given the relative
    frequencies derived from a test set.
    :param lm: the language model (from exercise 2)
    :param rfs: the relative frequencies
    :return: a perplexity score
    """
    return np.NINF


def plot_pps(pps:List) -> None:
    """
    Should plot perplexity value vs. language model
    :param pps: a list of perplexity scores
    :return:
    """
