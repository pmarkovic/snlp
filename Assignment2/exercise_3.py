import numpy as np
from typing import Dict, List
from collections import Counter
import matplotlib.pyplot as plt


def train_test_split(corpus:List, test_size:float) -> (List, List):
    """
    Should split a corpus into a train set of size len(corpus)*(1-test_size)
    and a test set of size len(corpus)*test_size.
    :param text: the corpus, i. e. a list of strings
    :param test_size: the size of the training corpus
    :return: the train and test set of the corpus
    """
    k = int(len(corpus) * (1-test_size))
    return corpus[:k], corpus[k:]


def relative_frequencies(tokens:List, model='unigram') -> dict:
    """
    Should compute the relative n-gram frequencies of the test set of the corpus.
    :param tokens: the tokenized test set
    :param model: the identifier of the model ('unigram, 'bigram', 'trigram')
    :return: a dictionary with the ngrams as keys and the relative frequencies as values
    """
    if model == 'unigram':
        uni_counts = Counter()
        for tok in tokens:
            uni_counts[tok] += 1
        rel_freq = { k : uni_counts[k]/len(tokens) for k in uni_counts}
        return rel_freq

    elif model == 'bigram':
        bi_counts = Counter()
        for i in range(len(tokens)-1):
            bi_counts[(tokens[i], tokens[i+1])] += 1

        bi_counts[(tokens[len(tokens)-1], tokens[0])] += 1
        rel_freq = { k : bi_counts[k] / sum(bi_counts.values()) for k in bi_counts}
        #print(sum(bi_counts.values()), len(tokens))
        return rel_freq

    else:
        tri_counts = Counter()
        for i in range(len(tokens) - 2):
            tri_counts[(tokens[i], tokens[i + 1], tokens[i + 2])] += 1

        tri_counts[(tokens[len(tokens) - 2], tokens[len(tokens) - 1], tokens[0])] += 1
        tri_counts[(tokens[len(tokens) - 1], tokens[0], tokens[1])] += 1

        rel_freq = { k :tri_counts[k] / sum(tri_counts.values()) for k in tri_counts}
        return rel_freq


def pp(lm:Dict, rfs:Dict) -> float:
    """
    Should calculate the perplexity score of a language model given the relative
    frequencies derived from a test set.
    :param lm: the language model (from exercise 2)
    :param rfs: the relative frequencies
    :return: a perplexity score
    """
    expectation = 0
    for key, rel_freq in rfs.items():
        expectation += rel_freq * np.log2(lm[key])

    return np.power(2, -(expectation))


def plot_pps(pps:List) -> None:
    """
    Should plot perplexity value vs. language model
    :param pps: a list of perplexity scores
    :return:
    """
    fig = plt.figure(figsize=(50, 30))
    plt.rcParams.update({'xtick.labelsize': 30, 'ytick.labelsize': 30})
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(["unigram", "bigram", "trigram"], pps)
    plt.show()

