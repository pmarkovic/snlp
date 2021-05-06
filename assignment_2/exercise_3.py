import numpy as np
from typing import Dict, List
from collections import defaultdict


def train_test_split(corpus:List, test_size:float) -> (List, List):
    """
    Should split a corpus into a train set of size len(corpus)*(1-test_size)
    and a test set of size len(corpus)*test_size.
    :param text: the corpus, i. e. a list of strings
    :param test_size: the size of the training corpus
    :return: the train and test set of the corpus
    """
    sep_point = int(len(corpus)*(1-test_size))

    return corpus[:sep_point], corpus[sep_point:]


def relative_frequencies(tokens:List, model='unigram') -> dict:
    """
    Should compute the relative n-gram frequencies of the test set of the corpus.
    :param tokens: the tokenized test set
    :param model: the identifier of the model ('unigram, 'bigram', 'trigram')
    :return: a dictionary with the ngrams as keys and the relative frequencies as values
    """

    if model == "unigram":
        n = 0
    elif model == "bigram":
        n = 1
        tokens.append(tokens[0])
    elif model == "trigram":
        n = 2
        tokens.append(tokens[0])
        tokens.append(tokens[1])
    else:
        print("ERROR! Unsupported model!")
        
        return False

    # Calculate frequencies
    rel_freq = defaultdict(float)
    for i in range(n, len(tokens)):
        if n > 0:
            condition = ','.join(tokens[i-n:i]) 
            rel_freq[f"{tokens[i]}|{condition}"] += 1
        else:
            rel_freq[tokens[i]] += 1

    # Calculate probabilities
    num_ngrams = len(tokens) - n
    for key, value in rel_freq.items():
        rel_freq[key] = value / num_ngrams

    if model == "bigram":
        tokens.pop()
    elif model == "trigram":
        tokens.pop()
        tokens.pop()

    return rel_freq


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
