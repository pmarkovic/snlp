# Define imports
import re
from collections import Counter
from operator import itemgetter
from typing import Union
from ast import literal_eval

import matplotlib.pyplot as plt

def preprocess(text) -> list:
    """
    : param text: The text input which you must preprocess by
    removing punctuation and special characters, lowercasing,
    and tokenising

    : return: A list of tokens
    """
    removed_punctations = re.sub(r'[^\w\s]','',text)
    return removed_punctations.lower().split()

def find_ngram_probs(tokens, unigrams=None, bigrams = None, model='unigram') :
    """
    : param tokens: Pass the tokens to calculate frequencies
    param model: the identifier of the model ('unigram, 'bigram', 'trigram')
    You may modify the remaining function signature as per your requirements

    : return: n-grams and their respective probabilities
    """
    if model == 'unigram':
        uni_counts = Counter()
        for tok in tokens:
            uni_counts[tok] += 1
        uni_prob = { k : uni_counts[k]/len(tokens) for k in uni_counts}
        return uni_prob

    elif model == 'bigram':
        bi_counts = Counter()
        for i in range(len(tokens)-1):
            bi_counts[(tokens[i], tokens[i+1])] += 1

        bi_counts[(tokens[len(tokens)-1], tokens[0])] += 1
        bi_prob = { k : (bi_counts[k] / sum(bi_counts.values())) * (1/unigrams[k[0]]) for k in bi_counts}
        return bi_prob

    else:
        tri_counts = Counter()
        for i in range(len(tokens) - 2):
            tri_counts[(tokens[i], tokens[i + 1], tokens[i + 2])] += 1

        tri_counts[(tokens[len(tokens) - 2], tokens[len(tokens) - 1], tokens[0])] += 1
        tri_counts[(tokens[len(tokens) - 1], tokens[0], tokens[1])] += 1

        tri_prob = { k : (tri_counts[k] / sum(tri_counts.values())) * (1/bigrams[(k[0], k[1])] * (1/unigrams[k[0]])) for k in tri_counts}
        return tri_prob


def plot_most_frequent(ngrams, most_freq=None) -> Union[str,tuple]:
    """
    : param ngrams: The n-grams and their probabilities
    : most_freq : most-freq unigram (str) / bigram (tuple)
    """
    if isinstance(most_freq, tuple): # For the most frequent bigram
        filtered_ngrams = {key: ngrams[key] for key in ngrams if most_freq[0] == key[0] and most_freq[1] == key[1]}
    elif isinstance(most_freq,
                    str):  # For the most frequent unigram
        filtered_ngrams = {key: ngrams[key] for key in ngrams if key[0] == most_freq}
    else:# find most freq. unigrams
        filtered_ngrams = ngrams.items()

    most_freq_ngrams = sorted(filtered_ngrams.items(), key=itemgetter(1), reverse=True)[:20]
    most_freq_ngrams = [str(w) for w, _ in most_freq_ngrams]
    if most_freq == None:
        most_freq_values = [ngrams[word] for word in most_freq_ngrams]
    else:
        most_freq_values = [ngrams[literal_eval(word)] for word in most_freq_ngrams]


    fig = plt.figure(figsize=(60, 20))
    plt.rcParams.update({'xtick.labelsize':50, 'ytick.labelsize':50})
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(most_freq_ngrams, most_freq_values)
    plt.xticks(rotation=90)
    plt.show()

    if most_freq == None:
        return most_freq_ngrams[0]
    else:
        return literal_eval(most_freq_ngrams[0])

    
                







