from typing import List, Dict
import string
from collections import Counter
import matplotlib.pyplot as plt


def preprocess(text, lang) -> List:
    # lower text, remove punctuation and split the text into words based on whitespace
    text = text.replace("\n", " ").lower()
    punctuation_replacer = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    tokens = text.translate(punctuation_replacer).split()
    # if lang != "en":
    #     tokens = [tok for tok in tokens if not tok.isalpha()]
    return tokens


def train_test_split_data(text:List, test_size:float=0.1):
    k = int(len(text) * (1 - test_size))
    return text[:k], text[k:]


def get_oov_rates(train:List, test:List) -> List:
    word_freq = Counter(train)
    oov_rates = []
    #vocab = dict()
    # create vocabularies of sizes 1k,2k,..15k
    for i in range(1000, 16000, 1000):
        vocab = [tup[0] for tup in word_freq.most_common(i)]
        unseen_tokens = [tok for tok in test if tok not in vocab]
        oov_rates.append(len(unseen_tokens)/len(test))
    return oov_rates


def plot_oov_rates(oov_rates:Dict) -> None:
    fig, ax = plt.subplots()
    for k, v in oov_rates.items():
        plt.loglog(range(1000, 16000, 1000), v, label=k)
        ax.set_xlabel("vocab size")
        ax.set_ylabel("OOV rate")
    plt.legend()
    plt.show()