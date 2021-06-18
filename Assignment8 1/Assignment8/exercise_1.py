from collections import Counter
from pathlib import Path
import nltk
from nltk import RegexpTokenizer
from nltk.util import pr
nltk.download('reuters')
nltk.download('stopwords')
from nltk.corpus import reuters, stopwords

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List


def plot_category_frequencies(category_frequencies: Counter):
    x, y = [], []

    for ind, pair in enumerate(category_frequencies.most_common()):
        x.append(ind)
        y.append(pair[1])

    plt.loglog(x, y)
    plt.xlabel("Categories")
    plt.ylabel("Frequencies")

def plot_pmis(category: str, most_common: List[str], pmis: List[float]):
    plt.title = f"Category: {category}"
    plt.plot(most_common, pmis)

def plot_dfs(terms: List[str], dfs: List[int]):
    plt.plot(terms, dfs)


class Document:
    def __init__(self, id:str, text: str, categories: str, stop_words: List[str]):
        self.id = id
        # assume only 1 category per document for simplicity
        self.category = categories[0]
        self.stop_words = stop_words
        self.tokens = [word.lower().strip() for word in text.split(' ') 
                                                if word.lower() not in stop_words and word != '']


class Corpus:
    def __init__(self, documents: List[Document], categories: List[str]):
        self.documents = documents
        self.categories = set(categories)

    def df(self, term: str, category=None) -> int:
        """
        :param category: None if df is calculated over all categories, else one of the reuters.categories
        """
        numerator = 0
        denominator = 0

        if category is not None:
            denominator = self.category_freq[category]

            for document in self.documents:
                if document.category == category and term in document.tokens:
                    numerator += 1
        else:
            denominator = sum(self.category_freq.values())

            for document in self.documents:
                if term in document.tokens:
                    numerator += 1

        return numerator / denominator

    def pmi(self, category: str, term: str) -> float:
        p_category = self.category_freq[category] / len(self.documents)
        p_term = 0.0
        p_joint = 0.0

        for document in self.documents:
            if term in document.tokens:
                p_term += 1

                if category == document.category:
                    p_joint += 1
        
        p_term /= len(self.documents)
        p_joint /= len(self.documents)

        return np.log2(p_joint / (p_category * p_term))
        
    def term_frequencies(self, category) -> Counter:
        terms = []
        for document in self.documents:
            if document.category == category:
                terms += set(document.tokens)
        
        return Counter(terms)

    def category_frequencies(self):
        self.category_freq = Counter([document.category for document in self.documents])

        return self.category_freq