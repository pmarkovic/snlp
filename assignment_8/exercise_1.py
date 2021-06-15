from collections import Counter
from pathlib import Path
import nltk
from nltk import RegexpTokenizer
nltk.download('reuters')
nltk.download('stopwords')
from nltk.corpus import reuters, stopwords

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List


def plot_category_frequencies(category_frequencies: Counter):
    pass

def plot_pmis(category: str, most_common: List[str], pmis: List[float]):
    pass

def plot_dfs(terms: List[str], dfs: List[int]):
    pass


class Document:
    def __init__(self, id:str, text: str, categories: str, stop_words: List[str]):
        self.id = id
        # assume only 1 category per document for simplicity
        self.category = categories[0]
        self.stop_words = stop_words


class Corpus:
    def __init__(self, documents: List[Document], categories: List[str]):
        self.documents = documents
        self.categories = set(categories)

    def df(self, term: str, category=None) -> int:
        """
        :param category: None if df is calculated over all categories, else one of the reuters.categories
        """
        raise NotImplementedError

    def pmi(self, category: str, term: str) -> float:
        raise NotImplementedError
        
    def term_frequencies(self, category) -> Counter:
        raise NotImplementedError

    def category_frequencies(self):
        raise NotImplementedError