from collections import Counter, defaultdict

from typing import Dict, List

import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from tqdm.utils import _term_move_up

class Document:
    def __init__(self, id:str, text: str, categories: str, stop_words: List[str]):
        self.id = id
        # Determines wheter the document belongs to the train and test set
        self.section = id.split("/")[0]
        # assume only 1 category per document for simplicity
        self.category = categories[0]
        self.stop_words = stop_words
        
        tokens = RegexpTokenizer('\w+').tokenize(text.lower())
        tokens = [token for token in tokens if token not in self.stop_words]
        
        lemmatizer = WordNetLemmatizer()
        lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
        # count terms
        self._term_counts = Counter(lemmatized)

    def f(self, term: str) -> int:
        """ returns the frequency of a term in the document """
        return self.term_frequencies[term]

    @property
    def term_frequencies(self):
        return self._term_counts

    @property
    def terms(self):
        return set(self._term_counts.keys())


class Corpus:
    def __init__(self, documents: List[Document], categories: List[str]):
        self.documents = documents
        self.categories = sorted(list(set(categories)))
        self.idfs = defaultdict(float)

    def __len__(self):
        return len(self.documents)

    def _tf_idfs(self, document:Document, features:List[str]) -> List[float]:
        return [document.f(term) / self.idfs[term] for term in features]

    def _idfs(self, features: List[str]) -> Dict[str, float]:
        for document in self.documents:
            for term in features:
                if document.f(term) > 0:
                    self.idfs[term] += 1
        
        self.idfs = {key: len(self.documents) / value for key, value in self.idfs.items()}

        return self.idfs

    def _category2index(self, category:str) -> int:
        return self.categories.index(category)

    def reduce_vocab(self, min_df: int, max_df: float) -> List[str]:
        reduced_vocab = set()

        for document in self.documents:
            reduced_vocab.update([term for term in document.terms if min_df <= document.f(term) <= max_df])
            
        return list(sorted(reduced_vocab))

    def compile_dataset(self, reduced_vocab: List[str]) -> Dict:
        train_data, train_labels = [], []
        test_data, test_labels = [], []

        for document in self.documents:
            if document.section == "training":
                train_data.append(self._tf_idfs(document, reduced_vocab))
                train_labels.append(document.category)
            else:
                test_data.append(self._tf_idfs(document, reduced_vocab))
                test_labels.append(document.category)

        return (np.array(train_data), np.array(train_labels)), (np.array(test_data), np.array(test_labels))

    def category_frequencies(self):
        return Counter([document.category for document in self.documents])

    def terms(self):
        terms = set()
        for document in self.documents:
            terms.update(document.terms)
        return sorted(list(terms))
