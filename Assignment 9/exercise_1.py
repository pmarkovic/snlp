from collections import Counter
import re
from typing import Dict, List
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np

wnl = WordNetLemmatizer()

class Document:
    def __init__(self, id:str, text: str, categories: str, stop_words: List[str]):
        self.id = id
        # Determines wheter the document belongs to the train and test set
        self.section = id.split("/")[0]
        # assume only 1 category per document for simplicity
        self.category = categories[0]
        self.stop_words = stop_words
        # remove punctuations
        text = re.sub(r'[^\w\s]', ' ', text)
        # tokenize!
        tokens = nltk.word_tokenize(text.lower())
        # remove stopwords!
        tokens = [tok for tok in tokens if tok not in stop_words]
        # lemmatize
        lemmatized = [wnl.lemmatize(tok) for tok in tokens]
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

    def __len__(self):
        return len(self.documents)

    def _tf_idfs(self, document:Document, features:List[str], idfs: Dict[str, float]) -> List[float]:
        return [document.term_frequencies[term] * idfs[term] for term in features]

    def _idfs(self, features: List[str]) -> Dict[str, float]:
        self.idfs = dict.fromkeys(features, 0.0)
        total_docs = len(self.documents)
        for term, df in self.reduced_vocab.items():
            self.idfs[term] = np.log(total_docs / df)
        return self.idfs

    def _category2index(self, category:str) -> int:
        return self.categories.index(category)

    def reduce_vocab(self, min_df: int, max_df: float) -> List[str]:
        shared_vocab_doc_freq = dict.fromkeys(self.terms(), 0)
        for doc in self.documents:
            for term in doc.terms:
                shared_vocab_doc_freq[term] += 1
        self.reduced_vocab = {tm : df for tm, df in shared_vocab_doc_freq.items() if df >= min_df and df <= max_df * len(self.documents) }
        return list(sorted(self.reduced_vocab.keys()))


    def compile_dataset(self, reduced_vocab: List[str]) -> tuple:
        X_train, Y_train, X_test, Y_test = [], [], [], []
        idfs = self._idfs(reduced_vocab)
        for doc in self.documents:
            if "train" in doc.section:
                X_train.append(self._tf_idfs(doc, reduced_vocab, idfs))
                Y_train.append(self._category2index(doc.category))
            else:
                X_test.append(self._tf_idfs(doc, reduced_vocab, idfs))
                Y_test.append(self._category2index(doc.category))

        return (np.array(X_train), np.array(Y_train)), (np.array(X_test), np.array(Y_test))

    def category_frequencies(self):
        return Counter([document.category for document in self.documents])

    def terms(self):
        terms = set()
        for document in self.documents:
            terms.update(document.terms)
        return sorted(list(terms))
