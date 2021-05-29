from collections import Counter, defaultdict
from copy import deepcopy
import numpy as np
from typing import List


def get_bigrams(corpus):
    bigrams = []
    for k, node in corpus.items():
        for bigram, _ in node.children.items():
            bigrams.append(bigram)
    return bigrams


def get_trigrams(corpus):
    trigrams = []
    for _, node in corpus.items():
        for _, bigram in node.children.items():
            for trigram, _ in bigram.children.items():
                trigrams.append(trigram)
    return trigrams


class NGramTreeNode:
    def __init__(self, word):
        self.children = dict()
        self.word = word
        self.count = 0

    def insert(self, leading_words):
        node = self
        leading_words = leading_words.split(",")
        for wd in leading_words:
            gram = f"{node.word},{wd}"
            if gram not in node.children:
                node.children[gram] = NGramTreeNode(gram)
            node = node.children[gram]

    def set_count(self, gram, count):
        node = self
        for wd in gram.split(","):
            gm = f"{node.word},{wd}"
            node = node.children[gm]
        node.count = count

    def get_count(self, words):
        node = self
        if words == node.word: return node.count, None
        parent_count = 0
        for k, child in node.children.items():
            if k in words:
                if k == words: return child.count, node.count
                else:
                    parent_count = node.count
                    if k in node.children:
                        node = node.children[k]
                    else: break
        return 0, parent_count


class LanguageModel:
    
    def __init__(self, train_tokens: List[str], test_tokens: List[str], N: int, alpha: float, epsilon=1.e-10):
        """ 
        :param train_tokens: list of tokens from the train section of your corpus
        :param test_tokens: list of tokens from the test section of your corpus
        :param N: n of the highest-order n-gram model
        :param alpha: pseudo count for lidstone smoothing
        :param epsilon: threshold for probability mass loss, defaults to 1.e-10
        """
        self.N = N
        self.alpha = alpha
        self.train = train_tokens
        self.test = test_tokens
        self.ngram_train = self.build_ngram_tree(train_tokens)
        self.ngram_test = self.build_ngram_tree(test_tokens)
        self.shared = self.build_ngram_tree(train_tokens+test_tokens)
        self.shared_bigram_types = len(set(self.get_ngrams(2)))
        self.shared_trigram_types = len(set(self.get_ngrams(3)))
        self.epsilon = epsilon

    def perplexity(self, n: int):
        """ returns the perplexity of the language model for n-grams with n=n """
        expectation = 0
        rel_freq = self.get_relative_freq(n)

        for key, value in rel_freq.items():
            expectation += value * np.log2(self.lidstone_smoothing(self.alpha, key) + self.epsilon)

        return np.power(2, -expectation)

    def lidstone_smoothing(self, alpha: float, key: str):
        """ applies lidstone smoothing on train counts

        :param alpha: the pseudo count
        :return: the smoothed counts
        """
        n = len(key.split(","))
        start = key.split(",")[0]
        if start not in self.ngram_train:
            node_count, parent_count = 0, 0
        else:
            node_count, parent_count = self.ngram_train[start].get_count(key)
        if n == 1:
            smoothed_prob = node_count + alpha / (len(self.train) + alpha * len(set(self.shared)))
        elif n == 2:
            smoothed_prob = (node_count + alpha) / (parent_count + alpha * self.shared_bigram_types)
        elif n == 3:
            smoothed_prob = (node_count + alpha) / (parent_count + alpha * self.shared_trigram_types)
        else:
            raise ValueError(f"value of N should lie between [1,3]")

        return smoothed_prob

    def get_ngrams(self, n, corpus=None):
        if corpus == None:
            corpus = self.shared
        if n == 1:
            return [key for key in corpus]
        elif n == 2:
            return get_bigrams(corpus)
        elif n == 3:
            return get_trigrams(corpus)

    def get_relative_freq(self, n):
        rel_freq = {}
        ngrams = self.get_ngrams(n, corpus=self.ngram_test)
        ngrams = sorted(ngrams)
        start_ngrams = defaultdict(float)
        for key in ngrams:
            start = key.split(",")[0]
            count, _ = self.ngram_test[start].get_count(key)
            rel_freq[key] = count / len(ngrams)
            start_ngrams[start] += rel_freq[key]
        return rel_freq

    def build_ngram_tree(self, tokens):
        ngrams = dict() # condition = { to : {Node(to) : Node(to, be)}, }

        for n in range(self.N):

            num_tokens = len(tokens)
            counts = defaultdict(int)

            for i in range(n): # circular corpus
                tokens.append(tokens[i])

            # Calculate frequencies
            for i in range(0, num_tokens-n):
                if n > 0:
                    start = tokens[i]
                    gram = ",".join(tokens[i+1:i+n])
                    counts[f"{start}-{gram}"] += 1
                    ngrams[start].insert(gram)
                else:
                    ngrams[tokens[i]] = NGramTreeNode(tokens[i])
                    counts[tokens[i]] += 1

            # set counts
            for key, value in counts.items():
                wd = key.split('-')[0]
                if n > 0:
                    gram = key.split('-')[1]
                    ngrams[wd].set_count(gram, value)
                else:
                    ngrams[wd].count = value

            for i in range(n):
                tokens.pop()

        return ngrams




