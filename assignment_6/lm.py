from collections import Counter, defaultdict
from copy import deepcopy
import numpy as np
from typing import List


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
        self.train_tokens = train_tokens
        self.test_tokens = test_tokens
        self.model = self.train_model()

    def train_model(self):
        train_size = len(self.train_tokens)
        model = defaultdict(float)

        for ind in range(train_size-self.N+1):
            tokens = self.train_tokens[ind:ind+self.N]
            key = tokens[-1] + '|' + ','.join(tokens[:-1])

            model[key] += 1

        for key, value in model.items():
            model[key] = (value + self.alpha) / (train_size + self.alpha*len(model))

        return model

    def perplexity(self, n: int):
        """ returns the perplexity of the language model for n-grams with n=n """
        
        expectation = 0
        rel_freq = self.get_relative_freq(n)

        for key, value in rel_freq.items():
            expectation += value * np.log2(self.lidstone_smoothing(self.alpha, key) + self.epsilon)

        return np.power(2, -expectation)


    def lidstone_smoothing(self, alpha: float):
        """ applies lidstone smoothing on train counts

        :param alpha: the pseudo count
        :return: the smoothed counts
        """
        raise NotImplementedError      