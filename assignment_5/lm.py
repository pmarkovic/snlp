from collections import Counter
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

    def perplexity(self, n: int):
        """ returns the perplexity of the language model for n-grams with n=n """
        raise NotImplementedError


    def lidstone_smoothing(self, alpha: float):
        """ applies lidstone smoothing on train counts

        :param alpha: the pseudo count
        :return: the smoothed counts
        """
        raise NotImplementedError      