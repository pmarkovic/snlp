from os import set_blocking
import string
import re
from collections import Counter, defaultdict
from typing import List, Tuple


#TODO: Implement
def preprocess(text) -> List:
    '''
    params: text-text corpus
    return: tokens in the text
    '''

    return [token for token in re.sub(r'[^\w\s]', '', text.lower()).split(' ') if token != '']


class KneserNey:
    def __init__(self, tokens: List[str], N: int, d: float):
        '''
        params: tokens - text corpus tokens
        N - highest order of the n-grams
        d - discounting paramater
        '''
        self.tokens = tokens
        self.n = N
        self.d = d

        self.trigrams = Counter([f"{tokens[i]},{tokens[i+1]},{tokens[i+2]}" for i in range(len(tokens)-2)])
        self.bigrams = Counter([f"{tokens[i]},{tokens[i+1]}" for i in range(len(tokens)-1)])
        self.unigrams = Counter(tokens)
        self.c_trigrams_end = Counter([self._convert(key, True, False) for key in self.trigrams.keys()])
        self.c_trigrams_middle = Counter([self._convert(key, True, True) for key in self.trigrams.keys()])
        self.c_bigrams_end = Counter([self._convert(key, True, False) for key in self.bigrams.keys()])
        self.c_bigrams_all = Counter({"*,*": len(self.bigrams)})
        self.c_trigrams_start = Counter([self._convert(key, False, True) for key in self.trigrams.keys()])
        self.c_bigrams_start = Counter([self._convert(key, False, True) for key in self.bigrams.keys()])
        

    def _convert(self, s, start=False, end=False, split=','):
        words = s.split(f'{split}')

        if start:
            words[0] = '*'
        if end:
            words[-1] = '*'
        
        return ','.join(words)

    def get_n_grams(self, tokens: List[str], n: int) -> List[Tuple[str]]:
        return []

    def get_params(self, trigram) -> None:
        first = self.trigrams[self._convert(trigram, split=' ')]
        second = self.bigrams[self._convert(','.join(trigram.split(' ')[:-1]))]
        third = self.c_trigrams_end[self._convert(trigram, True, False, ' ')]
        fourth = self.c_trigrams_middle[self._convert(trigram, True, True, ' ')]
        fifth = self.c_bigrams_end[self._convert(','.join(trigram.split(' ')[1:]), True, False)]
        sixth = self.c_bigrams_all['*,*']
        seventh = self.c_trigrams_start[self._convert(trigram, False, True, ' ')]
        eight = self.c_bigrams_start[self._convert(','.join(trigram.split(' ')[1:]), False, True)]
        ninth = self.unigrams[trigram.split(' ')[1]]

        print(f"Input: {trigram}")
        print(f"N(w1w2w3): {first}")
        print(f"N(w1w2): {second}")
        print(f"N+(*w2w3): {third}")
        print(f"N+(*w2*): {fourth}")
        print(f"N+(*w3): {fifth}")
        print(f"N+(**): {sixth}")
        print(f"N+(w1w2*): {seventh}")
        print(f"N+(w2*): {eight}")
        print(f"lambda(w1w2): {self.d / second * seventh}")
        print(f"lambda(w2): {self.d / ninth * eight}")
