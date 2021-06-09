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
        self.c_trigrams_end = Counter([self._convert(key, True, False) for key in self.trigrams.keys()])
        self.c_trigrams_middle = Counter([self._convert(key, True, True) for key in self.trigrams.keys()])
        self.c_bigrams_end = Counter([self._convert(key, True, False) for key in self.bigrams.keys()])
        self.c_bigrams_all = Counter({"*,*": len(self.bigrams)})
        self.c_trigrams_start = Counter([self._convert(key, False, True) for key in self.trigrams.keys()])
        self.c_bigrams_start = Counter([self._convert(key, False, True) for key in self.bigrams.keys()])
        

    def _convert(self, s, start=False, end=False):
        words = s.split(',')

        if start:
            words[0] = '*'
        if end:
            words[-1] = '*'
        
        return ','.join(words)

    def get_n_grams(self, tokens: List[str], n: int) -> List[Tuple[str]]:
        return []

    def get_params(self, trigram) -> None:
        print(f"Input: {trigram}")
        print(f"N(w1w2w3)")


if __name__ == "__main__":
    file = open("data/alice_in_wonderland.txt", "r")
    text = file.read()

    # TODO: Preprocess text
    tokens = preprocess(text)

    model = KneserNey(tokens, 3, 0.75)
    print(model.c_bigrams_start.most_common(10))