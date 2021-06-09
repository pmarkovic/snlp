import string
import re
from collections import Counter
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
        pass

    def get_n_grams(self, tokens: List[str], n: int) -> List[Tuple[str]]:
        return []

    def get_params(self, trigram) -> None:
        pass
