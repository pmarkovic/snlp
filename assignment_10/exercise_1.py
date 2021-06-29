from nltk.corpus import senseval
from nltk.corpus import wordnet
from nltk.corpus import stopwords

def create_corpus(): #use nltk.corpus.senseval
    """
    :return: A dictionary-like structure with the corpora for
    'hard' and 'serve' with the respective maximal occurring sense
    """
    return {}

def sim_lesk(word, context, bestsense) -> str:
    """
    :param word: ambiguous word
    :param context: context words
    :param bestsense: most frequent sense (by default)
    :return: sense assigned by Lesk's algorithm
    """
    return bestsense

def get_accuracy(true, predictions):
    pass

def run_lesk(corpus, term, mapping, nltk=False):
    """
    :param corpus: corpus containing terms and contexts
    :param term: ambiguous word
    :param mapping: dictionary mapping Senseval senses to WordNet senses
    :param nltk: bool stating usage of nltk's Lesk implementation
    :return: None, or accuracy predictions
    """
    true = []
    predictions = []
    #TODO: Write code to iterate over sentences and get best sense
    get_accuracy(true, predictions)