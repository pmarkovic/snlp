import operator
from nltk.wsd import lesk
from nltk.corpus import senseval
from nltk.corpus import wordnet
from nltk.corpus import stopwords

from collections import defaultdict


EXAMPLES = "examples"
WORD = "word"
SENSE = "sense"
CONTEXT = "context"
MAX_OCC = "max_occ"


def create_corpus(): #use nltk.corpus.senseval
    """
    :return: A dictionary-like structure with the corpora for
    'hard' and 'serve' with the respective maximal occurring sense
    """

    instances = ['hard.pos', 'serve.pos']
    corpus = defaultdict(dict)
    max_occ = defaultdict(dict)

    for instance in instances:
        for it in senseval.instances(instance):
            word = it.word.split('-')[0]
            context = set(pair[0] for pair in it.context)
            sense = it.senses[0]

            if EXAMPLES not in corpus[word]:
                corpus[word][EXAMPLES] = []
            corpus[word][EXAMPLES].append({SENSE: sense, CONTEXT: context})
            
            if sense not in max_occ[word]:
                max_occ[word][sense] = 0
            
            max_occ[word][sense] += 1

    for word, value in max_occ.items():
        corpus[word][MAX_OCC] = max(value.items(), key=operator.itemgetter(1))[0]
    
    return corpus

def sim_lesk(word, context, bestsense) -> str:
    """
    :param word: ambiguous word
    :param context: context words
    :param bestsense: most frequent sense (by default)
    :return: sense assigned by Lesk's algorithm
    """
    max_overlap = 0
    stop_words = set(stopwords.words('english'))

    for sense in wordnet.synsets(word):
        definition = set([w for w in sense.definition().split(' ') if not w in stop_words])
        overlap = 2*len(context.intersection(definition)) / len(context.union(definition))

        if overlap > max_overlap:
            max_overlap = overlap
            bestsense = sense.name()

    return bestsense

def get_accuracy(true, predictions):
    acc = 0.0

    for ind in range(len(true)):
        acc += int(true[ind] == predictions[ind])

    print(acc / len(true))

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
    for example in corpus[term][EXAMPLES]:
        if nltk:
            predictions.append(lesk(example[CONTEXT], term).name())
        else:
            predictions.append(sim_lesk(term, example[CONTEXT], corpus[term][MAX_OCC]))
        true.append(mapping[term][example[SENSE]])

    get_accuracy(true, predictions)