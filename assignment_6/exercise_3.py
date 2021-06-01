from typing import List


def train_test_split_data(text: List[str], test_size=0.2):
    """ Splits the input corpus in a train and a test set
    :param text: input corpus
    :param test_size: size of the test set, in fractions of the original corpus
    :return: train and test set
    """

def k_validation_folds(text: List[str], k_folds=10):
    """ Splits a corpus into k_folds cross-validation folds
    :param text: input corpus
    :param k_folds: number of cross-validation folds
    :return: the cross-validation folds
    """
