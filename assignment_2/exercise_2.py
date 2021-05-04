# Define imports
from collections import defaultdict
import matplotlib.pyplot as plt


def preprocess(text) -> list:
    # TODO Exercise 2.2.
    """
    : param text: The text input which you must preprocess by
    removing punctuation and special characters, lowercasing,
    and tokenising

    : return: A list of tokens
    """

    clean_text = ''.join(c for c in text.replace('\n', ' ').lower() if c.isalnum() or c == ' ')

    return clean_text.split()

def find_ngram_probs(tokens, model='unigram') -> dict:
    # TODO Exercise 2.2
    """
    : param tokens: Pass the tokens to calculate frequencies
    param model: the identifier of the model ('unigram, 'bigram', 'trigram')
    You may modify the remaining function signature as per your requirements

    : return: n-grams and their respective probabilities
    """

    if model == "unigram":
        n = 0
    elif model == "bigram":
        n = 1
        tokens.append(tokens[0])
    elif model == "trigram":
        n = 2
        tokens.append(tokens[0])
        tokens.append(tokens[1])
    else:
        print("ERROR! Unsupported model!")
        
        return False

    num_tokens = len(tokens)
    cond_words = defaultdict(int)
    probs = defaultdict(float)

    # Calculate frequencies
    for i in range(n, num_tokens):
        if n > 0:
            condition = ','.join(tokens[i-n:i]) 
            cond_words[condition] += 1
            probs[f"{tokens[i]}|{condition}"] += 1
        else:
            probs[tokens[i]] += 1

    # Calculate probabilities
    for key, value in probs.items():
        if n > 0:
            condition = key.split('|')[1]
            probs[key] = value / cond_words[condition]
        else:
            probs[key] = value / num_tokens
        
    return probs


def plot_most_frequent(ngrams, start=None) -> str:
    # TODO Exercise 2.2
    """
    : param ngrams: The n-grams and their probabilities
    param start: The starting n-gram
    Your function must find the most frequent ngrams and plot their respective probabilities

    You may modify the remaining function signature as per your requirements
    """

    if start != None:
        ngrams = {key: value for key, value in ngrams.items() if key.split('|')[1] == start}

    top_20_keys = sorted(ngrams, key=ngrams.get, reverse=True)[:20]
    top_20_values = [ngrams[key] for key in top_20_keys]


    fig = plt.figure(figsize=(20,8))
    ax = fig.add_axes([0,0,1,1])
    ax.bar(top_20_keys, top_20_values)
    plt.show()

    return ','.join(reversed(top_20_keys[0].split('|')))

if __name__ == "__main__":
    file = open("data/orient_express.txt", "r")
    text = file.read()

    tokens = preprocess(text)
    probs_uni = find_ngram_probs(tokens, model="unigram")
    start = plot_most_frequent(probs_uni, start=None)
