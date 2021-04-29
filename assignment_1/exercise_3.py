from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def get_frequencies(tokens):
    """
    Function returns frequencies of tokens sorted in descending order.
    ...
    
    Parameters:
    -----------
    tokens : list
        List of tokens from the corpus.
        
    Returns:
    --------
    list
        Sorted list of frequencies in descending order.
    """
    
    tokens_dict = defaultdict(float)
    
    for token in tokens:
        tokens_dict[token] += 1
    
    return np.array(sorted(tokens_dict.values(), reverse=True))


def analysis(name, data):
  """
  Plot Zipfian distribution of words + true Zipfian distribution. Compute MSE.

  :param name: title of the graph
  :param data: list of words
  """
  print(name)

  data_frequencies = get_frequencies(data)
  m = data_frequencies[0]
  ideal_zipfs = np.array([m/r for r in range(1, len(data_frequencies)+1)])
  mse = ((data_frequencies - ideal_zipfs)**2).mean()
  ranks = range(1, len(data_frequencies)+1)

  print(f"MSE: {round(mse, 10)}")

  plt.clf()
  plt.xscale("log")
  plt.yscale("log")
  plt.title(f"Zipf plot for {name}")
  plt.xlabel("rank")
  plt.ylabel("frequency")
  plt.plot(ranks, data_frequencies, "r*")
  plt.plot(ranks, ideal_zipfs, "b-")
  plt.show()
