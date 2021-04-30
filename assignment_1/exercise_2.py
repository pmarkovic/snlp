import numpy as np
from numpy.random import Generator, PCG64
import matplotlib.pyplot as plt

G = Generator(PCG64())

def stick_breaking(n:int,alpha:float) -> np.ndarray:
  """
  Draws n samples from a stick-breaking process with beta distribution intensity alpha.

  :param n: number of samples
  :param alpha: intensity parameter of the beta distribution
  :returns: stick lengths
  """
  betas = G.beta(a=alpha, b=1.0, size=n)
  betas[1:] *= np.cumprod(1-betas[:-1])
  weights = np.sort(betas)[::-1]
  return weights

def plot_stick_lengths(stick_lengths:np.array,
		       alpha:float,
		       B:int) -> None:
  """
  Plots -log2(sticks)
  :param sticks: list of stick lenghts
  """

  rf = [(r+1, f) for r, f in enumerate(stick_lengths)]
  rs, fs = zip(*rf)

  m = stick_lengths[0]
  zipfs = [(r, m/pow(alpha+r, B)) for r in range(1, len(stick_lengths)+1)]
  rz, rzf = zip(*zipfs)

  plt.clf()
  plt.xscale('log')
  plt.yscale('log')
  plt.title('Zipf plot')
  plt.xlabel('rank')
  plt.ylabel('frequency')
  plt.plot(rs, fs, 'ro')
  plt.plot(rz, rzf, 'bo')
  plt.show()
