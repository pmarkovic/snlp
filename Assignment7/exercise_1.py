
from collections import defaultdict
import math


class TreeNode:
  def __init__(self, root):
    self.children = dict()
    self.word = root
    self.count = 0
    self.pruned = False
    self.pruned_count = 0

class CountTree():
  def __init__(self, n=4):
    self.n = n
    self.roots = dict()
  
  def add(self, ngram):
    if ngram[-1] not in self.roots:
      self.roots[ngram[-1]] = TreeNode(ngram[-1])
    self.roots[ngram[-1]].count += 1
    root = self.roots[ngram[-1]]
    hist = ngram[:-1]
    while hist:
      if hist[-1] not in root.children:
        root.children[hist[-1]] = TreeNode(hist[-1])
      child = root.children[hist[-1]]
      child.count += 1
      root = child
      hist = hist[:-1]

  def get(self, ngram):
    ngram = ngram[::-1] # ABCE -> ECBA
    if len(ngram) == 0: return 0
    if ngram[0] not in self.roots:
      return 0
    root = self.roots[ngram[0]]
    for x in ngram[1:]:
      if x in root.children:
        root = root.children[x]
      elif root.pruned: return root.pruned_count
      else: return 0
    return root.count

  def perplexity(self, ngrams, vocab):
    pass

  def prune_node(self, key, parent_dict, node, k):
    if node.count < k:
      if key in parent_dict:
        del_node = parent_dict.pop(key)
        return del_node.count
    else:
      for child in list(node.children):
        node.pruned_count += self.prune_node(child, node.children, node.children[child], k)
        node.pruned = True
    return 0

  def prune(self, k):
    for key in list(self.roots):
      self.prune_node(key, self.roots, self.roots[key], k)


