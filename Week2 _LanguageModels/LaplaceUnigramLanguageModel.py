from __future__ import division
from collections import Counter
import math

class LaplaceUnigramLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    # TODO your code here
    self.words = set([]) # corpus types
    self.tokens = [] # corpus tokens
    self.freqs = {} # dict where keys are types, values their frequencies
    self.probs = {} # dict where keys are types, values their probabilities with add-one smoothing
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    # TODO your code here
    for sentence in corpus.corpus: # iterate over sentences in the corpus
        for datum in sentence.data: # iterate over datums in the sentence
            word = datum.word # get the word
            self.tokens.append(word)
            self.words.add(word)

    self.freqs = Counter(self.tokens) # create dict with help of

    for key in self.freqs:
        self.probs[key] = (self.freqs[key] + 1) / (len(self.tokens) + len(self.words))

    pass

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    # TODO your code here
    score = 0.0
    for word in sentence:
        if word in self.probs:
            score += math.log(self.probs[word]) 
        else:
            score += math.log(1 / (len(self.tokens) + len(self.words))) # words unseen in train data
    return score





