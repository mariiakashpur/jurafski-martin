from __future__ import division
from collections import Counter, defaultdict
import math

class CustomLanguageModel: # Good-Turing smoothing

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    # TODO your code here
    self.words = set([]) # corpus types
    self.tokens = [] # corpus tokens
    self.freqs = defaultdict(int)
    self.gt_freqs = {}
    self.unifreqs = {}
    self.probs = {} 
    self.N1 = 0
    self.N2 = 0
    self.N3 = 0
    self.N4 = 0
    self.N5 = 0
    self.N6 = 0
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

    for i in range(len(self.tokens) - 1):
        self.freqs[(self.tokens[i], self.tokens[i+1])] += 1

    self.unifreqs = Counter(self.tokens)


    # count how many bigrams occuring once,twice 
   
    for key in self.freqs: 
        if self.freqs[key] == 1:
            self.N1 += 1
        elif self.freqs[key] == 2:
            self.N2 += 1
        elif self.freqs[key] == 3:
            self.N3 += 1
        elif self.freqs[key] == 4:
            self.N4 += 1
        elif self.freqs[key] == 5:
            self.N5 += 1
        elif self.freqs[key] == 6:
            self.N6 += 1

    # k = 5, c* = c for c > k
    # recounting frequencies with Good-Turing algorithm
    for key in self.freqs:
        if self.freqs[key] == 1:
            self.gt_freqs[key] = ((2 * self.N2 / self.N1) - (1 * 6 * self.N6 / self.N1)) / (1 - (6 * self.N6 / self.N1))
        elif self.freqs[key] == 2:
            self.gt_freqs[key] = ((3 * self.N3 / self.N2) - (2 * 6 * self.N6 / self.N1)) / (1 - (6 * self.N6 / self.N1))
        elif self.freqs[key] == 3:
            self.gt_freqs[key] = ((4 * self.N4 / self.N3) - (3 * 6 * self.N6 / self.N1)) / (1 - (6 * self.N6 / self.N1))
        elif self.freqs[key] == 4:
            self.gt_freqs[key] = ((5 * self.N5 / self.N4) - (4 * 6 * self.N6 / self.N1)) / (1 - (6 * self.N6 / self.N1))
        elif self.freqs[key] == 5:
            self.gt_freqs[key] = ((6 * self.N6 / self.N5) - (5 * 6 * self.N6 / self.N1)) / (1 - (6 * self.N6 / self.N1))
        else:
            self.gt_freqs[key] = self.freqs[key] 

    for key in self.gt_freqs:
        self.probs[key] = self.gt_freqs[key] / self.unifreqs[key[0]]

    pass

    
  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    # TODO your code here
    score = 0.0
    zero_count = 0
    for i in range(len(sentence) - 1):
        if (sentence[i], sentence[i+1]) not in self.probs:
            zero_count += 1
    for i in range(len(sentence) - 1):
        if (sentence[i], sentence[i+1]) in self.probs:
            score += math.log(self.probs[(sentence[i], sentence[i+1])]) 
        else:
            score += math.log(self.N1 / len(self.probs) / zero_count) # words unseen in train data
    return score






    
