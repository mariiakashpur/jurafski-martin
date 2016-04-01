from __future__ import division
from collections import defaultdict, Counter
import math


class StupidBackoffLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    # TODO your code here
    self.words = set([])
    self.tokens = []
    self.unifreqs = {}
    self.bifreqs = defaultdict(int)
    self.probs = {}
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

    self.unifreqs = Counter(self.tokens)

    for i in range(len(self.tokens) - 1):
        self.bifreqs[(self.tokens[i], self.tokens[i+1])] += 1

    
    for key in self.bifreqs:
        self.probs[key] = self.bifreqs[key] / self.unifreqs[key[0]] 
    pass

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    # TODO your code here
    score = 0.0
    for i in range(len(sentence) - 1):
        if (sentence[i], sentence[i+1]) in self.probs:
            score += math.log(self.probs[(sentence[i], sentence[i+1])]) 
        elif sentence[i+1] in self.unifreqs:
            score += math.log(0.4 * (self.unifreqs[sentence[i+1]] + 1) / (len(self.tokens) + len(self.words)))
        else:
            score += math.log(0.4 * 1 / (len(self.tokens) + len(self.words))) 
    return score









