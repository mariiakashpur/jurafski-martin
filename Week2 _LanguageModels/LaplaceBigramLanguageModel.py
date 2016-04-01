from __future__ import division
from collections import defaultdict, Counter
import math

class LaplaceBigramLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    # TODO your code here
    self.words = set([])
    self.tokens = []
    self.freqs = defaultdict(int)
    self.unifreqs = {}
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

    for i in range(len(self.tokens) - 1):
        self.freqs[(self.tokens[i], self.tokens[i+1])] += 1

    self.unifreqs = Counter(self.tokens)
    
    for key in self.freqs:
        self.probs[key] = (self.freqs[key] + 1) / (self.unifreqs[key[0]] + len(self.words))

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
        else:
            score += math.log(1 / (self.unifreqs[sentence[0]] + len(self.words))) # words unseen in train data
    return score









