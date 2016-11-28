import json, sys
import base64
from Datum import Datum
import re

class FeatureFactory:
    """
    Add any necessary initialization steps for your features here
    Using this constructor is optional. Depending on your
    features, you may not need to intialize anything.
    """
    def __init__(self):
        pass


    """
    Words is a list of the words in the entire corpus, previousLabel is the label
    for position-1 (or O if it's the start of a new sentence), and position
    is the word you are adding features for. PreviousLabel must be the
    only label that is visible to this method. 
    """

    def computeFeatures(self, words, previousLabel, position):
        features = []
        currentWord = words[position]
        if position - 1 >= 0:
            prevWord = words[position-1]
        else:
            prevWord = "NOTEKZIST"


        """ Baseline Features """
        features.append("word=" + currentWord)
        features.append("prevLabel=" + previousLabel)
        features.append("word=" + currentWord + ", prevLabel=" + previousLabel)


	"""
        Warning: If you encounter "line search failure" error when
        running the program, considering putting the baseline features
	back. It occurs when the features are too sparse. Once you have
        added enough features, take out the features that you don't need. 
	"""


	""" TODO: Add your features here """
        if currentWord[0].isupper():
            features.append("capitalized=YES")
        else:
            features.append("capitalized=NO")
        if currentWord.isupper():
            features.append("allcap=YES")
        else:
            features.append("allcap=NO")
        if currentWord.islower():
            features.append("lower=YES")
        else:
            features.append("lower=NO")

        if re.search(r"-", currentWord):
            features.append("hyphen=YES")
        else:
            features.append("hyphen=NO")
        if re.search(r"\d", currentWord):
            features.append("containsDigit=YES")
        else:
            features.append("containsDigit=NO")

        # word shapes (see Jurafsky on NER)
        shape = re.sub(r"\d", "d", re.sub(r"[a-z]", "x", re.sub(r"[A-Z]", "X", currentWord)))
        if len(shape) > 4:
            beg = shape[:2]
            middle = ''.join(set(shape[2:-2]))
            end = shape[-2:]
            shape = beg+middle+end
        features.append(shape)

        # features.append("SUFF3=" + currentWord[-3:])  # feature suffix - last 3 chars
        # features.append("SUFF2=" + currentWord[-2:])  # feature suffix - last 2 chars
        # features.append("PREF3=" + currentWord[0:3])  # feature prefix - first 3 chars
        # features.append("PREF2=" + currentWord[0:2])  # feature prefix - first 2 chars
        # features.append("prevWord=" + prevWord)
        # if prevWord[0].isupper():
        #     features.append("prevCapitalized=YES")
        # else:
        #     features.append("prevCapitalized=NO")
        return features

    """ Do not modify this method """
    def readData(self, filename):
        data = [] 
        
        for line in open(filename, 'r'):
            line_split = line.split()
            # remove emtpy lines
            if len(line_split) < 2:
                continue
            word = line_split[0]
            label = line_split[1]

            datum = Datum(word, label)
            data.append(datum)

        return data

    """ Do not modify this method """
    def readTestData(self, ch_aux):
        data = [] 
        
        for line in ch_aux.splitlines():
            line_split = line.split()
            # remove emtpy lines
            if len(line_split) < 2:
                continue
            word = line_split[0]
            label = line_split[1]

            datum = Datum(word, label)
            data.append(datum)

        return data


    """ Do not modify this method """
    def setFeaturesTrain(self, data):
        newData = []
        words = []

        for datum in data:
            words.append(datum.word)

        ## This is so that the feature factory code doesn't
        ## accidentally use the true label info
        previousLabel = "O"
        for i in range(0, len(data)):
            datum = data[i]

            newDatum = Datum(datum.word, datum.label)
            newDatum.features = self.computeFeatures(words, previousLabel, i)
            newDatum.previousLabel = previousLabel
            newData.append(newDatum)

            previousLabel = datum.label

        return newData

    """
    Compute the features for all possible previous labels
    for Viterbi algorithm. Do not modify this method
    """
    def setFeaturesTest(self, data):
        newData = []
        words = []
        labels = []
        labelIndex = {}

        for datum in data:
            words.append(datum.word)
            if not labelIndex.has_key(datum.label):
                labelIndex[datum.label] = len(labels)
                labels.append(datum.label)
        
        ## This is so that the feature factory code doesn't
        ## accidentally use the true label info
        for i in range(0, len(data)):
            datum = data[i]

            if i == 0:
                previousLabel = "O"
                datum.features = self.computeFeatures(words, previousLabel, i)

                newDatum = Datum(datum.word, datum.label)
                newDatum.features = self.computeFeatures(words, previousLabel, i)
                newDatum.previousLabel = previousLabel
                newData.append(newDatum)
            else:
                for previousLabel in labels:
                    datum.features = self.computeFeatures(words, previousLabel, i)

                    newDatum = Datum(datum.word, datum.label)
                    newDatum.features = self.computeFeatures(words, previousLabel, i)
                    newDatum.previousLabel = previousLabel
                    newData.append(newDatum)

        return newData

    """
    write words, labels, and features into a json file
    Do not modify this method
    """
    def writeData(self, data, filename):
        outFile = open(filename + '.json', 'w')
        for i in range(0, len(data)):
            datum = data[i]
            jsonObj = {}
            jsonObj['_label'] = datum.label
            jsonObj['_word']= base64.b64encode(datum.word)
            jsonObj['_prevLabel'] = datum.previousLabel

            featureObj = {}
            features = datum.features
            for j in range(0, len(features)):
                feature = features[j]
                featureObj['_'+feature] = feature
            jsonObj['_features'] = featureObj
            
            outFile.write(json.dumps(jsonObj) + '\n')
            
        outFile.close()

