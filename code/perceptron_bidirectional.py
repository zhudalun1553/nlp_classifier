#!/usr/bin/env python3
"""
ANLP A4: Perceptron

Usage: python perceptron.py NITERATIONS

(Adapted from Alan Ritter)
"""
import sys, os, glob

from collections import Counter
import numpy as np
from nltk.util import ngrams

from sklearn.model_selection import train_test_split


from evaluation import Eval

BIAS_FEATURE = "^bias"
NLP_DATA = "./corpus/bidirectional.txt"

def extract_feats(sentence):
    """
    Extract input features (percepts) for a given document.
    Each percept is a pairing of a name and a boolean, integer, or float value.
    A document's percepts are the same regardless of the label considered.
    """
    feats = {}
    ff = Counter()
    words = sentence.split()
    for word in words:
        if word == "<CL>":
            continue
        ff[word] = 1
    bigrams = ngrams(words, 2)
    for gram in bigrams:
        ff[gram] = 1
    noun = words[-1]
    ff["<noun>" + noun + "</noun>"] = 1
    # add bias feature
    ff[BIAS_FEATURE] = 1
    feats["noun"] = noun
    feats["features"] = ff
    return feats

def load_featurized_docs(X):
    featdocs = []
    for sentence in X:
        featdocs.append(extract_feats(sentence))
    return featdocs

def load_file(file):
    X = []
    y = []
    classifier_set = set()
    with open(file, mode="r", encoding="utf-8") as f:
        for line in f:
            tokens = line.split("\t")
            assert 2 == len(tokens)
            X.append(tokens[0])

            #baseline
            # y.append(tokens[1])
            # classifier_set.add(tokens[1])

            #bidirectional
            cls = tokens[1].split()
            y.append(cls)
            classifier_set.add(cls[0])


    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2)
    return X_train, y_train, X_val, y_val, X_test, y_test, list(classifier_set)

class Perceptron:
    def __init__(self, train_docs, train_labels, classes, MAX_ITERATIONS=100, dev_docs=None, dev_labels=None, classifier_per_noun=None):
        self.CLASSES = classes
        self.MAX_ITERATIONS = MAX_ITERATIONS
        self.dev_docs = dev_docs
        self.dev_labels = dev_labels
        self.weights = {l: Counter() for l in self.CLASSES}
        self.classifier_per_noun = classifier_per_noun
        self.learn(train_docs, train_labels)


    def copy_weights(self):
        """
        Returns a copy of self.weights.
        """
        return {l: Counter(c) for l,c in self.weights.items()}

    def learn(self, train_docs, train_labels):
        """
        Train on the provided data with the perceptron algorithm.
        Up to self.MAX_ITERATIONS of learning.
        At the end of training, self.weights should contain the final model
        parameters.
        """
        for i in range(self.MAX_ITERATIONS):
            updates = 0
            train_zip = list(zip(train_docs, train_labels))
            for doc, labels in train_zip:

                #bidirectional
                label = labels[0]

                pred = self.predict(doc)
                if pred != label:
                    updates += 1
                    self.weights[label] += doc["features"]
                    self.weights[pred].subtract(doc["features"])
            trainAcc = self.test_eval(train_docs, train_labels)
            devAcc = self.test_eval(self.dev_docs, self.dev_labels)
            params = np.sum([len(self.weights[label]) for label in self.CLASSES])
            print("iteration: {} updates={}, trainAcc={}, devAcc={}, params={}"\
                  .format(i, updates, trainAcc, devAcc, params),\
                  file=sys.stderr)
            if updates == 0:
                break



    def score(self, doc, label):
        """
        Returns the current model's score of labeling the given document
        with the given label.
        """
        weigth = self.weights[label]
        score = 0
        ff = doc["features"]
        for word in ff:
            score += ff[word] * weigth[word]
        return score

    def predict(self, doc):
        """
        Return the highest-scoring label for the document under the current model.
        """
        noun = doc["noun"]
        if noun in self.classifier_per_noun:
            classes = self.classifier_per_noun[noun]
            scores = [self.score(doc, label) for label in classes]
            return classes[np.argmax(scores)]
        else:
            scores = [self.score(doc, label) for label in self.CLASSES]
            return self.CLASSES[np.argmax(scores)]

    def test_eval(self, test_docs, test_labels):
        pred_labels = [self.predict(d) for d in test_docs]
        ev = Eval(test_labels, pred_labels)
        return ev.accuracy()

    def report(self):
        print("Highest weighted features")
        for c in self.CLASSES:
            print("{}:{}\n".format(c, self.weights[c].most_common(10)))
        print("Lowest weighted features")
        for c in self.CLASSES:
            print("{}:{}\n".format(c, list(reversed(self.weights[c].most_common()[-10:]))))
        print("Bias")
        for c in self.CLASSES:
            print("{}:{}".format(c, self.weights[c][BIAS_FEATURE]))

def get_classifier_of_noun(X_train, y_train):
    classifier_of_noun = {}
    for sentence, classifiers in zip(X_train, y_train):
        words = sentence.split()
        noun = words[-1]
        if noun not in classifier_of_noun:
            classifier_of_noun[noun] = set()
        classifier_of_noun[noun].add(classifiers[0])
    ret = {}
    for noun in classifier_of_noun:
        ret[noun] = list(classifier_of_noun[noun])
    return ret

if __name__ == "__main__":
    args = sys.argv[1:]
    niters = int(args[0])

    X_train, y_train, X_val, y_val, X_test, y_test, c_classes = load_file(NLP_DATA)
    X_train_docs = load_featurized_docs(X_train)
    print(len(X_train_docs), 'training docs with',
        sum(len(d["features"]) for d in X_train_docs)/len(X_train_docs), 'percepts on avg', file=sys.stderr)

    X_val_docs = load_featurized_docs(X_val)
    print(len(X_val_docs), 'dev docs with',
        sum(len(d["features"]) for d in X_val_docs)/len(X_val_docs), 'percepts on avg', file=sys.stderr)

    X_test_docs = load_featurized_docs(X_test)
    print(len(X_test_docs), 'test docs with',
        sum(len(d["features"]) for d in X_test_docs)/len(X_test_docs), 'percepts on avg', file=sys.stderr)

    ptron = Perceptron(X_train_docs, y_train, c_classes, MAX_ITERATIONS=niters, dev_docs=X_val_docs, dev_labels=y_val, classifier_per_noun=get_classifier_of_noun(X_train, y_train))
    acc = ptron.test_eval(X_test_docs, y_test)
    print(acc, file=sys.stderr)
    ptron.report()

