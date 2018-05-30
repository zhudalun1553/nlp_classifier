#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
from collections import Counter

NLP_DATA = "./corpus/43000.txt"

def baseline(file):
    X = []
    y = []
    #y_counter = Counter()
    with open(file, mode="r", encoding="utf-8") as f:
        for line in f:
            tokens = line.split("\t")
            assert 2 == len(tokens)
            X.append(tokens[0])
            y.append(tokens[1])
            #y_counter[tokens[1]] += 1

    print(len(y))
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2)
    c_counter = Counter()
    classifier_of_noun = {}
    for sentence, classifier in zip(X_train, y_train):
        noun = get_noun(sentence)
        if noun not in classifier_of_noun:
            classifier_of_noun[noun] = Counter()
        c_counter[classifier] += 1
        classifier_of_noun[noun][classifier] += 1
    #print(classifier_of_noun)
    ge = c_counter.most_common(1)[0]
    print("the most common classifier is {}".format(ge[0]))
    count_train = 0
    for sentence, classifier in zip(X_train, y_train):
        noun = get_noun(sentence)
        pred = classifier_of_noun[noun].most_common(1)[0][0]
        if classifier == pred:
            count_train += 1
    print("baseline accuracy of mccpn  on training dataset is {}.".format(count_train / len(y_train)))

    count_val = 0
    for sentence, classifier in zip(X_val, y_val):
        noun = get_noun(sentence)
        if noun in classifier_of_noun:
            pred = classifier_of_noun[noun].most_common(1)[0][0]
        else:
            pred = ge
        if classifier == pred:
            count_val += 1
    print("baseline accuracy on validate dataset is {}.".format(count_val / len(y_val)))

    count_test = 0
    for sentence, classifier in zip(X_test, y_test):
        noun = get_noun(sentence)
        if noun in classifier_of_noun:
            pred = classifier_of_noun[noun].most_common(1)[0][0]
        else:
            pred = ge
        if classifier == pred:
            count_test += 1
    print("baseline accuracy on test dataset is {}.".format(count_test / len(y_test)))

def get_noun(sentence):
    return sentence.split()[-1]

if __name__ == '__main__':
    baseline(NLP_DATA)