#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
from collections import Counter

NLP_DATA = "../corpus/cleaned.txt"

def baseline(file):
    X = []
    y = []
    y_counter = Counter()
    with open(file, mode="r", encoding="utf-8") as f:
        for line in f:
            tokens = line.split("\t")
            assert 2 == len(tokens)
            X.append(tokens[0])
            y.append(tokens[1])
            y_counter[tokens[1]] += 1

    print(len(y))
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2)
    c_counter = Counter()
    for classifier in y_train:
        c_counter[classifier] += 1
    ge = c_counter.most_common(1)[0]
    print("the most common classifier is {}".format(ge[0]))
    print("baseline accuracy on training dataset is {}.".format(ge[1] / len(X_train)))

    count_val = 0
    for classifier in y_val:
        if classifier == ge[0]:
            count_val += 1
    print("baseline accuracy on validate dataset is {}.".format(count_val / len(y_val)))

    count_test = 0
    for classifier in y_test:
        if classifier == ge[0]:
            count_test += 1
    print("baseline accuracy on validate dataset is {}.".format(count_test / len(y_test)))

if __name__ == '__main__':
    baseline(NLP_DATA)