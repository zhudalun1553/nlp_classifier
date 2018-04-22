#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
from collections import Counter

NLP_DATA = "../test.txt"


def baseline(file):
    X = []
    y = []
    with open(file, mode="r", encoding="utf-8") as f:
        for line in f:
            tokens = line.split("\t")
            assert 2 == len(tokens)
            X.append(tokens[0])
            y.append(tokens[1])
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, stratify=y_train_val)
    c_Counter = Counter()
    for classifier in y_train:
        c_Counter[classifier] += 1
    ge = c_Counter.most_common(1)[0]
    print("baseline accurary is {}, the most common classifier is {}".format(ge[1] / len(X_train), ge[0]))

def calculate_baseline(X_train, )

if __name__ == '__main__':
    baseline(NLP_DATA)