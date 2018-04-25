#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter

NLP_DATA = "../corpus/cleaned.txt"

def report(file):
    classifier_of_noun = {}
    with open(file, mode='r', encoding="utf-8") as f:
        for line in f:
            tokens = line.split("\t")
            assert 2 == len(tokens)
            words = tokens[0].split()
            noun = words[-1]
            classifier = tokens[1]
            if noun not in classifier_of_noun:
                classifier_of_noun[noun] = Counter()
            classifier_of_noun[noun][classifier] += 1
    for noun in classifier_of_noun:
        print("{}:{}".format(noun, classifier_of_noun[noun]))

if __name__ == '__main__':
    report(NLP_DATA)