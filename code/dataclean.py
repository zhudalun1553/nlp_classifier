#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from collections import Counter
import re

TRAIN_DATA = "../corpus/data/train.csv"
DEV_DATA = "../corpus/data/dev.csv"
TEST_DATA = "../corpus/data/test.csv"

NEW_DATA = "../corpus/cleaneddata_CL.csv"

NOUN_RE = re.compile("""<h>([^<]+)<\/h>""")

def get_noun_classifier_count(csv_file):
    df = pd.read_csv(csv_file)
    dict = {}
    for index, row in df.iterrows():
        matched = NOUN_RE.findall(row['tagged'])
        if (len(matched) == 0):
            print(row['tagged'])
        else:
            noun = matched[0]
            if noun not in dict:
                dict[noun] = Counter()
            classifier = row['classifier']
            dict[noun][classifier] += 1
    return dict;

if __name__ == '__main__':
    dict = get_noun_classifier_count(NEW_DATA)
    multiple_count = 0
    single_count = 0
    with open("output.txt", mode='w') as f:
        for noun in dict:
            if len(dict[noun]) > 1:
                f.write("{}->{}\n".format(noun, dict[noun]))
                multiple_count += 1
            else:
                single_count += 1

        f.write("noun with multiple classifiers: {}\n".format(multiple_count))
        f.write("noun with single classifiers:{}\n".format(single_count))
