#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
RAW_FILE = "../corpus/rawdata_CL.txt"

def extract_data(file):
    np_list = []
    classifier_list = []
    with open(file, mode='r') as f:
        for line in f:
            tokens = line.split("\t")
            classifier = tokens[4]
            np = tokens[1]
            np = np.replace(classifier, "<CL>", 1)
            words = np.split()
            words[-1] = "<h>{}</h>".format(words[-1])
            np = " ".join(words)
            np_list.append(np)
            classifier_list.append(classifier)
    data = {'tagged':np_list, 'classifier':classifier_list}
    df = pd.DataFrame(data=data)
    df.to_csv("../corpus/cleaneddata_CL.csv")


if __name__ == '__main__':
    extract_data(RAW_FILE)