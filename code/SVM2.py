#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors
import gensim
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics


W2V_FILE = "../wiki.zh.vec"

def to_vector(row):
    words = row['features'].split()
    vec = np.zeros(300)
    for word in words:
        if words != '<CL>':
            if word in model.wv:
                vec += model.wv[word]
    return vec


if __name__ == '__main__':
    df = pd.read_csv('../corpus/cleaned.txt',encoding='utf-8',sep='\t',names=['features','label'])
    model = KeyedVectors.load_word2vec_format(W2V_FILE, binary=False)
    df['vector'] = df.apply(to_vector, axis=1)
    features = df['vector']
    X_train, X_test, Y_train, Y_test = train_test_split(features, df['label'], test_size=.2, random_state=1)
    svc_model = LinearSVC()
    svc_model.fit(X_train, Y_train)
    predicted = svc_model.predict(X_test)

    print(metrics.classification_report(Y_test, predicted))

