import numpy as np
import pandas as pd
import sys

from sklearn.linear_model import SGDClassifier

sys.path.insert(0,'../..')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm
class my_model():
    def fit(self, X, y):
        # do not exceed 29 mins

        # preprocessing text data
        X['final']=X['description']+X['requirements']+X['title']
        X['final'] = X['final'].str.replace('[^a-zA-Z]', ' ')
        #vectorizing
        self.tfidf = TfidfVectorizer(stop_words='english', norm='l2', use_idf=False, smooth_idf=False)

        finalvec = self.tfidf.fit_transform(X['final'].astype('U'))

        params = {
            "loss": ["hinge", "log", "squared_hinge", "modified_huber"],
            "alpha": [0.0001, 0.001, 0.01, 0.1],
            "penalty": ["l2", "l1"],
        }

        model = SGDClassifier(max_iter=1000, class_weight="balanced")
        self.clf = GridSearchCV(model, param_grid=params)
        self.clf.fit(finalvec, y)

        return


    def predict(self, X):
        # remember to apply the same preprocessing in fit() on test data before making predictions
        X['final'] = X['description'] + X['requirements'] + X['title']
        X['final'] = X['final'].str.replace('[^a-zA-Z]', ' ')
        finalvec = self.tfidf.transform(X['final'].astype('U'))
        predictions = self.clf.predict(finalvec)
        return predictions