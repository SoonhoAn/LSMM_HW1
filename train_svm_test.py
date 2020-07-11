#!/bin/python

import numpy
import os
from sklearn.svm.classes import SVC
import cPickle
import sys
import time

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    t1 = time.time()
    event_name = "P003"
    feat_dir = "kmeans/"
    feat_dim = 50
    output_file = "mfcc_pred/svm.%s.model" % event_name

    fread = open("list/train", "r")
    clf = SVC(probability=True)
    X, Y = [], []
    for i in fread.readlines():
        i = i.split(" ")
        line = i[0]
        label = i[1].replace('\n', '')
        kmeans_path = "kmeans/" + line + ".kmeans.txt"
        if os.path.exists(kmeans_path):
            kmeans_feat = numpy.genfromtxt(kmeans_path, delimiter=";")
        else:
            kmeans_feat = numpy.zeros(feat_dim)
            label = "NULL"
        if label != event_name:
            label = "NULL"
        X.append(kmeans_feat)
        Y.append(label)
    X = numpy.array(X)
    Y = numpy.array(Y)
    clf.fit(X, Y)
    cPickle.dump(clf, open(output_file, "wb"))
    print " "
    t2 = time.time() - t1
    print "Time taken for training %s SVM : %f seconds" % (event_name, t2)

    print 'SVM trained successfully for event %s!' % event_name
