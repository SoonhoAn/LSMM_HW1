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
    if len(sys.argv) != 5:
        print "Usage: {0} event_name feat_dir feat_dim output_file".format(sys.argv[0])
        print "event_name -- name of the event (P001, P002 or P003 in Homework 1)"
        print "feat_dir -- dir of feature files"
        print "feat_dim -- dim of features"
        print "output_file -- path to save the svm model"
        exit(1)

    event_name = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]

    fread = open("list/train", "r")
    clf = SVC(probability=True)
    X, Y = [], []
    for i in fread.readlines():
        i = i.split(" ")
        line = i[0]
        label = i[1].replace('\n', '')
        feature_path = feat_dir + line + ".%s.txt" % feat_dir[:-1]
        if os.path.exists(feature_path):
            feature = numpy.genfromtxt(feature_path, delimiter=";")
        else:
            feature = numpy.zeros(feat_dim)
            label = "NULL"
        if label != event_name:
            label = "NULL"
        X.append(feature)
        Y.append(label)
    X = numpy.array(X)
    Y = numpy.array(Y)
    clf.fit(X, Y)
    cPickle.dump(clf, open(output_file, "wb"))
    t2 = time.time() - t1
    print "Time taken for training %s SVM : %f seconds" % (event_name, t2)

    print 'SVM trained successfully for event %s!' % event_name
