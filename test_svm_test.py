#!/bin/python

import numpy
import os
from sklearn.svm.classes import SVC
import cPickle
import sys
import time

# Apply the SVM model to the testing videos; Output the score for each video

if __name__ == '__main__':
    t1 = time.time()
    event_name = "P001"
    model_file = "mfcc_pred/svm.%s.model" % event_name
    feat_dir = "kmeans/"
    feat_dim = 50
    output_file = "mfcc_pred/%s_mfcc.lst" % event_name
    event_name = model_file.split(".")[1]

    fread = open("list/test.video", "r")
    clf = cPickle.load(open(model_file, "rb"))
    X = []
    for line in fread.readlines():
        kmeans_path = "kmeans/" + line.replace('\n', '') + ".kmeans.txt"
        if os.path.exists(kmeans_path):
            kmeans_feat = numpy.genfromtxt(kmeans_path, delimiter=";")
        else:
            kmeans_feat = numpy.zeros(feat_dim)
        X.append(kmeans_feat)
    X = numpy.array(X)
    predict = clf.predict_proba(X)
    numpy.savetxt(output_file, predict[:, 1])
    t2 = time.time() - t1
    print "Time taken for testing %s SVM : %f seconds" % (event_name, t2)

    print 'SVM tested successfully for event %s!' % event_name



