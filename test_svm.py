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
    if len(sys.argv) != 6:
        print "Usage: {0} model_file feat_dir feat_dim output_file".format(sys.argv[0])
        print "model_file -- path of the trained svm file"
        print "feat_dir -- dir of feature files"
        print "feat_dim -- dim of features; provided just for debugging"
        print "output_file -- path to save the prediction score"
        exit(1)

    model_file = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]
    sset = sys.argv[5]
    event_name = model_file.split(".")[1]

    fread = open("list/%s.video" % sset, "r")
    clf = cPickle.load(open(model_file, "rb"))
    X = []
    for line in fread.readlines():
        feature_path = feat_dir + line.replace('\n', '') + ".%s.txt" % feat_dir[:-1]
        if os.path.exists(feature_path):
            feature = numpy.genfromtxt(feature_path, delimiter=";")
        else:
            feature = numpy.zeros(feat_dim)
        X.append(feature)
    X = numpy.array(X)
    predict = clf.predict_proba(X)
    numpy.savetxt(output_file, predict[:, 1])
    t2 = time.time() - t1
    print "Time taken for testing %s SVM : %f seconds" % (event_name, t2)

    print 'SVM tested successfully for event %s!' % event_name


