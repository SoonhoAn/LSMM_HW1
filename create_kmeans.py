#!/bin/python
import numpy
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
import sys
import time

# Generate k-means features for videos; each video is represented by a single vector

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} kmeans_model, cluster_num, file_list".format(sys.argv[0])
        print "kmeans_model -- path to the kmeans model"
        print "cluster_num -- number of cluster"
        print "file_list -- the list of videos"
        exit(1)

    t1 = time.time()
    kmeans_model = sys.argv[1]
    file_list = sys.argv[3]
    cluster_num = int(sys.argv[2])

    # load the kmeans model
    kmeans = cPickle.load(open(kmeans_model, "rb"))
    fread = open(file_list, "r")

    for line in fread.readlines():
        mfcc_path = "mfcc/" + line.replace('\n', '') + ".mfcc.csv"
        kmeans_path = "kmeans_%d/" % cluster_num + line.replace('\n', '') + ".kmeans.txt"
        if not os.path.exists(mfcc_path):
            continue
        mfcc_feat = numpy.genfromtxt(mfcc_path, delimiter=";")
        feat_dim = mfcc_feat.shape[1]
        pred_centers = kmeans.predict(mfcc_feat)
        histogram = numpy.zeros(cluster_num)
        for i in range(cluster_num):
            histogram[i] = numpy.count_nonzero(pred_centers == i)
        histogram /= numpy.sum(histogram)
        numpy.savetxt(kmeans_path, histogram)
        print "\r%s" % kmeans_path,
    print " "
    t2 = time.time() - t1
    print "Time taken for creating kmeans : %f seconds" % t2

    print "K-means features generated successfully!"
