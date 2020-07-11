#!/bin/python
import numpy
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
import sys
import time

# Generate k-means features for videos; each video is represented by a single vector

if __name__ == '__main__':

    kmeans_model = "kmeans.50.model"
    file_list = "list/all.video"
    cluster_num = 50

    # load the kmeans model
    t1 = time.time()
    kmeans = cPickle.load(open(kmeans_model, "rb"))

    for j in range(6):
        mfcc_path = "mfcc/HVC%d.csv" % (j+1)
        kmeans_path = "kmeans/HVC%d.kmeans.txt" % (j+1)

        raw_feat = numpy.genfromtxt(mfcc_path, delimiter=";")

        pred_centers = kmeans.predict(raw_feat)
        histogram = numpy.zeros(cluster_num)
        for i in range(cluster_num):
            histogram[i] = numpy.count_nonzero(pred_centers == i)
        histogram /= numpy.sum(histogram)
        numpy.savetxt(kmeans_path, histogram)
        print "\r%s" % kmeans_path,

    print " "
    t2 = time.time() - t1
    print "Time taken for creating kmenas : %f seconds" % t2

    print "K-means features generated successfully!"
