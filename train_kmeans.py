#!/bin/python 

import numpy
import os
from sklearn.cluster.k_means_ import KMeans
import cPickle
import sys
import time

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} mfcc_csv_file cluster_num output_file".format(sys.argv[0])
        print "mfcc_csv_file -- path to the mfcc csv file"
        print "cluster_num -- number of cluster"
        print "output_file -- path to save the k-means model"
        exit(1)

    t1 = time.time()
    mfcc_csv_file = sys.argv[1]
    output_file = sys.argv[3]
    cluster_num = int(sys.argv[2])

    raw_feat_mat = numpy.loadtxt(mfcc_csv_file, delimiter=",", dtype="float")

    kmeans = KMeans(n_clusters=cluster_num, random_state=0, n_init=5, n_jobs=-1).fit(raw_feat_mat)

    cPickle.dump(kmeans, open(output_file, "wb"))

    print "K-means trained successfully!"
