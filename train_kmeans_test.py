#!/bin/python

import numpy
import os
from sklearn.cluster.k_means_ import KMeans
import cPickle
import sys

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    mfcc_csv_file = "mfcc/HVC961.mfcc.csv"
    output_file = "kmeans.50.model"
    cluster_num = 50

    raw_feat_mat = numpy.loadtxt(mfcc_csv_file, delimiter=";", dtype="float")

    kmeans = KMeans(n_clusters=cluster_num, random_state=0, n_init=5, n_jobs=-1).fit(raw_feat_mat)

    cPickle.dump(kmeans, open(output_file, "wb"))

    print "K-means trained successfully!"
