#!/bin/python
import numpy
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
import sys
import time

if __name__ == '__main__':

    t1 = time.time()
    file_list = "list/all.video"
    cluster_num = 100

    # load the kmeans model
    fread = open(file_list, "r")

    for line in fread.readlines():
        kmeans_path = "kmeans/kmeans/" + line.replace('\n', '') + ".kmeans_%d.txt" % cluster_num
        if not os.path.exists(kmeans_path):
            continue
        asr_feat_path = "asrfeat2/" + line.replace('\n', '') + ".asrfeat2.txt"
        if not os.path.exists(asr_feat_path):
            continue
        best_path = "best/" + line.replace('\n', '') + ".best.txt"
        kmeans_feat = numpy.genfromtxt(kmeans_path)
        asr_feat = numpy.genfromtxt(asr_feat_path)
        best_feat = numpy.append(kmeans_feat, asr_feat)
        numpy.savetxt(best_path, best_feat)
        print "\r%s" % kmeans_path,
    print " "
    t2 = time.time() - t1
    print "Time taken for creating kmeans : %f seconds" % t2

    print "best features generated successfully!"
