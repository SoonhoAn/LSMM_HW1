#!/bin/python
# Randomly select

import numpy
import os
import sys

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} file_list select_ratio output_file".format(sys.argv[0])
        print "file_list -- the list of video names"
        print "select_ratio -- the ratio of frames to be randomly selected from NULL audio file"
        print "output_file -- path to save the selected frames (feature vectors)"
        exit(1)

    file_list = sys.argv[1]  # Use train instead of train.video
    output_file = sys.argv[3]
    ratio = float(sys.argv[2])

    fread = open(file_list,"r")
    fwrite = open(output_file,"w")

    fread = fread.read().split("\n")
    np_fread = []
    for line in fread:
        newline = line.split(" ")
        np_fread.append(newline)
    np_fread = numpy.array(np_fread)
    num_P001 = numpy.count_nonzero(np_fread == "P001")
    num_P002 = numpy.count_nonzero(np_fread == "P002")
    num_P003 = numpy.count_nonzero(np_fread == "P003")
    num_NULL = numpy.count_nonzero(np_fread == "NULL")
    arg_P001 = numpy.argwhere(np_fread[:, 1] == "P001")
    arg_P002 = numpy.argwhere(np_fread[:, 1] == "P002")
    arg_P003 = numpy.argwhere(np_fread[:, 1] == "P003")
    arg_NULL = numpy.argwhere(np_fread[:, 1] == "NULL")
    numpy.random.seed(18877)
    numpy.random.shuffle(arg_NULL)
    select_size = round((num_P001 + num_P002 + num_P003) * ratio)
    arg_NULL = arg_NULL[:select_size]

    numpy.random.seed(18877)
    np_fread = np_fread[:, 0]
    np_select_P001 = numpy.take(np_fread, arg_P001)
    np_select_P002 = numpy.take(np_fread, arg_P002)
    np_select_P003 = numpy.take(np_fread, arg_P003)
    np_select_NULL = numpy.take(np_fread, arg_NULL)
    np_select_file = numpy.append(np_select_P001, np_select_P002)
    np_select_temp = numpy.append(np_select_P003, np_select_NULL)
    np_select_file = numpy.append(np_select_file, np_select_temp)

    for i in range(np_select_file.shape[0]):
        mfcc_path = "mfcc/" + np_select_file[i] + ".mfcc.csv"
        if not os.path.exists(mfcc_path):
            continue
        array = numpy.genfromtxt(mfcc_path, delimiter=";")
        select_size = int(array.shape[0])
        feat_dim = array.shape[1]

        for n in xrange(select_size):
            line = str(array[n][0])
            for m in range(1, feat_dim):
                line += ';' + str(array[n][m])
            fwrite.write(line + '\n')
    fwrite.close()

