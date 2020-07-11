#!/bin/python
import numpy
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
import sys
from nltk import word_tokenize
from gensim.models import Word2Vec

if __name__ == '__main__':
    file_list = "list/all.video"
    vocab = "asrfeat/vocab"
    feat_dim = 100

    fread = open(file_list, "r")
    vocabulary = []
    for line in fread.readlines():
        asr_path = "asrs/" + line.replace('\n', '') + ".txt"
        if not os.path.exists(asr_path):
            continue
        asr_file = open(asr_path, "rt")
        asr_script = asr_file.read()
        asr_file.close()
        tokens = word_tokenize(asr_script)
        words = [word for word in tokens if word.isalpha()]
        vocabulary.append(words)
        print "\r%s" % asr_path,
    fread.close()

    model = Word2Vec(vocabulary, size=feat_dim, window=5, min_count=1, workers=4)
    cPickle.dump(model, open(vocab, "wb"))
    fread = open(file_list, "r")
    for line in fread.readlines():
        asr_path = "asrs/" + line.replace('\n', '') + ".txt"
        asr_feat_path = "asrfeat/" + line.replace('\n', '') + ".asrfeat.txt"
        if not os.path.exists(asr_path):
            continue
        asr_file = open(asr_path, "rt")
        asr_script = asr_file.read()
        asr_file.close()
        tokens = word_tokenize(asr_script)
        words = [word for word in tokens if word.isalpha()]
        if len(words) == 0:
            asr_vector = numpy.zeros(feat_dim)
        else:
            asr_vector = numpy.sum(model[words], axis=0) / len(words)
        numpy.savetxt(asr_feat_path, asr_vector)
        print asr_feat_path

    print("ASR features generated successfully!")
