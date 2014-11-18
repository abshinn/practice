#!/usr/bin/env python2.7 -B -tt
"""
Simple Implementation of Term Frequency - Inverse Document Frequency
"""

import os
from collections import Counter

import numpy as np
np.set_printoptions(linewidth=100)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer

rg = RegexpTokenizer(r"(?u)\b\w\w+\b")
stopset = stopwords.words("english")


def download_mininewsgroups():
    """ Download data with wget and extract with tar. """

    baseurl = "https://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/"
    os.system("mkdir -p DATA/twenty_newsgroups/")
    os.system("wget {} -P DATA/twenty_newsgroups/".format(baseurl + "mini_newsgroups.tar.gz"))
    os.system("tar xvf DATA/twenty_newsgroups/mini_newsgroups.tar.gz")
    os.system("mv mini_newsgroups/ DATA/twenty_newsgroups/")


def create_dictionary(bigram=False):
    """ Compile dictionary for the mini newsgroups data set. """

    baseurl = "DATA/twenty_newsgroups/mini_newsgroups/"

    if os.path.isfile(baseurl + "talk.religion.misc/84567") == False:
        download_mininewsgroups()

    newsgroups = os.listdir(baseurl)
    word_corpus = {}

    for newsgroup in newsgroups:
        dictionary = set()

        threads = os.listdir("{}{}".format(baseurl, newsgroup))
        all_words = []

        for thread in threads:
            with open("{}{}/{}".format(baseurl, newsgroup, thread), "rb") as t:
                words = [word for word in rg.tokenize(t.read().lower()) if word not in stopset]
              
            if bigram:

                bigrams = []
                for i in xrange(len(words)-1):
                    bigrams.append( " ".join(words[i:i+2]) )

                words.extend(bigrams)
           
            all_words.extend(words)

            dictionary = dictionary.union( set(words) )

        word_corpus[newsgroup] = all_words

    dictionary = np.array(list(dictionary))

    print "dictionary length: {}".format(len(dictionary))

    return word_corpus, dictionary


class TFIDF(object):
    """ TFIDF Vectorizer. """

    def __init__(self, dictionary):
        self.dictionary = dictionary

    def index_dictionary(self, dictionary):
        """ Turn word-dictionary from a set to a numpy array. """
        return np.array(list(dictionary))

    def vectorize(self, corpus):
        """ Turn a corpus of documents into a numpy 2D array: rows and columns correspond to documents and words, respectively. """

        term_count = np.zeros((len(corpus), self.dictionary.shape[0]), dtype=np.float32)

        print "size: {}".format(term_count.shape)

        documents = sorted(corpus.keys())
        for document_index, document in enumerate(documents):
            document_count = Counter(corpus[document])

            for word, count in document_count.items():
                term_count[document_index, np.where(self.dictionary == word)[0]] = count

        tf = term_count / term_count.max(axis=0) 

        N = len(corpus)
        n = (term_count > 0).sum(axis=0).astype(np.float32)

        idf = np.log( N / n )

        assert idf.dtype == np.float32
        assert idf.shape == (self.dictionary.shape[0],)

        tfidf = tf * idf

        return tfidf


def cluster_newsgroups():
    """ Cluster newsgroup categories. """
    from kmeans import KMeans
    from similarity import simMatrix

    corpus, dictionary = create_dictionary(bigram=True)
    tfidf = TFIDF(dictionary)
    newsgroups = tfidf.vectorize(corpus)

    categories = sorted(corpus.keys())

    N = 6
    print "\n{}-Most Common Words".format(N)
    for index, category in enumerate(categories):
        nlargest = np.argpartition(newsgroups[index,:], -N)[-N:]
        nlargest = nlargest[np.argsort(newsgroups[index,nlargest])][::-1]
        print "{:>24} {}".format(category, dictionary[nlargest])
    print

    K = 3
    km = KMeans(n_clusters=K)
    km.fit(newsgroups)

    labels = km.labels_

    print "\nKMeans Label Assignment, K = {}".format(K)
    for category, label, in zip(categories, labels):
        print int(label), category

    simMatrix(newsgroups).plot().show()

if __name__ == "__main__":
    cluster_newsgroups()

