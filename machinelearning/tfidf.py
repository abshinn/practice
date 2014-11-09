#!/usr/bin/env python2.7 -B -tt
"""
Simple Implementation of Term Frequency - Inverse Document Frequency
"""

import os

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer

# rg = RegexpTokenizer(r"\w+")
rg = RegexpTokenizer(r"[a-zA-Z]+")
stopset = stopwords.words("english")


def download_mininewsgroups():
    """ Download data with wget and extract with tar. """

    baseurl = "https://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/"
    os.system("mkdir -p DATA/twenty_newsgroups/")
    os.system("wget {} -P DATA/twenty_newsgroups/".format(baseurl + "mini_newsgroups.tar.gz"))
    os.system("tar xvf DATA/twenty_newsgroups/mini_newsgroups.tar.gz")
    os.system("mv mini_newsgroups/ DATA/twenty_newsgroups/")


def create_dictionary():
    """ Compile dictionary for the mini newsgroups data set. """

    baseurl = "DATA/twenty_newsgroups/mini_newsgroups/"

    if os.path.isfile(baseurl + "talk.religion.misc/84567") == False:
        download_mininewsgroups()

    newsgroups = os.listdir(baseurl)
    word_corpus = {}

    for newsgroup in newsgroups:
        dictionary = set()

        threads = os.listdir("{}{}".format(baseurl, newsgroup))

        for thread in threads:
            with open("{}{}/{}".format(baseurl, newsgroup, thread), "rb") as t:
                words = [word for word in rg.tokenize(t.read()) if word not in stopset]

            dictionary = dictionary.union( set(words) )

    print "dictionary length: {}".format(len(dictionary))


class TFIDF(object):
    """ TFIDF Vectorizer. """

    def __init__(self, corpus, dictionary):
        self.corpus = corpus
        self.dictionary = dictionary


if __name__ == "__main__":
    create_dictionary()

