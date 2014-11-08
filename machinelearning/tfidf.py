#!/usr/bin/env python2.7 -B -tt
"""
Simple Implementation of Term Frequency - Inverse Document Frequency
"""


def download_mininewsgroups():
    """ Download data with wget and extract with tar. """

    baseurl = "https://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/"
    os.system("mkdir -p DATA/twenty_newsgroups/")
    os.system("wget {} -P DATA/twenty_newsgroups/".format(baseurl + "mini_newsgroups.tar.gz"))
    os.system("tar xvf DATA/twenty_newsgroups/mini_newsgroups.tar.gz")
    os.system("mv mini_newsgroups/ DATA/twenty_newsgroups/")

if __name__ == "__main__":
    download_mininewsgroups()

