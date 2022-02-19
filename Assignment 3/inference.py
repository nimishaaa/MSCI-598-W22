# -*- coding: utf-8 -*-
"""
Nimisha Saxena - 20716444
Assignment 3
MSCI 598
"""

import sys
from gensim.models import Word2Vec
import re

path = sys.argv[1]
corpus = []
with open(path, "r") as f:
  for line in f:
    corpus.append(line)

model = Word2Vec.load('data/w2v.model')

def most_similar_func (word):
  return model.wv.most_similar(word,topn=20)
