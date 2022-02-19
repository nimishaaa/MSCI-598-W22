# -*- coding: utf-8 -*-
"""
MSCI 598 - Assignment 3
February 18, 2022

Nimisha Saxena - 20716444

Write a python script using genism library to train a Word2Vec model on the Amazon corpus.
"""

!wget -q https://raw.githubusercontent.com/fuzhenxin/textstyletransferdata/master/sentiment/neg.txt
!wget -q https://raw.githubusercontent.com/fuzhenxin/textstyletransferdata/master/sentiment/pos.txt
!wget -q "https://gist.githubusercontent.com/sebleier/554280/raw/7e0e4a1ce04c2bb7bd41089c9821dbcf6d0c786c/NLTK's%2520list%2520of%2520english%2520stopwords"

import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec,KeyedVectors
from gensim.test.utils import datapath
import re
import unicodedata
from tqdm import tqdm
import gensim
import multiprocessing
import random

special_characters = "!\"#$%&()*+/:;<=>@[\]^`{|}~\t\n"
tokenized_sentences = []

def tokenize_func (filename,given_set):
  with open(filename) as f:
    for line in f:
      for char in special_characters:
        l = line.replace(char,' ')
      tokenized_sentences.append(l.split())
    return tokenized_sentences

neg_corpus = tokenize_func('neg.txt',tokenized_sentences)
pos_corpus = tokenize_func('pos.txt',tokenized_sentences)

stop_words = []
with open("NLTK's list of english stopwords") as f:
  for line in f:
    stop_words.append(line.split('\n')[0])

corpus = neg_corpus + pos_corpus

corpus_nostopwords_file = []
for sentence in corpus:
  for word in sentence:
    if word not in stop_words:
      corpus_nostopwords_file.append(word)

corpus_nostopwords = []
for sentence in corpus:
  new_sentence = [word for word in sentence if word not in stop_words]
  corpus_nostopwords.append(new_sentence)

with open("out.txt", "w") as file:
  for word in corpus_nostopwords_file:
	  file.write(word+'\n')
  file.close()

# Creating the Word2Vec model
model = Word2Vec(corpus_nostopwords)
model

# output the model
model.save('w2v.model')

def most_similar_func (word):
  return model.wv.most_similar(word,topn=20)

