# -*- coding: utf-8 -*-
"""
MSCI 598 - Assignment 2

Nimisha Saxena - 20716444
"""

!wget -q https://raw.githubusercontent.com/fuzhenxin/textstyletransferdata/master/sentiment/neg.txt
!wget -q https://raw.githubusercontent.com/fuzhenxin/textstyletransferdata/master/sentiment/pos.txt
!wget -q "https://gist.githubusercontent.com/sebleier/554280/raw/7e0e4a1ce04c2bb7bd41089c9821dbcf6d0c786c/NLTK's%2520list%2520of%2520english%2520stopwords"

import sys
import numpy as np
import pandas as pd
import string
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk import word_tokenize
nltk.download('punkt')

pos_text = pd.read_csv("pos.txt", delimiter='\t')
neg_text = pd.read_csv("neg.txt", delimiter='\t')

pos_text.columns = ['text']
pos_text['label'] = 1
neg_text.columns = ['text']
neg_text['label'] = 0

corpus = pd.concat([pos_text, neg_text])
corpus.reset_index(inplace=True)
del corpus['index']

corpus['text'] = corpus['text'].str.replace('[^\w\s]', '')
corpus['text'] = corpus['text'].str.lower()

corpus['text'] = corpus['text'].apply(word_tokenize)

corpus_nostopwords = corpus.copy()

stop_words = []
with open("NLTK's list of english stopwords") as f:
  for line in f:
    stop_words.append(line.split('\n')[0])

corpus_nostopwords['text'] = corpus_nostopwords['text'].apply(lambda x: [item for item in x if item not in stop_words])
corpus_nostopwords['text'] = corpus_nostopwords['text'].apply(lambda x: " ".join(x))
corpus['text'] = corpus['text'].apply(lambda x: " ".join(x))
corpus_nostopwords['text']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(corpus_nostopwords['text'], corpus_nostopwords['label'], test_size=0.2, random_state=42)

"""Unigrams no stop words"""

x_test = x_test[:int(len(x_test) * 0.5)]
x_val = x_test[int(len(x_test) * 0.5):]
y_test = y_test[:int(len(y_test) * 0.5)]
y_val = y_test[int(len(y_test) * 0.5):]

count_vect = CountVectorizer()
count_vect.fit(x_train)
x_train_t = count_vect.transform(x_train)
x_test_t = count_vect.transform(x_test)
x_tval_t = count_vect.transform(x_val)

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(x_train_t, y_train)

file_1 = 'mnb_uni_ns'

with open(file_1, 'wb') as f:
  pickle.dump((model, count_vect), f)

print(model.score(x_test_t, y_test))

"""Unigrams no stop words"""

count_vect = CountVectorizer(ngram_range=(2, 2))
count_vect.fit(x_train)
x_train_t = count_vect.transform(x_train)
x_test_t = count_vect.transform(x_test)
x_tval_t = count_vect.transform(x_val)

model = MultinomialNB()
model.fit(x_train_t, y_train)

file_2 = 'mnb_bi_ns'
with open(file_2, 'wb') as f:
  pickle.dump((model, count_vect), f)

print(model.score(x_test_t, y_test))

"""Unigram+Bigram no stop words

"""

count_vect = CountVectorizer(ngram_range=(1, 2))
count_vect.fit(x_train)
x_train_t = count_vect.transform(x_train)
x_test_t = count_vect.transform(x_test)
x_tval_t = count_vect.transform(x_val)

model = MultinomialNB()
model.fit(x_train_t, y_train)

file_3 = 'mnb_uni_bi_ns'
with open(file_3, 'wb') as f:
  pickle.dump((model, count_vect), f)

print(model.score(x_test_t, y_test))

"""Unigram with stop words


"""

x_train, x_test, y_train, y_test = train_test_split(corpus['text'], corpus['label'], test_size=0.2, random_state=42)

x_test = x_test[:int(len(x_test) * 0.5)]
x_val = x_test[int(len(x_test) * 0.5):]
y_test = y_test[:int(len(y_test) * 0.5)]
y_val = y_test[int(len(y_test) * 0.5):]

count_vect = CountVectorizer()
count_vect.fit(x_train)
x_train_t = count_vect.transform(x_train)
x_test_t = count_vect.transform(x_test)
x_tval_t = count_vect.transform(x_val)

model = MultinomialNB()
model.fit(x_train_t, y_train)
print(model.score(x_test_t, y_test))

file_4 = 'mnb_uni'
with open(file_4, 'wb') as f:
  pickle.dump((model, count_vect), f)

"""Bigram with stop words"""

count_vect = CountVectorizer(ngram_range=(2, 2))
count_vect.fit(x_train)
x_train_t = count_vect.transform(x_train)
x_test_t = count_vect.transform(x_test)
x_tval_t = count_vect.transform(x_val)

model = MultinomialNB()
model.fit(x_train_t, y_train)

file_5 = 'mnb_bi'
with open(file_5, 'wb') as f:
  pickle.dump((model, count_vect), f)

print(model.score(x_test_t, y_test))

"""Unigram+Bigram with stop words

"""

count_vect = CountVectorizer(ngram_range=(1, 2))
count_vect.fit(x_train)
x_train_t = count_vect.transform(x_train)
x_test_t = count_vect.transform(x_test)
x_tval_t = count_vect.transform(x_val)

model = MultinomialNB()
model.fit(x_train_t, y_train)

file_6 = 'mnb_uni_bi'
with open(file_6, 'wb') as f:
  pickle.dump((model, count_vect), f)

print(model.score(x_test_t, y_test))

"""Tuning the model"""

from sklearn.model_selection import GridSearchCV
parameters = {'alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001)}
grid_search = GridSearchCV(model, parameters)
grid_search.fit(x_tval_t, y_val)

grid_search.score(x_test_t, y_test)
