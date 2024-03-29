# -*- coding: utf-8 -*-
"""
Write a python script to perform the following data preparation activities: 
1. Tokenize the corpus 
2. Remove the following special characters: !"#$%&()*+/:;<=>@[\\]^`{|}~\t\n 
3. Create two versions of your dataset: (1) with stopwords and (2) without stopwords. 
Stopword lists are available online. 
4. Randomly split your data into training (80%), validation (10%) and test (10%) sets
"""

!wget -q https://raw.githubusercontent.com/fuzhenxin/textstyletransferdata/master/sentiment/neg.txt
!wget -q https://raw.githubusercontent.com/fuzhenxin/textstyletransferdata/master/sentiment/pos.txt
!wget -q "https://gist.githubusercontent.com/sebleier/554280/raw/7e0e4a1ce04c2bb7bd41089c9821dbcf6d0c786c/NLTK's%2520list%2520of%2520english%2520stopwords"

import csv
import random

special_characters = "!\"#$%&()*+/:;<=>@[\]^`{|}~\t\n"
tokenized_sentences = []

def tokenize_func (filename,given_set):
  with open(filename) as f:
    for line in f:
      for char in special_characters:
        l = line.replace(char,' ')
      tokenized_sentences.append(l.split())

tokenize_func('neg.txt',tokenized_sentences)
tokenize_func('pos.txt',tokenized_sentences)

def write_to_csv (filename, lst):
  with open(filename, 'w') as csvfile: 
    writer = csv.writer(csvfile) 
    writer.writerows(lst)

write_to_csv("out.csv", tokenized_sentences)

stop_words = []
with open("NLTK's list of english stopwords") as f:
  for line in f:
    stop_words.append(line.split('\n')[0])

no_stopwords = []
for sentence in tokenized_sentences:
  new_sentence = [word for word in sentence if word not in stop_words]
  no_stopwords.append(new_sentence)

write_to_csv("out_ns.csv", no_stopwords)

training_num = round(len(tokenized_sentences)*0.8)
validation_num = round(len(tokenized_sentences)*0.1)
testing_num = round(len(tokenized_sentences)*0.1)

random.shuffle(tokenized_sentences)
training = tokenized_sentences[:training_num]
validation = tokenized_sentences[training_num:validation_num+training_num]
test = tokenized_sentences[validation_num+training_num:]

write_to_csv("training.csv",training)
write_to_csv("val.csv",validation)
write_to_csv("test.csv",test)

training_num = round(len(no_stopwords)*0.8)
validation_num = round(len(no_stopwords)*0.1)
testing_num = round(len(no_stopwords)*0.1)

random.shuffle(no_stopwords)
training_ns = no_stopwords[:training_num]
validation_ns = no_stopwords[training_num:validation_num+training_num]
test_ns = no_stopwords[validation_num+training_num:]

write_to_csv("training_ns.csv",training_ns)
write_to_csv("val_ns.csv",validation_ns)
write_to_csv("test_ns.csv",test_ns)

