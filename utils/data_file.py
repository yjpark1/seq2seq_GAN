# -*- coding: utf-8 -*-
"""
Created on Dec  11 11:26:44 2018

@author: Yooyeon
"""

import glob
import random
import struct
import csv
import numpy as np
import pandas as pd
from tensorflow.core.example import example_pb2


PAD_TOKEN = '[PAD]'
UNKNOWN_TOKEN = '[UNK]'
START = '[START]'
STOP = '[STOP]'
batch_size =16

vocab_file = 'Z:/1. 프로젝트/2018_한화Summary/komoran_preprocessed/vocab.csv'
abstractive_article_komoran = np.load('Z:/1. 프로젝트/2018_한화Summary/komoran_preprocessed/abstractive_article_komoran.npy')
abstractive_summary_komoran = np.load('Z:/1. 프로젝트/2018_한화Summary/komoran_preprocessed/abstractive_summary_komoran.npy')
extractive_article_komoran = np.load('Z:/1. 프로젝트/2018_한화Summary/komoran_preprocessed/extractive_article_komoran.npy')
extractive_summary_komoran = np.load('Z:/1. 프로젝트/2018_한화Summary/komoran_preprocessed/extractive_summary_komoran.npy')
unlabeled_article_komoran = np.load('Z:/1. 프로젝트/2018_한화Summary/komoran_preprocessed/unlabeled_article_komoran.npy')

labeled_article = np.concatenate((abstractive_article_komoran,extractive_article_komoran),axis=0)
labeled_summary = np.concatenate((abstractive_summary_komoran,extractive_summary_komoran),axis=0)
unlabeled_article = unlabeled_article_komoran

class Vocab(object):
  def __init__(self, vocab_file = vocab_file):
    self._word_to_id = {}
    self._id_to_word = {}
    self._count = 0 

    # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
    for w in [UNKNOWN_TOKEN, PAD_TOKEN, START, STOP]:
      self._word_to_id[w] = self._count
      self._id_to_word[self._count] = w
      self._count += 1

    with open(vocab_file, 'r',encoding='cp949') as vocab_f:
      
      for w in vocab_f:
        w = w[:-1]
        self._word_to_id[w] = self._count
        self._id_to_word[self._count] = w
        self._count += 1
        print(self._count)
        print(w)
    print("Finished constructing vocabulary of %i total words. Last word added: %s" % (self._count, self._id_to_word[self._count-1]))
  
  def word2id(self, word):
    if word not in self._word_to_id:
      return self._word_to_id[UNKNOWN_TOKEN]
    return self._word_to_id[word]

  def id2word(self, word_id):
    return self._id_to_word[word_id]

  def size(self):
    return self._count


def unlabeled_article2ids(self,article_words,vocab):
    ids = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in article_words:
        i = vocab.word2id(w)
        ids.append(i)
    return ids

def labeled_article2ids(self,article_words, vocab):
    ids = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in article_words:
        i = vocab.word2id(w)
        ids.append(i)
    return ids

def labeled_summary2ids(summary_words, vocab):
    ids = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in summary_words:
        i = vocab.word2id(w)
        ids.append(i)
    return ids

def outputids2words(id_list, vocab, article_oovs):
  words = []
  for i in id_list:
      w = vocab.id2word(i)
      words.append(w)
  return words


def addtoken(input_summary,vocab):
    tmp = np.array([[START]+doc+[STOP] for doc in input_summary])
    return tmp

def labeled_article_summary_generator(input_article,input_summary, vocab, batch_size):
    input_article = input_article.T
    input_summary = addtoken(input_summary,vocab)
    input_summary = input_summary.T
    l = len(input_article)
    for ndx in range(0,l, batch_size):
        input_article_ = input_article[ndx:(ndx + batch_size)]
        input_summary_ = input_summary[ndx:(ndx + batch_size)]
        input_article_id = [[vocab.word2id(words) for words in doc] for doc in input_article_]
        input_summary_id = [[vocab.word2id(words) for words in doc] for doc in input_summary_]
        output_article, output_summary = input_article_id, input_summary_id
        yield output_article, output_summary

def unlabeled_article_generator(input_article, vocab, batch_size):
    l = len(input_article)
    for ndx in range(0, l, batch_size):
        ndx =64
        input_article_ = input_article[ndx:(ndx + batch_size)]
        input_article_id = [[vocab.word2id(words) for words in doc] for doc in input_article_]
        output_article = input_article_id
        yield output_article

build_vocab = Vocab()
label_gen = labeled_article_summary_generator(labeled_article,labeled_summary, build_vocab, batch_size)
k = next(label_gen)
k[0]#article id
k[1]#summary id
unlabel_gen = unlabeled_article_generator(unlabeled_article,build_vocab, batch_size)
n = next(unlabel_gen) #article id


        


