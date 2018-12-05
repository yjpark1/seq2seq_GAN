# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 10:55:51 2018

@author: HQ
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tqdm import tqdm
from collections import Counter
from itertools import chain
from contextlib import contextmanager

import numpy as np
import pandas as pd
import glob
import re
from konlpy.tag import Komoran
import tensorflow as tf

special_token = {"<PAD>" : 0, "<UNK>" : 1, "<START>" : 2, "<END>" : 3}

#path = "data/*.csv"
def read_labeled_file(paths):
    # <read labeled text files>
    data = []
    for p in paths:
        file = pd.read_csv(p, encoding='cp949', engine='python')
        data.append(file)
    data = pd.concat(data)

    # <split text & summary>
    summary = data.body.values
    text = data.title.values + data.origin.values

    return summary, text


def remove_token(x):
    x = re.sub('\n+', ' ', x)
    x = re.sub('[!@#Δ\'\`·…]', '', x)
    x = re.sub('\s+', ' ', x)
    return x

def morph_num(x):
    komoran = Komoran()
    x = komoran.pos(x)
    out = []
    for mrph, tg in x:
        if tg in ['SN']:
            mrph = '<num>'
        out.append(mrph)
    return out

def tokenizer(summary, text):
    doc = summary + text
    doc = morph_num(remove_token(doc))
    return doc

def build_vocab(summary, text, min_counts = 5):
    print("number of docs :", len(summary))
    counts = Counter()
    for i in tqdm(range(len(summary))):
        tokens = tokenizer(summary[i], text[i])
        for token in tokens:
            counts[token] += 1
    # get number of tokens
    word2idx = {word: idx + len(special_token) for idx, (word, count) in enumerate(counts.most_common()) if count > min_counts}
    word2idx.update(special_token)
    
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word

paths = glob.glob('data/*.csv')
summary, text = read_labeled_file(paths)
word2idx, idx2word = build_vocab(summary, text)

def vectorize(line, word2idx):
    summary_token, text_token = tokenizer(line)
    
    summary_vec = [word2idx.get(token, special_token["<UNK>"]) for token in summary_token]
    summary_vec = [special_token["<START>"] + summary_vec + special_token["<END>"]]
    
    text_vec = [word2idx.get(token, special_token["<UNK>"]) for token in text_token]
    text_vec = [special_token["<START>"] + text_vec + special_token["<END>"]]

    return summary_vec, text_vec
    

def preprocess(data):
    seq_length = tf.reduce_mean(tf.cast(tf.not_equal(data, 0), dtype = tf.int32), axis = 1)
    max_seq_len = tf.reduce_max(seq_length)
    data = data[:, :max_seq_len]
    
    source = data[:, :-1]
    target = data[:, 1:]
    seq_length -= -1
    return source, target, seq_length

###############################################################################
path = "data/*.csv"

word2idx, idx2word = build_vocab(path)

    
    
    
    
    
    