# -*- coding: utf-8 -*-
"""
Created on Dec  11 11:26:44 2018

@author: Yooyeon
"""

import numpy as np
from keras.preprocessing.sequence import pad_sequences

PAD_TOKEN = '[PAD]'
UNKNOWN_TOKEN = '[UNK]'
START = '[START]'
STOP = '[STOP]'



class Vocab(object):
  def __init__(self, vocab_file):
      ## TODO :: min_count ??
    self._word_to_id = {}
    self._id_to_word = {}
    self._count = 0 

    # [PAD], [UNK], [START] and [STOP] get the ids 0,1,2,3.
    for w in [PAD_TOKEN, UNKNOWN_TOKEN, START, STOP]:
      self._word_to_id[w] = self._count
      self._id_to_word[self._count] = w
      self._count += 1
      
    with open(vocab_file, 'r',encoding='cp949') as vocab_f:
      for w in vocab_f:
        w = w[:-1]
        self._word_to_id[w] = self._count
        self._id_to_word[self._count] = w
        self._count += 1
        #print(self._count)
        #print(w)
    print("Finished constructing vocabulary of %i total words." % (self._count))
    
  def word2id(self, word):
    if word not in self._word_to_id:
      return self._word_to_id[UNKNOWN_TOKEN]
    return self._word_to_id[word]

  def id2word(self, word_id):
    return self._id_to_word[word_id]

  def size(self):
    return self._count

def addtoken(input_summary,vocab):
    tmp = np.array([[START]+doc+[STOP] for doc in input_summary])
    return tmp


def one_hot(inputs, vocab_size, max_seq_len):
    
    padded_inputs = pad_sequences(inputs, padding='post', maxlen=max_seq_len)

    batch_one_hot = (np.arange(vocab_size) == padded_inputs[..., None]).astype(int)
    batch_one_hot = batch_one_hot.astype(np.float32)
    return batch_one_hot

def find_overlength(input_length, max_seq_len = 200):
    batch = len(input_length)
    for doc in range(batch):
        if input_length[doc] > max_seq_len:
            input_length[doc] = max_seq_len
    return input_length
    
def labeled_article_generator(input_article, vocab, batch_size):
    vocab_size = vocab._count
    
    input_article = input_article.T
    l = len(input_article)
    
    for ndx in range(0,l, batch_size):
        input_article_ = input_article[ndx:(ndx + batch_size)]
        input_article_id = [[vocab.word2id(words) for words in doc] for doc in input_article_]
        
        output_article_len = [len(article) for article in input_article_id]
        output_article_len = find_overlength(output_article_len, max_seq_len = 1000)
        output_article_len = np.array(output_article_len, dtype = np.int32)
        
        output_article = one_hot(input_article_id, vocab_size, max_seq_len = 1000)
        yield output_article, output_article_len

def labeled_summary_generator(input_summary, vocab, batch_size):
    vocab_size = vocab._count
    
    input_summary = addtoken(input_summary,vocab)
    input_summary = input_summary.T
    
    l = len(input_summary)
    
    for ndx in range(0,l, batch_size):
        input_summary_ = input_summary[ndx:(ndx + batch_size)]
        input_summary_id = [[vocab.word2id(words) for words in doc] for doc in input_summary_]
        
        output_summary_len = [len(summary) for summary in input_summary_id]
        output_summary_len = find_overlength(output_summary_len, max_seq_len = 200)
        output_summary_len = np.array(output_summary_len, dtype = np.int32)
        
        output_summary = one_hot(input_summary_id, vocab_size, max_seq_len = 200)
        
        yield output_summary, output_summary_len

def unlabeled_article_generator(input_article, vocab, batch_size):
    vocab_size = vocab._count
    l = len(input_article)
    
    for ndx in range(0, l, batch_size):
        input_article_ = input_article[ndx:(ndx + batch_size)]
        input_article_id = [[vocab.word2id(words) for words in doc] for doc in input_article_]
        
        output_article_len = [len(article) for article in input_article_id]
        output_article_len = find_overlength(output_article_len, max_seq_len = 1000)
        output_article_len = np.array(output_article_len, dtype = np.int32)
        
        output_article = one_hot(input_article_id, vocab_size, max_seq_len = 1000)
        yield output_article, output_article_len
        
        
def random_sample_generator(input_article, vocab, num_sample):
    vocab_size = vocab._count
    l = len(input_article)
    
    idx = np.random.choice(l, num_sample)
    
    input_article_ =input_article[idx]
    input_article_id = [[vocab.word2id(words) for words in doc] for doc in input_article_]
    
    output_article_len = [len(article) for article in input_article_id]
    output_article_len = find_overlength(output_article_len, max_seq_len = 1000)
    output_article_len = np.array(output_article_len, dtype = np.int32)
    
    output_article = one_hot(input_article_id, vocab_size, max_seq_len = 1000)
    origin_doc = input_article_
    
    yield output_article, output_article_len, origin_doc         
               

def detokenize(prediction, build_vocab):
    ## prediction shape - [batch_size, length] (argmax)
    ## build_vocab 
    id2word = build_vocab._id_to_word
    
    generated_summary = []
    for line in prediction:
        tmp_summary= []
        for idx in line:
            w = id2word[idx]
            tmp_summary.append(w)
            if w == '[STOP]':
                break
        print(" ".join(tmp_summary))
        generated_summary.append(tmp_summary)
        
    return generated_summary
        
'''
build_vocab = Vocab()
label_gen = labeled_article_summary_generator(labeled_article,labeled_summary, build_vocab, batch_size)
k = next(label_gen)
k[0]#article id

k[1]#summary id

unlabel_gen = unlabeled_article_generator(unlabeled_article,build_vocab, batch_size)
n = next(unlabel_gen) #articlshape
'''


        


