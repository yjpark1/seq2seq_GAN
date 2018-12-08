# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 21:07:29 2018

@author: HQ
"""

import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

def batch(inputs, max_seq_len = None, vocab_size = None):
    """ One hot vector"""
    #inputs = next(labeled_dat)
    batch_seq_len = [len(seq) for seq in inputs]
    batch_seq_len = np.array(batch_seq_len, dtype = np.int32)

    padded_inputs = pad_sequences(inputs, padding='post', maxlen = max_seq_len)
    batch_one_hot = (np.arange(vocab_size) == padded_inputs[...,None]).astype(int)    
    batch_one_hot = batch_one_hot.astype(np.float32)    
    return batch_one_hot, batch_seq_len

def random_sequences(length_from, length_to,
                     vocab_lower, vocab_upper,
                     batch_size):
    """ Generates batches of random integer sequences,
        sequence length in [length_from, length_to],
        vocabulary in [vocab_lower, vocab_upper]
    """
    if length_from > length_to:
            raise ValueError('length_from > length_to')

    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)
    
    while True:
        yield [
            np.random.randint(low=vocab_lower,
                              high=vocab_upper,
                              size=random_length()).tolist()
            for _ in range(batch_size)
        ]