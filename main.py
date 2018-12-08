# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 17:18:02 2018

@author: HQ
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import tensorflow as tf
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

## local import
from models.discriminator import RNNDiscriminator
from models.generator import Seq2SeqGenerator
from models.gan import SeqGAN

## hyperparameters
from train import hyperparameter as H

## toy example generator
import utils.simulation as helpers

###############################################################################
## setting hyperparameter
batch_size = H.batch_size
vocab_size = H.vocab_size
text_len = H.max_text_len
summary_len = H.max_summary_len

num_epoch = 100
num_steps = 1000
###############################################################################
## define functions
def plot_loss(losses):
    plt.figure(figsize = (5,4))
    plt.plot(losses["d"], label='discriminitive loss')
    plt.plot(losses["g"], label='generative loss')
    plt.plot(losses["r"], label='generative loss')
    plt.legend()
    plt.show()


def next_fd(dat, max_len = text_len, vocab_size = vocab_size): 
    batch = next(dat)
    batch_seq, seq_len = helpers.batch(batch, max_seq_len=max_len, vocab_size = vocab_size)
    return batch_seq, seq_len
 
    

def generate(model, num_samples=10): ## generate sample function 
    generated_sample =helpers.random_sequences(length_from=3, length_to=text_len,
                                               vocab_lower=10, vocab_upper=400,
                                               batch_size = num_samples)
    gen_seq, gen_len = next_fd(generated_sample, max_len = text_len)
    
    sequence = model.sess.run([model.generated_sequence],
                              feed_dict = {model.unlabeled_text: gen_seq,
                                           model.unlabeled_text_lengths: gen_len})
    #shape = (1, num_samples, summary_length, vocab_size)
    
    sequence = np.argmax(sequence[0], axis = 2) #shape = (num_samples, summary_lenth)
    
    ## TODO: detokenize function (idx --> character)
    return sequence    
###############################################################################
## build model
tf.reset_default_graph() 

init = tf.global_variables_initializer()
sess = tf.Session()
scope = tf.get_variable_scope()

discriminator = RNNDiscriminator(emb_scope=scope, num_classes=2, vocab_size=400,
                                 embedding_units=64, hidden_units=64)
generator = Seq2SeqGenerator(emb_scope=scope, namescope='generator', vocab_size=400,
                             embedding_units=64, enc_units=256, dec_units=256)
reconstructor = Seq2SeqGenerator(emb_scope=scope, namescope='reconstructor', vocab_size=400,
                                 embedding_units=64, enc_units=256, dec_units=256)
gan = SeqGAN(sess, discriminator, generator, reconstructor, emb_scope=scope)
gan.build_gan()

######################################################################
print("generate toy example...")
labeled_dat = helpers.random_sequences(length_from=3, length_to=text_len,
                                        vocab_lower=10, vocab_upper=400,
                                        batch_size = batch_size)
unlabeled_dat =helpers.random_sequences(length_from=3, length_to=text_len,
                                        vocab_lower=10, vocab_upper=400,
                                        batch_size = batch_size)
summary_dat = helpers.random_sequences(length_from=3, length_to=summary_len,
                                        vocab_lower=10, vocab_upper=400,
                                        batch_size = batch_size)


print('start training GAN model')


sess.run(init)
losses = {"d":[], "g":[], "r":[]}

for epoch in range(1, num_epoch+1):
    start = time.time()   
    train_L_D = 0.
    train_L_G = 0.
    train_L_R = 0.
    
    for step in tqdm(range(1, num_steps+1)):
        
        l_seq, l_len = next_fd(labeled_dat, max_len = text_len, vocab_size = 400)
        u_seq, u_len = next_fd(unlabeled_dat, max_len = text_len, vocab_size = 400)
        s_seq, s_len = next_fd(summary_dat, max_len = summary_len, vocab_size = 400)
        '''
        print("labeled sequence shape: ", l_seq.shape)
        print("unlabeled sequence shape: ", u_seq.shape)
        print("summary sequence shape: ", s_seq.shape)
        '''

        _, batch_g_loss, batch_d_loss, batch_r_loss = gan.sess.run(
                [gan.train_op, gan.gen_loss, gan.dis_loss, gan.rec_loss],
                feed_dict = {gan.labeled_text: l_seq,
                             gan.labeled_text_lengths: l_len,
                             gan.unlabeled_text: u_seq,
                             gan.unlabeled_text_lengths: u_len,
                             gan.real_summary: s_seq,
                             gan.real_summary_length: s_len})

        train_L_D += batch_d_loss
        train_L_G += batch_g_loss
        train_L_R += batch_r_loss
    
    train_L_D /= num_steps
    train_L_G /= num_steps
    train_L_R /= num_steps
    
    losses["d"].append(train_L_D)
    losses["g"].append(train_L_G)
    losses["r"].append(train_L_R)
    
    done = time.time()
    print('Time:',np.round(done-start,3), 'Epoch:', epoch,'| D loss:',train_L_D,'|G loss:', train_L_G,'|R loss:', train_L_R)

    if epoch % 10 == 0:
        generated_sample = generate(gan, 1)
        print(generated_sample)
