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
from keras.preprocessing import text, sequence
import pandas as pd

## local import
from models.discriminator import RNNDiscriminator
from models.generator import Seq2SeqGenerator
from models.gan import SeqGAN
# from train.utils import (lbl_summary, lbl_text, ulbl_text,
#                          len_lbl_summary, len_lbl_text,
#                          len_ulbl_text, tokenizer)
from train.utils import Generator, Token_startend

## hyperparameters
from train import hyperparameter as H

###############################################################################
## setting hyperparameter
batch_size = H.batch_size
vocab_size = H.vocab_size
text_len = H.max_text_len
summary_len = H.max_summary_len

num_epoch = 100
num_steps = 500


###############################################################################
## define functions
def plot_loss(losses):
    plt.figure(figsize=(5, 4))
    plt.plot(losses["d"], label='discriminitive loss')
    plt.plot(losses["g"], label='generative loss')
    plt.plot(losses["r"], label='generative loss')
    plt.legend()
    plt.show()


######################################################################
# tokenizer
print("load data..")
TextWithSummary = pd.read_csv('datasets/TextWithSummary.csv', encoding='utf8', dtype=object)
TextWithoutSummary = pd.read_csv('datasets/TextWithoutSummary.csv', encoding='utf8', dtype=object)

# add <UNK>, <START>, <END>
TextWithSummary_summary = TextWithSummary.summary.values.tolist()
TextWithSummary_text = TextWithSummary.text.values.tolist()
TextWithoutSummary_text = TextWithoutSummary.text.values.tolist()

TextWithSummary_summary = [Token_startend(x) for x in TextWithSummary_summary]
TextWithSummary_text = [Token_startend(x) for x in TextWithSummary_text]
TextWithoutSummary_text = [Token_startend(x) for x in TextWithoutSummary_text]

docs = TextWithSummary_summary + TextWithSummary_text + TextWithoutSummary_text
docs = [x.split(' ') for x in docs]

tokenizer = text.Tokenizer(num_words=H.vocab_size, filters='', oov_token='<UNK>')
tokenizer.fit_on_texts(docs)

TextWithSummary_summary = tokenizer.texts_to_sequences(TextWithSummary_summary)
TextWithSummary_text = tokenizer.texts_to_sequences(TextWithSummary_text)
TextWithoutSummary_text = tokenizer.texts_to_sequences(TextWithoutSummary_text)

###############################################################################
## build model
tf.reset_default_graph()
sess = tf.Session()
scope = tf.get_variable_scope()

discriminator = RNNDiscriminator(emb_scope=scope, num_classes=2, vocab_size=H.vocab_size,
                                 embedding_units=H.embedding_units, hidden_units=64)
generator = Seq2SeqGenerator(emb_scope=scope, namescope='generator', vocab_size=H.vocab_size,
                             embedding_units=H.embedding_units, enc_units=H.rnn_units, dec_units=H.rnn_units,
                             tokenizer=tokenizer)
reconstructor = Seq2SeqGenerator(emb_scope=scope, namescope='reconstructor', vocab_size=H.vocab_size,
                                 embedding_units=H.embedding_units, enc_units=H.rnn_units, dec_units=H.rnn_units,
                                 tokenizer=tokenizer)
gan = SeqGAN(sess, discriminator, generator, reconstructor, emb_scope=scope)

######################################################################
# train
gen_lbl_summary = Generator(TextWithSummary_summary, max_len=H.max_summary_len)
gen_lbl_text = Generator(TextWithSummary_text, max_len=H.max_text_len)
gen_ulbl_text = Generator(TextWithoutSummary_text, max_len=H.max_text_len)

lbl_summary = tf.data.Dataset().from_generator(gen_lbl_summary.gen_data,
                                               output_types=tf.float32,
                                               output_shapes=(tf.TensorShape([H.max_summary_len, H.vocab_size])))

lbl_text = tf.data.Dataset().from_generator(gen_lbl_text.gen_data,
                                            output_types=tf.float32,
                                            output_shapes=(tf.TensorShape([H.max_text_len, H.vocab_size])))

ulbl_text = tf.data.Dataset().from_generator(gen_ulbl_text.gen_data,
                                             output_types=tf.float32,
                                             output_shapes=(tf.TensorShape([H.max_text_len, H.vocab_size])))

len_lbl_summary = tf.data.Dataset().from_generator(gen_lbl_summary.gen_len,
                                                   output_types=tf.int32,
                                                   output_shapes=(tf.TensorShape([])))

len_lbl_text = tf.data.Dataset().from_generator(gen_lbl_text.gen_len,
                                                output_types=tf.int32,
                                                output_shapes=(tf.TensorShape([])))

len_ulbl_text = tf.data.Dataset().from_generator(gen_ulbl_text.gen_len,
                                                 output_types=tf.int32,
                                                 output_shapes=(tf.TensorShape([])))

dcomb = tf.data.Dataset.zip({'real_summary': lbl_summary.repeat(),
                             'labeled_text': lbl_text.repeat(),
                             'unlabeled_text': ulbl_text.repeat(),
                             'real_summary_len': len_lbl_summary.repeat(),
                             'labeled_text_len': len_lbl_text.repeat(),
                             'unlabeled_text_len': len_ulbl_text.repeat()}).batch(H.batch_size)

iterator = dcomb.make_initializable_iterator()
# extract an element
next_element = iterator.get_next()
# (g_real, g_fake, d_real, d_fake, r_real, r_target, r_fake,
#  d_loss, r_loss) = gan.build_gan(inputs=next_element)

r_real_logits, weight_l_txt, labeled_text, labeled_text_lengths = gan.build_gan(inputs=next_element)
gan.sess.run(iterator.initializer)
a = gan.sess.run([r_real_logits, weight_l_txt, labeled_text, labeled_text_lengths])
a[0].shape
a[1].shape
a[2].shape
a[3].shape

np.sum(a[1], axis=-1)
a[1][0]

print('start training GAN model')
gan.sess.run(iterator.initializer)

losses = {"d": [], "g": [], "r": []}

import math

for step in tqdm(range(1, num_steps + 1)):
    # (_, batch_g_loss, batch_d_loss, batch_r_loss,
    #  batch_g_real, batch_g_fake, batch_d_real, batch_d_fake,
    #  batch_r_real, batch_r_target, batch_r_fake, b_d_loss, b_r_loss
    #  ) = gan.sess.run(
    #     [gan.train_op, gan.gen_loss, gan.dis_loss, gan.rec_loss] +
    #     [g_real, g_fake, d_real, d_fake, r_real, r_target, r_fake, d_loss, r_loss],
    # )
    print('D: {:.3f}, G: {:.3f}, R: {:.3f}'.format(batch_d_loss, batch_g_loss, batch_r_loss))
    stop = False
    for a in b_d_loss + b_r_loss:
        if math.isnan(a):
            stop = True
            break
    if stop:
        break

