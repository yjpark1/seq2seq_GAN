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
# import utils.data_file as helper
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
num_steps = 1000
###############################################################################
## define functions
def plot_loss(losses):
    plt.figure(figsize=(5, 4))
    plt.plot(losses["d"], label='discriminitive loss')
    plt.plot(losses["g"], label='generative loss')
    plt.plot(losses["r"], label='generative loss')
    plt.legend()
    plt.show()


###############################################################################
## build model
tf.reset_default_graph() 

init = tf.global_variables_initializer()
sess = tf.Session()
scope = tf.get_variable_scope()

discriminator = RNNDiscriminator(emb_scope=scope, num_classes=2, vocab_size=H.vocab_size,
                                 embedding_units=64, hidden_units=64)
generator = Seq2SeqGenerator(emb_scope=scope, namescope='generator', vocab_size=H.vocab_size,
                             embedding_units=64, enc_units=128, dec_units=128)
reconstructor = Seq2SeqGenerator(emb_scope=scope, namescope='reconstructor', vocab_size=H.vocab_size,
                                 embedding_units=64, enc_units=128, dec_units=128)
gan = SeqGAN(sess, discriminator, generator, reconstructor, emb_scope=scope)


######################################################################
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

tokenizer = text.Tokenizer(num_words=24353, filters='', oov_token='<UNK>')
tokenizer.fit_on_texts(docs)
TextWithSummary_summary = tokenizer.texts_to_sequences(TextWithSummary_summary)
TextWithSummary_text = tokenizer.texts_to_sequences(TextWithSummary_text)
TextWithoutSummary_text = tokenizer.texts_to_sequences(TextWithoutSummary_text)

gen_lbl_summary = Generator(TextWithSummary_summary, max_len=200)
gen_lbl_text = Generator(TextWithSummary_text, max_len=1000)
gen_ulbl_text = Generator(TextWithoutSummary_text, max_len=1000)

lbl_summary = tf.data.Dataset().from_generator(gen_lbl_summary.gen_data,
                                               output_types=tf.float32,
                                               output_shapes=(tf.TensorShape([200, 24353])))

lbl_text = tf.data.Dataset().from_generator(gen_lbl_text.gen_data,
                                            output_types=tf.float32,
                                            output_shapes=(tf.TensorShape([1000, 24353])))

ulbl_text = tf.data.Dataset().from_generator(gen_ulbl_text.gen_data,
                                             output_types=tf.float32,
                                             output_shapes=(tf.TensorShape([1000, 24353])))

len_lbl_summary = tf.data.Dataset().from_generator(gen_lbl_summary.gen_len,
                                                   output_types=tf.int32,
                                                   output_shapes=(tf.TensorShape([])))

len_lbl_text = tf.data.Dataset().from_generator(gen_lbl_text.gen_len,
                                                output_types=tf.int32,
                                                output_shapes=(tf.TensorShape([])))

len_ulbl_text = tf.data.Dataset().from_generator(gen_ulbl_text.gen_len,
                                                 output_types=tf.int32,
                                                 output_shapes=(tf.TensorShape([])))

dcomb = tf.data.Dataset.zip((lbl_summary.repeat(), lbl_text.repeat(), ulbl_text.repeat(),
                             len_lbl_summary.repeat(), len_lbl_text.repeat(),
                             len_ulbl_text.repeat())).batch(4)
iterator = dcomb.make_initializable_iterator()
# extract an element
next_element = iterator.get_next()
gan.build_gan(inputs=next_element)

print('start training GAN model')
gan.sess.run(init)
gan.sess.run(iterator.initializer)
losses = {"d": [], "g": [], "r": []}

for epoch in range(1, num_epoch+1):
    start = time.time()   
    train_L_D = 0.
    train_L_G = 0.
    train_L_R = 0.
    
    for step in tqdm(range(1, num_steps+1)):
        _, batch_g_loss, batch_d_loss, batch_r_loss = gan.sess.run(
                [gan.train_op, gan.gen_loss, gan.dis_loss, gan.rec_loss],
                )

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
    print('Time: ', np.round(done-start, 3),
          ' Epoch:', epoch,
          ' | D loss:', train_L_D,
          ' | G loss:', train_L_G,
          ' | R loss:', train_L_R)

    # if epoch % 10 == 0:
    #     generated_sample = generate(gan, 1)
    #     print(generated_sample)

