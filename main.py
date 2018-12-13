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
from keras.preprocessing import text
import pandas as pd
import codecs

## local import
from models.discriminator import RNNDiscriminator
from models.generator import Seq2SeqGenerator
from models.gan import SeqGAN
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

# filter short article out
TextWithoutSummary_text = [x for x in TextWithoutSummary_text if len(x) > H.max_summary_len]

# add <end> token
TextWithSummary_summary = [Token_startend(x) for x in TextWithSummary_summary]
TextWithSummary_text = [Token_startend(x) for x in TextWithSummary_text]
TextWithoutSummary_text = [Token_startend(x) for x in TextWithoutSummary_text]

docs = TextWithSummary_summary + TextWithSummary_text + TextWithoutSummary_text + ['<start>']
docs = [x.split(' ') for x in docs]

tokenizer = text.Tokenizer(num_words=H.vocab_size,
                           filters='◎△▶°•㎡\t\n▲◆®©!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',
                           oov_token='<UNK>')
tokenizer.fit_on_texts(docs)
tokenizer.index_word[0] = '<PAD>'

TextWithSummary_summary = tokenizer.texts_to_sequences(TextWithSummary_summary)
TextWithSummary_text = tokenizer.texts_to_sequences(TextWithSummary_text)
TextWithoutSummary_text = tokenizer.texts_to_sequences(TextWithoutSummary_text)

###############################################################################
## build model
tf.reset_default_graph()
sess = tf.Session()
scope = tf.get_variable_scope()
gumbel_temp = tf.placeholder(tf.float32)
discriminator = RNNDiscriminator(emb_scope=scope, num_classes=2, vocab_size=H.vocab_size,
                                 embedding_units=H.embedding_units, hidden_units=64)
generator = Seq2SeqGenerator(emb_scope=scope, namescope='generator', temp=gumbel_temp, vocab_size=H.vocab_size,
                             embedding_units=H.embedding_units, enc_units=H.rnn_units, dec_units=H.rnn_units,
                             tokenizer=tokenizer)
reconstructor = Seq2SeqGenerator(emb_scope=scope, namescope='reconstructor', temp=gumbel_temp, vocab_size=H.vocab_size,
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
indvL_D, indvL_G, indvL_R = gan.build_gan(inputs=next_element)

print('start training GAN model')
gan.sess.run(iterator.initializer)

losses = {"d": [], "g": [], "r": []}
num_steps = int(np.ceil(len(TextWithSummary_summary) / H.batch_size))
print(num_steps)

for epoch in range(1, num_epoch + 1):
    start = time.time()
    train_L_D = 0.
    train_L_G = 0.
    train_L_R = 0.

    for step in range(1, num_steps + 1):
        _, batch_g_loss, batch_d_loss, batch_r_loss, bch_indvL_D, bch_indvL_G, bch_indvL_R = gan.sess.run(
            [gan.train_op, gan.gen_loss, gan.dis_loss, gan.rec_loss] +\
            [indvL_D, indvL_G, indvL_R],
            feed_dict={gumbel_temp: max(10 * 0.95 ** epoch, 1e-3)}
        )

        print('★step: {}, D: {:.3f}, G: {:.3f}, R: {:.3f}'.format(step, batch_d_loss, batch_g_loss, batch_r_loss))
        print('D(real): {:.3f}, D(fake): {:.3f}'.format(bch_indvL_D[0], bch_indvL_D[1]), end=' | ')
        print('G(real): {:.3f}, D(fake): {:.3f}'.format(bch_indvL_G[0], bch_indvL_G[1]), end=' | ')
        print('R(real_model): {:.3f}, R(fake): {:.3f}, R(real): {:.3f}'.format(bch_indvL_R[0],
                                                                               bch_indvL_R[1],
                                                                               bch_indvL_R[2]))

        train_L_D += batch_d_loss
        train_L_G += batch_g_loss
        train_L_R += batch_r_loss

        if step % 10 == 0:
            test_out = gan.sess.run([gan.generated_sequence, gan.unlabeled_text],
                                    feed_dict={gumbel_temp: 1e-3})
            sequence = test_out[0]
            sequence = np.argmax(sequence, axis=2)
            sequence = tokenizer.sequences_to_texts(sequence)
            for s in sequence:
                print(s, end='\n\n')
    
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

    ckpt_path = gan.saver.save(gan.sess, "saved/", epoch)

    if epoch % 1 == 0:
        test_out = gan.sess.run([gan.generated_sequence, gan.unlabeled_text],
                                feed_dict={gumbel_temp: 1e-3})
        sequence = test_out[0]
        origin = test_out[1]
        # shape = (batch, num_samples, summary_length, vocab_size)
        sequence = np.argmax(sequence, axis=2)
        # shape = (batch, summary_lenth)
        sequence = tokenizer.sequences_to_texts(sequence)
        for s in sequence:
            print(s, end='\n\n')

        origin = np.argmax(origin, axis=2)
        origin = tokenizer.sequences_to_texts(origin)

        f = open("summary_history/fake_summary_{}.txt".format(epoch), 'w')
        for fake_summary in sequence:
            fake_summary = fake_summary + '\n'
            f.write(fake_summary)

        f.write('<origin>\n\n\n')
        for txt in origin:
            txt = txt + '\n'
            f.write(txt)

        f.close()
