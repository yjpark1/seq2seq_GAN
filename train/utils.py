import numpy as np
import pandas as pd
import keras
from keras.preprocessing import text, sequence
from tqdm import tqdm
from collections import Counter
from itertools import chain
import matplotlib.pyplot as plt
import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class Generator:
    def __init__(self, x, max_len=1000):
        self.x = x
        self.max_len = max_len

    def gen_data(self):
        for b in self.x:
            b = sequence.pad_sequences([b], maxlen=self.max_len, truncating='post', padding='post')
            b = keras.utils.to_categorical(b, num_classes=24353)
            yield b[0]

    def gen_len(self):
        for b in self.x:
            yield len(b)


def Token_startend(x):
    return '<START> ' + x + '<END>'


if __name__ == "__main__":
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

    docs_flat = list(chain.from_iterable(docs))
    docs_cnt = Counter(docs_flat)
    print(len(docs_cnt))

    sorted_by_value = sorted(docs_cnt.items(), key=lambda kv: kv[1])
    sorted_value = np.array([x[1] for x in sorted_by_value])
    plt.hist(sorted_value, log=True)
    plt.show()
    plt.hist(sorted_value, range=(0, 100), log=True)
    plt.show()
    print('number of tokens more than 10: ', sum(sorted_value > 10))
    # 24353

    tokenizer = text.Tokenizer(num_words=24353, filters='', oov_token='<UNK>')
    tokenizer.fit_on_texts(docs)
    TextWithSummary_summary = tokenizer.texts_to_sequences(TextWithSummary_summary)
    TextWithSummary_text = tokenizer.texts_to_sequences(TextWithSummary_text)
    TextWithoutSummary_text = tokenizer.texts_to_sequences(TextWithoutSummary_text)

    len_summary = [len(x) for x in TextWithSummary_summary]
    len_text = [len(x) for x in TextWithSummary_text + TextWithoutSummary_text]

    np.mean(len_text)
    np.max(len_text)

    plt.hist(len_summary)
    plt.hist(len_text)
    plt.show()

    # tokenizer.texts_to_sequences([['dfder', '가', '위촉']])
    # tokenizer.sequences_to_texts([[1, 20, 1]])

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
                                 len_ulbl_text.repeat())).batch(16)

    iterator = dcomb.make_initializable_iterator()
    # extract an element
    next_element = iterator.get_next()
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for i in range(1):
            val = sess.run(next_element)
            print(val)

