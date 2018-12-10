import numpy as np
import pandas as pd
import keras
from keras.preprocessing import text, sequence
from tqdm import tqdm
from collections import Counter
from itertools import chain
import matplotlib.pyplot as plt


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, text, summary_input, summary_target, num_token_output, batch_size=16):
        """Initialization"""
        self.text = text
        self.summary_input = summary_input
        self.summary_target = summary_target
        self.batch_size = batch_size
        self.shuffle = False
        self.indexes = np.arange(len(text))
        self.num_token_output = num_token_output

    def sort_data(self):
        # make numpy array
        self.text = np.array(self.text)
        self.summary_input = np.array(self.summary_input)
        self.summary_target = np.array(self.summary_target)

        # sort by length: text, summary_input, summary_target
        text_length = [len(x) for x in self.text]
        index = np.argsort(text_length)

        self.text = self.text[index]
        self.summary_input = self.summary_input[index]
        self.summary_target = self.summary_target[index]

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.text) / self.batch_size))

    def __make_onehot_target(self, target):
        out = []
        for i, x in enumerate(target):
            x = np.array(x) - 1
            out.append(np.eye(self.num_token_output)[x])
        return out

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def __data_generation(self, indexes):
        """Generates data containing batch_size samples
        # X : (n_samples, *dim, n_channels)
        """
        # get batch data
        batch_text = self.text[indexes]
        batch_summary_input = self.summary_input[indexes]
        batch_summary_target = self.summary_target[indexes]

        max_len_txt = np.max([len(x) for x in batch_text])
        max_len_summ_input = np.max([len(x) for x in batch_summary_input])

        # preprocessing
        # 1) make onehot target
        batch_summary_target = self.__make_onehot_target(batch_summary_target)

        # 2) pad sequence
        batch_text = sequence.pad_sequences(batch_text, maxlen=max_len_txt, truncating='post', padding='pre')
        batch_summary_input = sequence.pad_sequences(batch_summary_input, maxlen=max_len_summ_input, padding='post')
        batch_summary_target = sequence.pad_sequences(batch_summary_target, maxlen=max_len_summ_input, padding='post')

        return [batch_text, batch_summary_input], batch_summary_target


if __name__ == "__main__":
    TextWithSummary = pd.read_csv('datasets/TextWithSummary.csv', encoding='utf8')
    TextWithoutSummary = pd.read_csv('datasets/TextWithoutSummary.csv', encoding='utf8')

    docs = TextWithSummary.summary.values.tolist() + TextWithSummary.text.values.tolist() + \
           TextWithoutSummary.text.values.tolist()
    docs = [x.split(' ') for x in docs]

    docs_flat = list(chain.from_iterable(docs))
    docs_cnt = Counter(docs_flat)
    print(len(docs_cnt))
    # 48828

    tokenizer = text.Tokenizer(filters='')
    tokenizer.fit_on_texts(docs)
    TextWithSummary_summary = tokenizer.texts_to_sequences(TextWithSummary.summary.values.tolist())
    TextWithSummary_text = tokenizer.texts_to_sequences(TextWithSummary.text.values.tolist())
    TextWithoutSummary_text = tokenizer.texts_to_sequences(TextWithoutSummary.text.values.tolist())

    len_summary = [len(x) for x in TextWithSummary_summary]
    len_text = [len(x) for x in TextWithSummary_text + TextWithoutSummary_text]

    np.mean(len_text)
    np.max(len_text)

    plt.hist(len_summary)
    plt.hist(len_text)
    plt.show()
