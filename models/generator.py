"""
input: long text (integer sequence)
model: input -> embedding -> seq2seq(RNN+attention) -> output
output: short text (integer sequence)
"""
import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class Seq2SeqGenerator:
    def __init__(self, vocab_size, embedding_dim, enc_units, dec_units, batch_size):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.enc_units = enc_units
        self.dec_units = dec_units
        self.batch_size = batch_size

    def build_model(self):
        train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
        pass

    def build_encoder(self, inputs):
        with tf.device('/cpu:0'):
            embeddings = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_dim], -1.0, 1.0))
            # 행렬에 트레이닝 데이터를 지정
            embed = tf.nn.embedding_lookup(embeddings, inputs)

        with tf.variable_scope('encode'):
            lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.enc_units//2,
                                                   state_is_tuple=False, name='lstm_fw')
            lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=0.5)
            lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.enc_units//2,
                                                   state_is_tuple=False, name='lstm_bw')
            lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=0.5)

            outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, embed,
                                                              dtype=tf.float32)

            # 나온 결과 값을 [batch_size, n_steps, n_hidden] -> [n_steps, batch_size, n_hidden]
            # outputs_fw = tf.transpose(outputs[0], [1, 0, 2])
            # outputs_bw = tf.transpose(outputs[1], [1, 0, 2])

            # BLSTM은 나오는 결과 값을 합쳐 준다.
            outputs_concat = tf.concat(outputs, axis=-1)

            return outputs_concat, states

    def build_decoder(self, inputs):
        with tf.device('/cpu:0'):
            embeddings = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_dim], -1.0, 1.0))
            # 행렬에 트레이닝 데이터를 지정
            embed = tf.nn.embedding_lookup(embeddings, inputs)

        with tf.variable_scope('encode'):
            lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.enc_units//2,
                                                   state_is_tuple=False, name='lstm_fw')
            lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=0.5)
            lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.enc_units//2,
                                                   state_is_tuple=False, name='lstm_bw')
            lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=0.5)

            outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, embed,
                                                              dtype=tf.float32)

            # 나온 결과 값을 [batch_size, n_steps, n_hidden] -> [n_steps, batch_size, n_hidden]
            # outputs_fw = tf.transpose(outputs[0], [1, 0, 2])
            # outputs_bw = tf.transpose(outputs[1], [1, 0, 2])

            # BLSTM은 나오는 결과 값을 합쳐 준다.
            outputs_concat = tf.concat(outputs, axis=-1)

            return outputs_concat, states


if __name__ == '__main__':
    model = Seq2SeqGenerator(vocab_size=100,
                             embedding_dim=64,
                             enc_units=256,
                             dec_units=256,
                             batch_size=32)
    inputs = tf.placeholder(tf.int32, shape=[None, 16,])
    model.build_encoder(inputs)
