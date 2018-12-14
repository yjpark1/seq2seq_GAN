"""
input: short text (integer sequence)
model: input -> embedding -> RNN -> output
output: real/fake logit
"""
## A RNN discriminator model for text classification
import math
import tensorflow as tf


class RNNDiscriminator:
    def __init__(self, emb, num_classes=2, vocab_size=1000,
                 embedding_units=128, hidden_units=64):
        self.num_classes = 1 if num_classes is 2 else num_classes
        self.vocab_size = vocab_size
        self.embedding_units = embedding_units
        self.hidden_units = hidden_units
        self.embeddings = emb

    def build_model(self, inputs_seq, reuse=False):
        # - embedding layer - #
        # embedding layer is firstly defined in generator
        with tf.variable_scope('', reuse=True):
            embedding_kernel = tf.get_variable('embedding')
        embed_dense = lambda x: tf.matmul(x, embedding_kernel)
        inputs_embed = embed_dense(inputs_seq)
        # inputs_embdded = self.embeddings(inputs_seq)


        with tf.variable_scope('discriminator', reuse=reuse):
            # rnn cell 
            cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_units)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell,
                                                 output_keep_prob=.5)
            rnn = tf.contrib.rnn.FusedRNNCellAdaptor(cell, True)

            ## TODO :: multiple layer
            input_enc = tf.transpose(inputs_embed, (1, 0, 2))
            rnn_output, _ = rnn(input_enc, dtype=tf.float32)
            rnn_output = tf.transpose(rnn_output, (1, 0, 2))

            # Reduces to binary prediction.
            logits = tf.layers.dense(rnn_output[:, -1, :], self.num_classes)
            preds = tf.sigmoid(logits)
            print("build discriminator done!")
        return logits, preds

