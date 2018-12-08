"""
input: short text (integer sequence)
model: input -> embedding -> RNN -> output
output: real/fake logit
"""
## A RNN discriminator model for text classification
import math
import tensorflow as tf


class RNNDiscriminator:
    def __init__(self, emb_scope, num_classes=2, vocab_size=1000,
                 embedding_units=128, hidden_units=64):
        self.num_classes = 1 if num_classes is 2 else num_classes
        self.vocab_size = vocab_size
        self.embedding_units = embedding_units
        self.hidden_units = hidden_units
        self.emb_scope = emb_scope

    def embeddings(self, inputs, emb_scope, reuse):
        with tf.variable_scope(emb_scope, reuse=reuse):
            embed = tf.layers.dense(inputs, self.embedding_units,
                                    name='embedding', use_bias=False)
        return embed

    def build_model(self, inputs_seq, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            # - embedding layer - #
            # embedding layer is firstly defined in generator
            inputs_embdded = self.embeddings(inputs_seq, self.emb_scope, reuse=True)
            
            # rnn cell 
            cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_units)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell,
                                                 output_keep_prob=.5)
            rnn = tf.contrib.rnn.FusedRNNCellAdaptor(cell, True)

            ## TODO :: multiple layer
            input_enc = tf.transpose(inputs_embdded, (1, 0, 2))
            rnn_output, _ = rnn(input_enc, dtype=tf.float32)
            rnn_output = tf.transpose(rnn_output, (1, 0, 2))

            # Reduces to binary prediction.
            logits = tf.layers.dense(rnn_output[:, -1, :], self.num_classes)
            preds = tf.sigmoid(logits)
            print("build discriminator done!")
        return logits, preds

if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    model = RNNDiscriminator(num_classes=2, vocab_size=1000, 
                             embedding_units=128, hidden_units=64)
    inputs = tf.placeholder(dtype=tf.int32, shape=(32, None), name='inputs_text')
    outputs = model.build_model(inputs)
    # output specification