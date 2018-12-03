"""
input: long text (integer sequence)
model: input -> embedding -> seq2seq(RNN+attention) -> output
output: short text (integer sequence)

<reference>
1) https://gist.github.com/ilblackdragon/c92066d9d38b236a21d5a7b729a10f12
2) https://github.com/hccho2/RNN-Tutorial
"""
import tensorflow as tf
from models.utils import CustomGreedyEmbeddingHelper


MAX_LEN = 1000


class Seq2SeqGenerator:
    def __init__(self, vocab_size, embedding_dim, enc_units, dec_units, batch_size):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.enc_units = enc_units
        self.dec_units = dec_units
        self.batch_size = batch_size
        self.output_max_length = 300
        self._tokenID_start = 0
        self._tokenID_end = 1

    def build_model(self):
        train_inputs = tf.placeholder(tf.float32, shape=[self.batch_size, MAX_LEN, self.vocab_size])
        # input_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(train_inputs, 1)), -1)
        input_lengths = tf.placeholder(tf.int32, shape=[self.batch_size, ])
        self.start_tokens = tf.one_hot(tf.fill([self.batch_size, ], self._tokenID_start),
                                       self.vocab_size)

        self.emb_scope = tf.get_variable_scope()
        embed = self.embeddings(train_inputs, self.emb_scope, reuse=False)

        encoder_outputs, encoder_states = self.build_encoder(embed, input_lengths)
        decoder_outputs = self.build_decoder(encoder_outputs, encoder_states,
                                             input_lengths=input_lengths)

        return decoder_outputs

    def embeddings(self, inputs, emb_scope, reuse):
        with tf.device('/cpu:0'), tf.variable_scope(emb_scope, reuse=reuse):
            embed = tf.layers.dense(inputs, self.embedding_dim,
                                    name='embedding', use_bias=False)
            # embeddings = tf.trainable_variables('embeddings')
        return embed

    def build_encoder(self, embed, input_length):
        with tf.variable_scope('encode'):
            lstm_fw_cell = tf.nn.rnn_cell.GRUCell(num_units=self.enc_units//2,
                                                  name='lstm_fw')
            lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=0.5)
            lstm_bw_cell = tf.nn.rnn_cell.GRUCell(num_units=self.enc_units//2,
                                                  name='lstm_bw')
            lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=0.5)

            outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, embed,
                                                              sequence_length=input_length, dtype=tf.float32)
            outputs_concat = tf.concat(outputs, axis=-1)
            states = tf.concat(states, axis=-1)

        return outputs_concat, states

    def build_decoder(self, encoder_outputs, encoder_states, input_lengths):
        with tf.variable_scope('decode'):
            # <attention>
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.dec_units,
                                                                       memory=encoder_outputs,
                                                                       memory_sequence_length=input_lengths)
            decoder_cell = tf.contrib.rnn.GRUCell(num_units=self.dec_units)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                               attention_layer_size=self.dec_units)

            decoder_initial_state = decoder_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=encoder_states)

            # <helper>
            embeddings = lambda x: self.embeddings(x, self.emb_scope, reuse=True)
            pred_helper = CustomGreedyEmbeddingHelper(embedding=embeddings,
                                                      start_tokens=self.start_tokens,
                                                      end_token=self._tokenID_end)

            # <projection layer>
            projection_layer = tf.layers.Dense(self.vocab_size, use_bias=False)
            # <dynamic decoding>
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, pred_helper, decoder_initial_state,
                                                      output_layer=projection_layer)
            outputs = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                        swap_memory=True,
                                                        maximum_iterations=self.output_max_length)
        print("done!")
        return outputs



if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    model = Seq2SeqGenerator(vocab_size=100,
                             embedding_dim=64,
                             enc_units=256,
                             dec_units=256,
                             batch_size=32)
    outputs = model.build_model()
    # output specification
    # logit (batch, time step, vocab size)
    outputs[0].rnn_output
    # <tf.Tensor 'decode/decoder/transpose:0' shape=(32, ?, 100) dtype=float32>

    # token id
    outputs[0].sample_id

    # states & etc...
    outputs[1]
    outputs[2]
