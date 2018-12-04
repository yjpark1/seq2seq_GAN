import tensorflow as tf
import os
import numpy as np

class Seq2SeqReconstructor:

    def __init__(self, vocab_size, embedding_dim, enc_units, dec_units, batch_size):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.enc_units = enc_units
        self.dec_units = dec_units
        self.batch_size = batch_size
        self.output_max_length = 800

    def build_model(self):
        with tf.get_default_graph().as_default():
        #with tf.Graph().as_default():
            train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size,self.enc_units])
            one_hot_train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size,100,self.vocab_size])
            one_hot_train_inputs = tf.cast(one_hot_train_inputs, dtype=tf.float32, name='dummy')
            input_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(train_inputs, 1)), -1, name='input_length')
            start_tokens = tf.zeros([self.batch_size], dtype=tf.int64)

            with tf.device('/cpu:0'):

                dense_emb = tf.layers.dense(one_hot_train_inputs, self.embedding_dim, use_bias=False, name='one_hot_embedding/dense_layer2',reuse=tf.AUTO_REUSE)

            encoder_outputs, encoder_states = self.build_encoder(dense_emb,input_lengths)
            decoder_outputs = self.build_decoder(encoder_outputs, encoder_states,
                                                 input_lengths=input_lengths,
                                                 start_tokens=start_tokens, embed_input = dense_emb)
            return decoder_outputs

    def build_encoder(self, embedding_value, input_len):
        with tf.variable_scope('encode',reuse=tf.AUTO_REUSE):
            lstm_fw_cell = tf.nn.rnn_cell.GRUCell(num_units=self.enc_units//2, name='lstm_fw')
            lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=0.5)
            lstm_bw_cell = tf.nn.rnn_cell.GRUCell(num_units=self.enc_units//2, name='lstm_bw')
            lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=0.5)

            output, state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, embedding_value,
                                                            sequence_length=input_len, dtype=tf.float32)
            outputs_concat = tf.concat(output,axis=-1)
            state = tf.concat(state, axis=-1)

            return outputs_concat, state

    def build_decoder(self, encoder_outputs, encoder_states, input_lengths, start_tokens,embed_input):
        with tf.variable_scope('decode',reuse=tf.AUTO_REUSE):
            # <attention>
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.dec_units,
                                                                       memory=encoder_outputs,
                                                                       memory_sequence_length=input_lengths)
            decoder_cell = tf.contrib.rnn.GRUCell(num_units=self.dec_units)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                               attention_layer_size=self.enc_units)
            out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                decoder_cell, self.vocab_size, reuse=tf.AUTO_REUSE)

            #decoder_initial_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_states)
            # <helper>
             #train_helper = tf.contrib.seq2seq.TrainingHelper(dense_emb, output_max_length)
            weights = tf.get_default_graph().get_tensor_by_name(os.path.split(embed_input.name)[0] + '/kernel:0')
            pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(weights,
                                                                   start_tokens=tf.to_int32(start_tokens),
                                                                   end_token=1)
            decoder_initial_state = decoder_cell.zero_state(self.batch_size, tf.float32).clone(
                cell_state=encoder_states)
            # <projection layer>
            projection_layer = tf.layers.Dense(self.vocab_size, use_bias=False)

            # <dynamic decoding>
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, pred_helper, decoder_initial_state,
                                                      output_layer=projection_layer)
            decode_outputs = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                        maximum_iterations=self.output_max_length)
            print("done!")
            return decode_outputs


if __name__ == '__main__':
    graph = tf.get_default_graph()
    model = Seq2SeqReconstructor(vocab_size=50000,
                             embedding_dim=128,
                             enc_units=64,
                             dec_units=64,
                             batch_size=16)
    outputs = model.build_model()
    outputs[0].rnn_output  # logit
    outputs[0].sample_id  # token id
    outputs[1]
    outputs[2]
    
