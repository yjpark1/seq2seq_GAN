"""
input: long text (integer sequence)
model: input -> embedding -> seq2seq(RNN+attention) -> output
output: short text (integer sequence)

<reference>
1) https://gist.github.com/ilblackdragon/c92066d9d38b236a21d5a7b729a10f12
2) https://github.com/hccho2/RNN-Tutorial
"""
import tensorflow as tf
import tensorflow_probability as tfp
from train import hyperparameter as H


class Seq2SeqReconstructor:
    def __init__(self, emb, temp, vocab_size,
                 embedding_units, enc_units, dec_units):
        self.embeddings = emb
        self.vocab_size = vocab_size
        self.embedding_units = embedding_units
        self.enc_units = enc_units
        self.dec_units = dec_units
        self.temp = temp

        self.max_output_length = H.max_text_len
        self.namescope = 'reconstructor'

    def build_model(self, model_inputs, len_model_inputs,
                    target_inputs, len_target_inputs,
                    reuse=False):
        with tf.variable_scope('', reuse=True):
            embedding_kernel = tf.get_variable('embedding')
        embed_dense = lambda x: tf.tensordot(x, embedding_kernel, axes = 1)
        embed = embed_dense(model_inputs)
        target_inputs_emb = tf.nn.embedding_lookup(embedding_kernel, target_inputs)
        # embed = self.embeddings(model_inputs)
        # target_inputs_emb = self.embeddings(target_inputs)

        with tf.variable_scope(self.namescope, reuse=reuse):
            # seq2seq encoder & decoder
            encoder_outputs, encoder_states = self.build_encoder(embed, len_model_inputs)

            # <helper>
            helper_teacher = tf.contrib.seq2seq.TrainingHelper(target_inputs_emb, len_target_inputs)

            # teacher helper
            (logit, sample_id), _, _ = self.build_decoder(encoder_outputs, encoder_states,
                                                          input_lengths=len_model_inputs,
                                                          helper=helper_teacher)
            # gumbel trick
            dist = tfp.distributions.RelaxedOneHotCategorical(temperature=self.temp, logits=logit)
            generate_sequence = dist.sample()
            print("build reconstructor done!")

        return logit, generate_sequence

    def build_encoder(self, embed, input_length):
        with tf.variable_scope('encode'):
            lstm_fw_cell = tf.nn.rnn_cell.GRUCell(num_units=self.enc_units // 2,
                                                  name='lstm_fw')
            lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=0.5)
            lstm_bw_cell = tf.nn.rnn_cell.GRUCell(num_units=self.enc_units // 2,
                                                  name='lstm_bw')
            lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=0.5)

            outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, embed,
                                                              sequence_length=input_length, dtype=tf.float32)
            outputs_concat = tf.concat(outputs, axis=-1)
            states = tf.concat(states, axis=-1)

        return outputs_concat, states

    def build_decoder(self, encoder_outputs, encoder_states, input_lengths, helper):
        with tf.variable_scope('decode'):
            batch_size = tf.shape(input_lengths)[0]
            # <attention>
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.dec_units,
                                                                       memory=encoder_outputs,
                                                                       memory_sequence_length=input_lengths)
            decoder_cell = tf.contrib.rnn.GRUCell(num_units=self.dec_units)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                               attention_layer_size=self.dec_units)

            decoder_initial_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_states)

            # <projection layer>
            projection_layer = tf.layers.Dense(self.vocab_size, use_bias=False)

            # <dynamic decoding>
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_initial_state,
                                                      output_layer=projection_layer)
            outputs = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                        swap_memory=True,
                                                        maximum_iterations=self.max_output_length,
                                                        output_time_major=False,
                                                        impute_finished=True)
        return outputs

