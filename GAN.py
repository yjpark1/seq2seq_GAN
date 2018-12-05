
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 20:54:27 2018

@author: hankyu
"""
import math
import tensorflow as tf
import numpy as np
#import gumbel_softmax as gs

def get_scope_variables(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = scope)

class EGAN(object):
    """
    The GAN model based on the seqGAN 
    reference: https://github.com/sminocha/text-generation-GAN/blob/master/model.py.
    """
    
    def __init__(self, sess, vocab_size, log_every=50, num_latent=100, embedding_units = 64,
                 batch_size = 32, output_max_length = 100,hidden_unit = 128, lr = 0.001):
        
        self._batch_size = batch_size 
        self._num_latent = num_latent
        self._vocab_size = vocab_size
        self._learn_phase = None
        self._embedding_units = embedding_units
        self._hidden_unit = hidden_unit
        self._log_every = log_every
        self._output_max_length = output_max_length
        self._lr = lr
        self._sess = sess
        

        self.inputs = tf.placeholder(dtype=tf.int32, shape=(self._batch_size, None), name='inputs_text')
        self.inputs_length = tf.reduce_sum(tf.to_int32(tf.not_equal(self.inputs, 1)), -1)
        self.latent = tf.placeholder(dtype=tf.float32, shape=(self._batch_size, self._num_latent), name='inputs_latent')
        self.sample_pl = tf.placeholder(dtype='bool', shape=(), name='sample')
        self._time = tf.Variable(0, name = 'time')

    def _generate_latent_variable(self, batch_size):
        return np.random.normal(size=(batch_size, self._num_latent))

    def get_weight(self, name, shape, trainable=True):
        sqrt3 = math.sqrt(3)
        initializer = (lambda shape, dtype, partition_info:
            tf.random_normal(shape, -sqrt3, sqrt3))
        weight = tf.get_variable(name = name,
                                 shape = shape,
                                 initializer = initializer,
                                 trainable = trainable)
        return weight
    
#tf.reset_default_graph()
    def build_generator(self, reuse = False):

        with tf.variable_scope('generator', reuse = reuse):
            
            # <* embedding *>
            embeddings = tf.get_variable('embedding', [self._vocab_size, self._embedding_units])
            inputs_embdded = tf.nn.embedding_lookup(embeddings, self.inputs)
            # [32, ?, embedding_units]

            # <* encoder*>
            lstm_fw_cell = tf.nn.rnn_cell.GRUCell(num_units=self._hidden_unit,name='lstm_fw')
            lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=0.5)
            lstm_bw_cell = tf.nn.rnn_cell.GRUCell(num_units=self._hidden_unit,name='lstm_bw')
            lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=0.5)

            outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                              lstm_bw_cell,
                                                              inputs_embdded,
                                                              sequence_length=self.inputs_length,
                                                              dtype=tf.float32)
            
            outputs_concat = tf.concat(outputs, axis=-1)
            encoder_states = tf.concat(states, axis=-1)
            
            # Gets the batch size from the latent pl.
            #batch_size = tf.shape(self.latent)[0]
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=(self._hidden_unit*2),
                                                                       memory=outputs_concat,
                                                                       memory_sequence_length=self.inputs_length)
            decoder_cell = tf.contrib.rnn.GRUCell(num_units = self._hidden_unit*2)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                               attention_layer_size=self._hidden_unit*2)
            decoder_initial_state = decoder_cell.zero_state(self._batch_size, tf.float32).clone(cell_state=encoder_states)

            # <helper>
            pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                   start_tokens=tf.tile([0], [self._batch_size]),
                                                                   end_token=1)
            projection_layer = tf.layers.Dense(self._vocab_size, use_bias=False)
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, pred_helper, decoder_initial_state,
                                                      output_layer=projection_layer)
            
            (logits, sample_id), _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                        swap_memory=True,
                                                        maximum_iterations=self._output_max_length)

            scores = tf.nn.softmax(logits)
            generate_sequence = tf.argmax(logits, axis= 2)     
            
        return scores, generate_sequence 
    
    
    def build_discriminator(self, inputs_seq, reuse = False):
        
        with tf.variable_scope('discriminator', reuse = reuse):

             # - embedding layer - # 
            embeddings = tf.get_variable('embedding', [self._vocab_size, self._embedding_units])
            inputs_embdded = tf.nn.embedding_lookup(embeddings, inputs_seq)
            
            ###################################################################
            
            cell = tf.nn.rnn_cell.LSTMCell(num_units = self._hidden_unit)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell = cell,
                                                 output_keep_prob = .5)
            cell = tf.contrib.rnn.FusedRNNCellAdaptor(cell, True)

            ## TODO :: multiple layer    
            input_enc = tf.transpose(inputs_embdded, (1, 0, 2))
            rnn_output, _ = cell(input_enc, dtype = tf.float32)
            rnn_output = tf.transpose(rnn_output, (1, 0, 2))

            # Reduces to binary prediction.
            pred_W = self.get_weight('pred_W', (self._hidden_unit, 1))
            preds = tf.einsum('ijk,kl->ijl', rnn_output, pred_W)

            logits = tf.squeeze(preds, [2])
            preds = tf.sigmoid(logits)   
            
        return logits, preds
    
    def discriminator_op(self, r_logits, f_logits, d_weights):
        with tf.variable_scope('loss/discriminator'):
            
            dis_optim = tf.train.AdamOptimizer(learning_rate = self._lr)
            
            r_loss = -tf.reduce_mean(r_logits)
            f_loss = tf.reduce_mean(f_logits)
            d_loss = r_loss + f_loss
            
            tf.summary.scalar('d_loss', d_loss)
            d_optim = dis_optim.minimize(d_loss, var_list = d_weights)
            
        return d_optim
        
    def generator_op(self, g_seq, f_logits, g_scores, g_weights):
        
        with tf.variable_scope('loss/generator'):
            gen_optim = tf.train.AdamOptimizer(learning_rate = self._lr)
            reward_op = tf.train.GradientDescentOptimizer(1e-3)
            
            g_seq = tf.one_hot(g_seq, self._vocab_size)
            g_scores = tf.clip_by_value(g_scores * g_seq, 1e-20, 1)
            
            expected_reward = tf.Variable(tf.zeros((self._output_max_length,)))
            reward = f_logits - expected_reward[:tf.shape(f_logits)[1]]
            mean_reward = tf.reduce_mean(reward)
            
            exp_reward_loss = tf.reduce_mean(tf.abs(reward))
            exp_op = reward_op.minimize(
                    exp_reward_loss, var_list = [expected_reward])
            
            reward = tf.expand_dims(tf.cumsum(reward, axis=1, reverse = True), -1)
            gen_reward = tf.log(g_scores) * reward
            gen_reward = tf.reduce_mean(gen_reward)
            
            gen_loss = -gen_reward
            
            gen_op = gen_optim.minimize(gen_loss, var_list = g_weights)
            
            g_optim = tf.group(gen_op, exp_op)
            
        tf.summary.scalar('loss/expected_reward', exp_reward_loss)
        tf.summary.scalar('reward/mean', mean_reward)
        tf.summary.scalar('reward/generator', gen_reward)
        
        return g_optim

   
    def build_model(self):
        
        #tf.reset_default_graph()
        g_scores, g_seq  = self.build_generator(reuse = False)
        
        r_logits, r_preds = self.build_discriminator(self.inputs, reuse = False)
        f_logits, f_preds = self.build_discriminator(g_seq, reuse=True)
        
        d_weights = get_scope_variables('discriminator')
        g_weights = get_scope_variables('generator')
        
        self.r_logits, self.f_logits = r_logits, f_logits
        self.generated_sequence = g_seq
        
        dis_op = self.discriminator_op(r_logits, f_logits, d_weights)
        gen_op = self.generator_op(g_seq, f_logits, g_scores, g_weights)
        
        step_op = self._time.assign(self._time+1)
        
        gan_train_op = tf.group(gen_op, dis_op)

        self.train_op = tf.group(gan_train_op, step_op)
        self.summary_op = tf.summary.merge_all()
        self._saver = tf.train.Saver()
        self._sess.run(tf.global_variables_initializer())
   
        
'''         
        self.summary_writer = tf.summary.FileWriter(
                self.logdir, self._sess.graph)
        

    def train_batch(self, batch):
        
        batch_size, seq_len = batch.shape
        noise = self._generate_latent_variable(batch_size)
        
        feed_dict = {
                self.inputs = batch,
                self.inputs_lenght = seq_len,
                self.latent = noise,
                self.sample_pl = False,
                }
        
        train = self._sess.run(self._time)
        if train % self.log_every:
            _sess.run(train_op, feed_dict = feed_dict)
        else:
            _, summary = self._sess.run([train_op, summary_op],
                                   feed_dict = feed_dict)
            self.summary_writer.add_summary(summary, train)
            
    def generate(self, sample_length):
        
        noise = self._generate_latent_variable(1)
        
        sequence, = self._sess.run([self.generated_sequence],
                              feed_dict = {self.latent: noise,
                                           self.inputs_length: sample_length,
                                           self.sample_pl: True})
        return sequence[0]
'''



if __name__ == '__main__':
   
    import math
    import tensorflow as tf

    sess = tf.Session()
    model = EGAN(sess, vocab_size= 1000,
                 log_every = 50,
                 num_latent = 100,
                 embedding_units = 64,
                 hidden_unit = 128,
                 batch_size = 32,
                 output_max_length = 100)
    
    model.build_model()

