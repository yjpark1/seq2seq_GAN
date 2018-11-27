"""
input: short text (integer sequence)
model: input -> embedding -> RNN -> output
output: real/fake logit
"""
## A RNN discriminator model for text classification
import math
import tensorflow as tf

class Discriminator():
    def __init__(self, seq_len = 100, num_classes = 2, vocab_size = 1000, batch_size = 32,
                 use_dropout = False, embedding_dim = 128, num_layer = 1, rnn_hidden_units = 64,
                 lr = .001, use_l2norm = False, l2_reg_lambda = .1):

        self.max_seq_len = seq_len
        self.num_classes = num_classes
        self.vocab_size = vocab_size,
        self.batch_size = batch_size,
        self.use_dropout = use_dropout
        self.embedding_dim = embedding_dim
        self.num_layer = num_layer
        self.rnn_hidden_units = rnn_hidden_units
        self.lr = lr
        self.use_l2norm = use_l2norm
        self.l2_reg_lambda = l2_reg_lambda

    def build_placeholder(self):
        # - initialize placeholder - #
        self.inputs = tf.placeholder(shape = [None, self.max_seq_len],
                                     dtype = tf.int32,
                                     name = 'inputs')
        self.inputs_length = tf.placeholder(shape = [None,],
                                            dtype = tf.int32,
                                            name = 'inputs_length')
        self.target = tf.placeholder(shape = [None, self.num_classes],
                                     dtype = tf.float32,
                                     name = 'target')

        self.dropout_keep_prob = tf.placeholder(dtype = tf.float32,
                                                name = 'dropout_keep_prob')
        self.l2_loss = tf.constant(0.0)

    def build_discriminator(self):
        with tf.variable_scope('discriminator'):

        # - Embedding - #
            with tf.name_scope('embedding'):
                sqrt3 = math.sqrt(3)
                self.embedding_matrix = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_dim],
                                                                      -sqrt3, sqrt3),name ='embedding')
                self.inputs_embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.inputs)

        # - discriminator / RNN layer - #     
            with tf.name_scope('RNNlayer'):
                cell = tf.nn.rnn_cell.LSTMCell(num_units = self.rnn_hidden_units)
                if self.use_dropout:
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell = cell,
                                                         output_keep_prob = self.dropout_keep_prob)

                ## TODO :: multiple layer    
                outputs, _ = tf.nn.dynamic_rnn(cell = cell,
                                       inputs = self.inputs_embedded,
                                       sequence_length= self.inputs_length,
                                       dtype = tf.float32)

                _batch_size = self.batch_size
                _max_length = int(outputs.get_shape()[1])
                _input_size = int(outputs.get_shape()[2])
                _index = tf.range(0, _batch_size) * _max_length + (self.inputs_length - 1)
                _flat = tf.reshape(outputs, [-1, _input_size])
                self.hidden_output = tf.gather(_flat, _index)

        # - discriminator / RNN output layer - #     
            with tf.name_scope('output'):
                W = tf.get_variable(name = 'weight', shape = [self.rnn_hidden_units, self.num_classes],
                                    initializer = tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(.1, shape = [self.num_classes]), name = 'bias')

                self.l2_loss += tf.nn.l2_loss(W)
                self.l2_loss += tf.nn.l2_loss(b)

                self.logits = tf.nn.xw_plus_b(self.hidden_output, W, b, name = 'logits')
                self.prediction = tf.argmax(self.logits, 1, name = 'prediction')

            # - calculate loss - #             
            with tf.name_scope('d_loss'):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.target)
                if self.use_l2norm:
                    self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss
                else:
                    self.loss = tf.reduce_mean(losses)

        self.params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
        d_optim = tf.train.AdamOptimizer(self.lr)
        grads_and_vars = d_optim.compute_gradients(self.loss, self.params, aggregation_method=2)
        self.train_op = d_optim.apply_gradients(grads_and_vars)

