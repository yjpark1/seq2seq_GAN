# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 10:19:48 2018

@author: HQ
"""

import math
import tensorflow as tf

## A CNN discriminator model for text classification

class Discriminator(object):
    
    def __init__(self, vocab_size, max_seq_len = 400, batch_size = 16,
                 lr = 0.001, embedding_dim = 64, num_class = 2, l2_reg_lambda = 1,
                 dropout_prob = .5):
        
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.lr = lr
        self.embedding_dim = embedding_dim
        self.num_class = 2 
        self.l2_reg_lambda = l2_reg_lambda
        self.dropout_keep_prob = tf.get_variable(name = 'dropout_prob', shape = [],
                                                 initializer=tf.constant_initializer(dropout_prob))
        
    
        self.filter_sizes = [1, 2, 5, 10, 15, 20]
        self.num_filters = [100, 200, 200, 100, 160, 160]

        self.input_x = tf.placeholder('int32', [None, self.max_seq_len], name = 'input_X')
        self.input_y = tf.placeholder('float32', [None, self.num_class], name = 'input_Y')

        self.build_model()


    def build_model(self):
        with tf.variable_scope('discriminator'):
            # -- embedding -- 
            with tf.variable_scope('word_embedding'):
                self.emb_W = tf.get_variable(name = 'W', 
                                    shape = [self.vocab_size, self.embedding_dim],
                                    initializer=tf.truncated_normal_initializer(stddev=6/math.sqrt(self.embedding_dim)))
                self.word_emb = tf.nn.embedding_lookup(params = self.emb_W, ids = self.input_x)
                self.word_emb_expand = tf.expand_dims(self.word_emb, axis=-1)
        
            pooled_output = []
            
            #filter_size = filter_sizes[0]
            #num_filter = num_filters[0]
            
            for filter_size, num_filter in zip(self.filter_sizes, self.num_filters):
                with tf.variable_scope('conv_maxpool_%s' % filter_size):
                    filter_shape = [filter_size, self.embedding_dim, 1, num_filter]
                    conv_W = tf.get_variable(name = 'conv_W', shape = filter_shape,
                                        initializer=tf.truncated_normal_initializer(stddev=.1))
                    conv_b = tf.get_variable(name = 'conv_b', shape=[num_filter],
                                        initializer=tf.constant_initializer(0.1))
                    conv = tf.nn.conv2d(input = self.word_emb_expand,
                                        filter = conv_W,
                                        strides=[1, 1, 1, 1],
                                        padding='VALID',
                                        name = 'conv')
                    conv_bias = tf.nn.bias_add(value = conv, bias = conv_b, name ='conv_bias')
                    h = tf.nn.relu(conv_bias, name = 'relu')
                    
                    pooled = tf.nn.max_pool(value = h,
                                            ksize=[1, self.max_seq_len - filter_size + 1, 1, 1],
                                            strides = [1, 1, 1, 1],
                                            padding = 'VALID',
                                            name = 'max_pooling')
                    pooled_output.append(pooled)
            
            total_num_filters = sum(self.num_filters)
            self.h_pool = tf.concat(values = pooled_output, axis = 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, total_num_filters])
            
            with tf.name_scope('highway'):
                self.h_highway = self._highway(input_ = self.h_pool_flat, size = self.h_pool_flat.get_shape()[1],
                                     num_layers=1, bias=0.)
                
            with tf.name_scope('dropout'):
                self.h_drop = tf.nn.dropout(x = self.h_highway, keep_prob=self.dropout_keep_prob)
                
            l2_loss = tf.constant(0.0)
            
            with tf.name_scope('output'):
                W = tf.get_variable(name = 'W', shape = [total_num_filters, self.num_class],
                                    initializer=tf.truncated_normal_initializer(stddev=.1))
                b = tf.get_variable(name = 'b', shape = [self.num_class],
                                    initializer=tf.constant_initializer(0.1))
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name ='scores')
                self.ypred_for_auc = tf.nn.softmax(self.scores)
                self.predictions = tf.argmax(self.scores, 1, name = 'predictions')
                
            with tf.name_scope('loss'):
                losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.scores, labels = self.input_y)
                self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss
            
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.params = [param for param in  tf.trainable_variables() if 'discriminator' in param.name]
        gradients = self.optimizer.compute_gradients(loss = self.loss, var_list = self.params, aggregation_method=2)
        self.train_op = self.optimizer.apply_gradients(gradients)
        
        
    def _highway(self, input_, size, num_layers = 1, bias = -2.0, scope = 'Highway'):
        with tf.variable_scope(scope):
            for idx in range(num_layers):
                g = tf.nn.relu(self._linear(input_, size, scope = 'highway_lin_%d' % idx))
                t = tf.sigmoid(self._linear(input_, size, scope = 'highway_gate_%d' % idx) + bias)
                output = t * g + (1. - t) * input_
                input_ = output
        return output
    
    def _linear(self, input_, output_size, scope = None):
        shape = input_.get_shape().as_list()
        input_size = shape[1]
        
        with tf.variable_scope(scope or 'SimpleLinear'):
            matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
            bias_term = tf.get_variable("Bias", [output_size], dtype = input_.dtype)
        
        return tf.matmul(input_, tf.transpose(matrix)) + bias_term            
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
            