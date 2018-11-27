"""
input: short text (integer sequence)
model: input -> embedding -> seq2seq(RNN+attention) -> output
output: long text (integer sequence)
"""
#100 -> 800
import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
import os

FLAGS = tf.app.flags.FLAGS

class Seq2SeqReconstructor:
    def __init__(vocab_size, embedding_dim, enc_units, dec_units, batch_size):
        self.vocab_size = vocab_size # 50000
        self.embedding_dim = embedding_dim # 128
        self.hidden_dim = hidden_dim # 64
        self.enc_units = enc_units #max enc number 100
        self.dec_units = dec_units #max dec number 800
        self.batch_size = batch_size #16

    def _add_placeholders(self):
        #encoder
        self._enc_batch = tf.placeholder(tf.int32,[batch_size,enc_units],name='enc_batch')
        self._enc_lens = tf.placeholder(tf.int32,[batch_size],name='enc_lens') #16 (number of samples )
        self._enc_padding_mask = tf.placeholder(tf.float32,[batch_size,None],name='enc_padding_mask')
        #decoder
        self._dec_batch = tf.placeholder(tf.int32,[batch_size,dec_units],name='dec_batch')
        self._target_batch=tf.placeholder(tf.int32,[batch_size,dec_units], name = 'target_batch')
        self._dec_padding_mask =  tf.placeholder(tf.float32,[batch_size,dec_units],name ='dec_padding_mask')

    def _add_encoder(self, encoder_inputs,seq_len):
        #input
        #encoder_inputs: tensor shape[batch_size, <= max_enc_steps,emb_size]
        #seq_len: lengths of encoder_input, tensor shape[batch_size]
        #output
        #encoder outputs: tensor shape[batch_size, <=max_enc_steps, 2*hidden_dim]
        #LSTMState Tuple shape ([batch_size, hidden_dim],[batch_size,hidden_dim])
        with tf.variable_scope('encoder',reuse=tf.AUTO_REUSE):
            cell_fw = tf.contrib.rnn.LSTMCell(hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(hidden_dim,initializer=self.rand_unif_init, state_is_tuple=True)
            (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs, dtype=tf.float32,sequence_length=seq_len)#16
            encoder_outputs = tf.concat(axis=2, values=encoder_outputs)  # concatenate the forwards and backwards states
        return encoder_outputs, fw_st, bw_st

    def _reduce_states(fw_st, bw_st):
        #add linear layer : reduce the encoder's final FW and BW state into a single initial state for the decoder
        #encoder is bidirectional but the decoder is not
        #fw_st,bw_st: 2 LSTMStateTuple[batch_size, hidden_dim]
        #return LSTMStateTuple[batch_size, hidden_dim]
        hidden_dim = self.hidden_dim
        with tf.variable_scope('reduce_final_st',reuse=tf.AUTO_REUSE):
            # Define weights and biases to reduce the cell and reduce the state
            w_reduce_c = tf.get_variable('w_reduce_c', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
            w_reduce_h = tf.get_variable('w_reduce_h', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
            bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim], dtype=tf.float32, initializer=trunc_norm_init)
            bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim], dtype=tf.float32, initializer=trunc_norm_init)

            # Apply linear layer
            old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c])  # Concatenation of fw and bw cell
            old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h])  # Concatenation of fw and bw state
            new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c)  # Get new cell from old cell
            new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h)  # Get new state from old state
        return tf.contrib.rnn.LSTMStateTuple(new_c, new_h)  # Return new cell and state

    def attention_decoder(decoder_inputs, initial_state, encoder_states, enc_padding_mask, cell, initial_state_attention=False):
        #pass each decoder input through linear layer with previous step's context vector
        #decoder_input: 2D list of tensor shaped as [batch_size, input_size]
        #initial state : 2D [batch_size,cell_state_size] (16,64)
        #encoder state: 3D [batch_size, attention_len, attention size]
        #enc_padding_mask: 2D tensor [batch_size, attention_len]
        #cell: rnn_cell.RNNCell
        #initial_state_attention: for encode =False(call for all decoder steps), for decode=True(call for each decoder steps)

        #return output: 2D list of same length as decoder_inputs[batch_size, cell_state_size]
        #state: final state of decoder
        #attention dists: [batchsize, atten_len] : attention distributions for each decoder step

        with variable_scope.variable_scope("attention_decoder",reuse=tf.AUTO_REUSE) as scope:
            batch_size = encoder_states.get_shape()[0].value# 16
            attn_size = encoder_states.get_shape()[2].value#128

            #reshape encoder_states(insert dimension)
            encoder_states = tf.expand_dims(encoder_states,axis=2) # [batch_size, attn_len,1,attn_size]
            # attention formulation:  v^T tanh(W_h h_i + W_s s_t + b_attn) # all of their vetor size: 128
            attention_vec_size = attn_size #128
            # apply W_h to every encoder state
            W_h = variable_scope.get_variable('W_h',[1,1,attn_size, attention_vec_size]) #[batch_size, attn_len,1, attention_vector_size]
            encoder_features = nn_ops.conv2d(encoder_states, W_h, [1,1,1,1],'SAME') #[batch_size, atten_len, 1, attention_vector_size]
            v = variable_scope.get_variable('v', [attention_vec_size])

        def attention(decoder_state):
            #calculate context vector and attention distribution per each decoder state
            #return context vector(weighted sum of encoder state), atten_dist
            with variable_scope.variable_scope("Attention"):
                #pass the decoder state through  a linear layer
                #W_s s_t + b_attn
                decoder_features = linear(decoder_state, attention_vec_size,True) # [batch_size, attention_vec_size]
                decoder_features = tf.expand_dims(tf.expand_dims(decoder_features,1),1) #reshape to [batch_size,1,1,attention_vec_size]

                def masked_attention(e):
                    """Take softmax of e then apply enc_padding_mask and re-normalize"""
                    attn_dist = nn_ops.softmax(e)  # take softmax. shape (batch_size, attn_length)
                    attn_dist *= enc_padding_mask  # apply mask
                    masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
                    return attn_dist / tf.reshape(masked_sums, [-1, 1])  # re-normalize

                # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
                e = math_ops.reduce_sum(v * math_ops.tanh(encoder_features + decoder_features),[2, 3])  # calculate e
                attn_dist = masked_attention(e)

                # Calculate the context vector from attn_dist and encoder_states
                context_vector = math_ops.reduce_sum(array_ops.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_states, [1, 2])  # shape (batch_size, attn_size).
                context_vector = array_ops.reshape(context_vector, [-1, attn_size])

            return context_vector, attn_dist

        outputs = []
        attn_dists = []
        state = initial_state #16,64
        context_vector = array_ops.zeros([batch_size, attn_size]) #[16,128]
        context_vector.set_shape([None, attn_size])  # Ensure the second shape of attention vectors is set.#[16,128]

        if initial_state_attention:  # true in decode mode
            # Re-calculate the context vector from the previous step so that we can pass it through a linear layer with this step's input to get a modified version of the input
            context_vector, _ = attention(initial_state)  # in decode mode, this is what updates the coverage vector
            for i, inp in enumerate(decoder_inputs):
                print(i,inp)
                tf.logging.info("Adding attention_decoder timestep %i of %i", i, len(decoder_inputs))
                if i > 0:
                    variable_scope.get_variable_scope().reuse_variables()
                # Merge input and previous attentions into one vector x of the same size as inp
                input_size = inp.get_shape().with_rank(2)[1]
                x = linear([inp] + [context_vector], input_size, True)
                # Run the decoder RNN cell. cell_output = decoder state
                cell_output, state = cell(x, state)
            # Run the attention mechanism.
                if i == 0 and initial_state_attention:  # always true in decode mode
                    with variable_scope.variable_scope(variable_scope.get_variable_scope(),reuse=tf.AUTO_REUSE):  # you need this because you've already run the initial attention(...) call
                        context_vector, attn_dist = attention(state)  # don't allow coverage to update
                    with variable_scope.variable_scope('AttnOutputprojection'):
                        output = linear([cell_output] + [context_vector], cell.output_size, True)
                    outputs.append(output)
        return outputs, state, attn_dists


    def _add_decoder(self,inputs):
        #Add attention decoder to the graph. In train or eval mode, you call this once to get output on ALL steps.
        #In decode (beam search) mode, you call this once for EACH decoder step.
        #input: word embeddings [batch_size, emb_dim]
        #outputs: list of tensor(outputs of decoder)
        #out_state: decoder final state
        #attn_dists: list of tensor(attention distribution)

        cell = tf.contrib.rnn.LSTMCell(hidden_dim,state_is_tuple=True,initializer=self.rand_unif_init)
        outputs, out_state, attn_dists = attention_decoder(inputs, self._dec_in_state, self._enc_states, self._enc_padding_mask , cell, initial_state_attention=False)
        return outputs, out_state, attn_dists

    def _add_seq2seq(self):
        vocab_size = self.vocab_size
        ids = self._topk_ids
        probs = self._topk_log_probs

        with tf.variable_scope('seq2seq',reuse=tf.AUTO_REUSE):
            self.rand_unif_init = tf.random_uniform_initializer(-0.02, 0.02,seed=123)
            self.trunc_norm_init = tf.truncated_normal_initializer(stddev=1e-4)

        #embedding matrix (encoder and decoder)
        with tf.variable_scope('embedding',reuse = tf.AUTO_REUSE):
            embedding = tf.get_variable('embedding',[vocab_size, emb_dim],dtype=tf.float32, initializer = self.trunc_norm_init)
            emb_enc_inputs = tf.nn.embedding_lookup(embedding,self._enc_batch) # shape[batch_size, max_enc_steps, emb_size]
            emb_dec_inputs = [tf.nn.embedding_lookup(embedding,x) for x in tf.unstack(self._dec_batch,axis=1)] # list of 800 seq in shape [batch_size, emb_size]

        #encoder
        enc_outputs,fw_st, bw_st = self._add_encoder(emb_enc_inputs,self._enc_lens)
        self._enc_states = enc_outputs
        #reduce the final encoder hidden state to fit the decoder hidden state(bidirectional-> directional)
        self._dec_in_state = self._reduce_states(fw_st,bw_st)

        #decoder
        with tf.variable_scope('decoder',reuse=tf.AUTO_REUSE):
            decoder_outputs, self._dec_out_state,self.attn_dists =self._add_decoder(emb_dec_inputs)

        # vocabulary distribution
        with tf.variable_scope('output_projection',reuse=tf.AUTO_REUSE):
            w = tf.get_variable('w', [hidden_dim, vocab_size], dtype=tf.float32, initializer=self.trunc_norm_init)
            w_t = tf.transpose(w)
            v = tf.get_variable('v', [vocab_size], dtype=tf.float32, initializer=self.trunc_norm_init)
            vocab_scores = []
            for i, output in enumerate(decoder_outputs):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                vocab_scores.append(tf.nn.xw_plus_b(output, w, v))  # apply the linear layer
            vocab_dists = [tf.nn.softmax(s) for s in vocab_scores]

            # final_dists list in shape (batch_size, extended_vsize)
            vocab_dists = vocab_dists[0]
            probs, ids = tf.nn.top_k(vocab_dists, batch_size * 2)  # take the k largest probs.
            self._topk_log_probs = tf.log(topk_probs)


def linear(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".

    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(axis=1, values=args), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [output_size], initializer=tf.constant_initializer(bias_start))
    return res + bias_term
