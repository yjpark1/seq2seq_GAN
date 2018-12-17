"""
https://github.com/sminocha/text-generation-GAN/blob/master/model.py
"""
import tensorflow as tf
from models.utils import dynamic_time_pad
from train import hyperparameter as H
from train.utils import Generator


def get_scope_variables(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)


class SeqGAN:
    def __init__(self, sess, discriminator, generator, reconstructor,
                 learning_rate=0.001):
        self.sess = sess
        self.D = discriminator
        self.G = generator
        self.R = reconstructor

        # hyperparameters
        self.lr = learning_rate

        # training parameter        
        #self.batch_size = H.batch_size
        self.vocab_size = H.vocab_size
        self.max_output_length = H.max_summary_len

        # manage scope
        # self.emb_scope = emb_scope

    def build_iterator(self, txtwithsumm_summ,
                       txtwithsumm_txt, txtwithoutsumm_txt):
        gen_lbl_summary = Generator(txtwithsumm_summ, max_len=H.max_summary_len)
        gen_lbl_text = Generator(txtwithsumm_txt, max_len=H.max_text_len)
        gen_ulbl_text = Generator(txtwithoutsumm_txt, max_len=H.max_text_len)

        lbl_summary = tf.data.Dataset().from_generator(gen_lbl_summary.gen_data,
                                                       output_types=tf.int64,
                                                       output_shapes=(
                                                           tf.TensorShape([H.max_summary_len + 1])))

        lbl_text = tf.data.Dataset().from_generator(gen_lbl_text.gen_data,
                                                    output_types=tf.int64,
                                                    output_shapes=(tf.TensorShape([H.max_text_len + 1])))

        ulbl_text = tf.data.Dataset().from_generator(gen_ulbl_text.gen_data,
                                                     output_types=tf.int64,
                                                     output_shapes=(tf.TensorShape([H.max_text_len + 1])))

        len_lbl_summary = tf.data.Dataset().from_generator(gen_lbl_summary.gen_len,
                                                           output_types=tf.int32,
                                                           output_shapes=(tf.TensorShape([])))

        len_lbl_text = tf.data.Dataset().from_generator(gen_lbl_text.gen_len,
                                                        output_types=tf.int32,
                                                        output_shapes=(tf.TensorShape([])))

        len_ulbl_text = tf.data.Dataset().from_generator(gen_ulbl_text.gen_len,
                                                         output_types=tf.int32,
                                                         output_shapes=(tf.TensorShape([])))

        dcomb = tf.data.Dataset.zip({'real_summary': lbl_summary.repeat(),
                                     'real_summary_len': len_lbl_summary.repeat(),
                                     'labeled_text': lbl_text.repeat(),
                                     'labeled_text_len': len_lbl_text.repeat(),
                                     'unlabeled_text': ulbl_text.repeat(),
                                     'unlabeled_text_len': len_ulbl_text.repeat()})

        dcomb = dcomb.batch(H.batch_size)
        iterator = dcomb.make_initializable_iterator()
        return iterator

    def make_decoder_teacher(self, x):
        """
        inputs: [<start>, X_1, x_2, ... X_n-1, <end>, <end>..., , <end>]
        decoder_input: [<start>, X_1, x_2, ... X_n-1, <end>] => length n
        decoder_output: [X_1, x_2, ... X_n-1, <end>, <end>] => length n
        """
        decoder_input = x[:, :-1]
        decoder_output = x[:, 1:]
        return decoder_input, decoder_output

    def build_gan(self, inputs):
        # placeholder: labeled text
        self.labeled_text = inputs['labeled_text']
        self.labeled_text_lengths = inputs['labeled_text_len']
        batch_size = tf.shape(self.labeled_text)[0]
        weight_l_txt = tf.sequence_mask(self.labeled_text_lengths, H.max_text_len, dtype=tf.float32)

        # placeholder: unlabeled text
        self.unlabeled_text = inputs['unlabeled_text']
        self.unlabeled_text_lengths = inputs['unlabeled_text_len']
        weight_u_txt = tf.sequence_mask(self.unlabeled_text_lengths, H.max_text_len, dtype=tf.float32)

        # placeholder: summary
        self.real_summary = inputs['real_summary']
        self.real_summary_length = inputs['real_summary_len']
        weight_s_txt = tf.sequence_mask(self.real_summary_length, H.max_summary_len, dtype=tf.float32)

        # split decoder teacher
        real_summary_di, real_summary_do = self.make_decoder_teacher(self.real_summary)
        labeled_text_di, labeled_text_do = self.make_decoder_teacher(self.labeled_text)
        unlabeled_text_di, unlabeled_text_do = self.make_decoder_teacher(self.unlabeled_text)
        real_summary_in = tf.one_hot(self.real_summary, self.vocab_size)

        # build generator
        g_real_logits, g_real_seq, g_real_len = self.G.build_model(self.labeled_text,
                                                                   self.labeled_text_lengths,
                                                                   target_inputs=real_summary_di,
                                                                   len_target_inputs=self.real_summary_length,
                                                                   reuse=False)
        g_fake_logits, g_fake_seq, g_fake_len = self.G.build_model(self.unlabeled_text,
                                                                   self.unlabeled_text_lengths,
                                                                   reuse=True)
        
        self.generated_sequence = g_fake_seq
        
        # build discriminator
        d_real_logits, d_real_preds = self.D.build_model(real_summary_in, reuse=False)
        d_fake_logits, d_fake_preds = self.D.build_model(g_fake_seq, reuse=True)

        # build reconstructor
        r_real_logits, r_real_seq = self.R.build_model(g_real_seq, g_real_len,
                                                       target_inputs=labeled_text_di,
                                                       len_target_inputs=self.labeled_text_lengths,
                                                       reuse=False)
        r_target_logits, r_target_seq = self.R.build_model(real_summary_in, self.real_summary_length,
                                                           target_inputs=labeled_text_di,
                                                           len_target_inputs=self.labeled_text_lengths,
                                                           reuse=True)
        r_fake_logits, r_fake_seq = self.R.build_model(g_fake_seq, g_fake_len,
                                                       target_inputs=unlabeled_text_di,
                                                       len_target_inputs=self.unlabeled_text_lengths,
                                                       reuse=True)

        # get trainable parameters
        self.d_weights = get_scope_variables('discriminator') + get_scope_variables('embedding')
        self.r_weights = get_scope_variables('reconstructor') + get_scope_variables('embedding')
        self.g_weights = get_scope_variables('generator') + get_scope_variables('embedding')

        # loss: discriminator
        d_r_loss = -tf.reduce_mean(0.9 * tf.log(tf.clip_by_value(d_real_preds, 1e-7, 1. - 1e-7)))  # r_preds -> 1.
        d_f_loss = -tf.reduce_mean(tf.log(1 - tf.clip_by_value(d_fake_preds, 1e-7, 1. - 1e-7)))  # g_preds -> 0.
        dis_loss = d_r_loss + d_f_loss

        # loss: reconstructor
        # labeled_text_target = tf.math.argmax(self.labeled_text, axis=-1)
        # unlabeled_text_target = tf.math.argmax(self.unlabeled_text, axis=-1)

        # pad output
        r_real_logits = dynamic_time_pad(r_real_logits, H.max_text_len, batch_size)
        r_fake_logits = dynamic_time_pad(r_fake_logits, H.max_text_len, batch_size)
        r_target_logits = dynamic_time_pad(r_target_logits, H.max_text_len, batch_size)

        # clippling to prevent NaN
        r_real_logits = r_real_logits + 1e-7
        r_fake_logits = r_fake_logits + 1e-7
        r_target_logits = r_target_logits + 1e-7

        r_r_loss = tf.contrib.seq2seq.sequence_loss(r_real_logits, labeled_text_do, weight_l_txt)
        r_f_loss = tf.contrib.seq2seq.sequence_loss(r_fake_logits, unlabeled_text_do, weight_u_txt)
        r_t_loss = tf.contrib.seq2seq.sequence_loss(r_target_logits, labeled_text_do, weight_l_txt)

        r_r_loss = tf.clip_by_value(r_r_loss, 0, 20)
        self.r_f_loss = tf.clip_by_value(r_f_loss, 0, 20)
        self.r_t_loss = tf.clip_by_value(r_t_loss, 0, 20)

        rec_loss = r_r_loss + self.r_f_loss + self.r_t_loss

        # loss: generator
        # from target summary
        g_real_logits = dynamic_time_pad(g_real_logits, H.max_summary_len, batch_size)
        g_r_loss = tf.contrib.seq2seq.sequence_loss(g_real_logits, real_summary_do, weight_s_txt)
        self.g_r_loss = tf.clip_by_value(g_r_loss, 0, 20)
        # from discriminator
        g_d_loss = -tf.reduce_mean(tf.log(tf.clip_by_value(d_fake_preds, 1e-7, 1. - 1e-7)))  # g_preds -> 0.
        gen_loss = self.g_r_loss + r_f_loss + g_d_loss
        
        self.dis_loss = dis_loss
        self.gen_loss = gen_loss
        self.rec_loss = rec_loss
                
        # optimization operator
        self.dis_op = self.train_operator(loss_scope='loss/discriminator', loss=dis_loss, weights=self.d_weights)
        self.rec_op = self.train_operator(loss_scope='loss/reconstructor', loss=rec_loss, weights=self.r_weights)
        self.gen_op = self.train_operator(loss_scope='loss/generator', loss=gen_loss, weights=self.g_weights)

        # pretraining operator
        self.pretrain_gen = self.train_operator(loss_scope='loss/pretrain_generator',
                                                loss=self.g_r_loss, weights=self.g_weights)
        self.pretrain_recon = self.train_operator(loss_scope='loss/pretrain_reconstructor',
                                                  loss=self.rec_loss, weights=self.r_weights)
        self.pretrain_auto = self.train_operator(loss_scope='loss/pretrain_autoencoder',
                                                 loss=self.g_r_loss + self.rec_loss,
                                                 weights=self.r_weights + self.g_weights)

        # define train operator
        self.train_op = tf.group(self.gen_op, self.dis_op, self.rec_op)
        self.saver = tf.train.Saver()
        print('build GAN model done!')
        return (d_r_loss, d_f_loss), (g_r_loss, g_d_loss), (r_r_loss, r_f_loss, r_t_loss)

    def train_operator(self, loss_scope, loss, weights):
        with tf.variable_scope(loss_scope):
            # optimizer
            optim = tf.train.AdamOptimizer(learning_rate=self.lr)

            gvs = optim.compute_gradients(loss, weights)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs if grad is not None]
            op_train = optim.apply_gradients(capped_gvs)
        return op_train
