"""
https://github.com/sminocha/text-generation-GAN/blob/master/model.py
"""
import tensorflow as tf
from models.utils import dynamic_time_pad
from models.discriminator import RNNDiscriminator
from models.generator import Seq2SeqGenerator
from train import hyperparameter as H


def get_scope_variables(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)


class SeqGAN:
    def __init__(self, sess, discriminator, generator, reconstructor, emb_scope,
                 learning_rate=0.001):
        self.sess = sess
        self.D = discriminator
        self.G = generator
        self.R = reconstructor

        # hyperparameters
        self.lr = learning_rate
        self.learn_phase = None

        # training parameter        
        self.batch_size = H.batch_size
        self.vocab_size = H.vocab_size
        self.max_output_length = H.max_summary_len
        self._time = tf.Variable(0, name='time')

        # manage scope
        self.emb_scope = emb_scope

    def build_gan(self):
        # placeholder: labeled text
        labeled_text = tf.placeholder(tf.float32, name='labeled_text',
                                      shape=[self.batch_size, H.max_text_len, self.vocab_size])
        labeled_text_lengths = tf.placeholder(tf.int32, name='labeled_text_len', shape=[self.batch_size, ])
        weight_l_txt = tf.placeholder(tf.float32, name='labeled_text_wgt', shape=[self.batch_size, H.max_text_len])

        # placeholder: unlabeled text
        unlabeled_text = tf.placeholder(tf.float32, name='unlabeled_text',
                                        shape=[self.batch_size, H.max_text_len, self.vocab_size])
        unlabeled_text_lengths = tf.placeholder(tf.int32, name='unlabeled_text_len', shape=[self.batch_size, ])
        weight_u_txt = tf.placeholder(tf.float32, name='unlabeled_text_wgt', shape=[self.batch_size, H.max_text_len])

        # placeholder: summary
        real_summary = tf.placeholder(tf.float32, name='real_summary',
                                      shape=[self.batch_size, H.max_summary_len, self.vocab_size])

        # build generator
        g_real_probs, g_real_seq = self.G.build_model(labeled_text, labeled_text_lengths,
                                                       reuse=False, emb_reuse=False)
        g_fake_probs, g_fake_seq = self.G.build_model(unlabeled_text, unlabeled_text_lengths,
                                                       reuse=True, emb_reuse=True)

        # build discriminator
        d_real_logits, d_real_preds = self.D.build_model(real_summary, reuse=False)
        d_fake_logits, d_fake_preds = self.D.build_model(g_fake_probs, reuse=True)

        # build reconstructor
        r_real_probs, r_real_seq = self.R.build_model(g_real_probs, labeled_text_lengths,
                                                       reuse=False, emb_reuse=True)
        r_target_probs, r_target_seq = self.R.build_model(real_summary, labeled_text_lengths,
                                                           reuse=True, emb_reuse=True)
        r_fake_logits, r_fake_seq = self.R.build_model(g_fake_logits, unlabeled_text_lengths,
                                                       reuse=True, emb_reuse=True)

        # get trainable parameters
        d_weights = get_scope_variables('discriminator') + get_scope_variables('embedding')
        r_weights = get_scope_variables('reconstructor') + get_scope_variables('embedding')
        g_weights = get_scope_variables('generator') + get_scope_variables('embedding')

        # loss: discriminator
        d_r_loss = -tf.reduce_mean(tf.log(d_real_preds))  # r_preds -> 1.
        d_f_loss = -tf.reduce_mean(tf.log(1 - d_fake_preds))  # g_preds -> 0.
        dis_loss = d_r_loss + d_f_loss

        # loss: reconstructor
        labeled_text_target = tf.math.argmax(labeled_text, axis=-1)
        unlabeled_text_target = tf.math.argmax(unlabeled_text, axis=-1)

        # pad output
        r_real_logits = dynamic_time_pad(r_real_logits, H.max_text_len)
        r_fake_logits = dynamic_time_pad(r_fake_logits, H.max_text_len)
        r_target_logits = dynamic_time_pad(r_target_logits, H.max_text_len)

        r_r_loss = tf.contrib.seq2seq.sequence_loss(r_real_logits, labeled_text_target, weight_l_txt)
        r_f_loss = tf.contrib.seq2seq.sequence_loss(r_fake_logits, unlabeled_text_target, weight_u_txt)
        r_t_loss = tf.contrib.seq2seq.sequence_loss(r_target_logits, labeled_text_target, weight_l_txt)

        rec_loss = r_r_loss + r_f_loss + r_t_loss

        # loss: generator
        gen_loss = r_f_loss - d_f_loss

        # optimization operator
        dis_op = self.train_operator(loss_scope='loss/discriminator', loss=dis_loss, weights=d_weights)
        rec_op = self.train_operator(loss_scope='loss/reconstructor', loss=rec_loss, weights=r_weights)
        gen_op = self.train_operator(loss_scope='loss/generator', loss=gen_loss, weights=g_weights)

        # define train operator
        step_op = self._time.assign(self._time + 1)
        gan_train_op = tf.group(gen_op, dis_op, rec_op)

        self.train_op = tf.group(gan_train_op, step_op)
        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        print('build done!')

    def train_operator(self, loss_scope, loss, weights):
        with tf.variable_scope(loss_scope):
            # optimizer
            optim = tf.train.AdamOptimizer(learning_rate=self.lr)

            # update
            optim = optim.minimize(loss, var_list=weights)
            tf.summary.scalar(loss_scope, loss)

        return optim


if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    sess = tf.Session()
    scope = tf.get_variable_scope()
    discriminator = RNNDiscriminator(emb_scope=scope, num_classes=2, vocab_size=400,
                                     embedding_units=64, hidden_units=64)
    generator = Seq2SeqGenerator(emb_scope=scope, namescope='generator', vocab_size=400,
                                 embedding_units=64, enc_units=256, dec_units=256, batch_size=32)
    reconstructor = Seq2SeqGenerator(emb_scope=scope, namescope='reconstructor', vocab_size=400,
                                     embedding_units=64, enc_units=256, dec_units=256, batch_size=32)
    gan = SeqGAN(sess, discriminator, generator, reconstructor, emb_scope=scope)
    gan.build_gan()
    print('done gan')
