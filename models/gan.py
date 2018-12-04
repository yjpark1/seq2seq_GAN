import math
import tensorflow as tf
import numpy as np

from models.discriminator import RNNDiscriminator
from models.generator import Seq2SeqGenerator

def get_scope_variables(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

global MAX_SUMMARY_LEN
global MAX_TEXT_LEN

MAX_SUMMARY_LEN = 100
MAX_TEXT_LEN = 1000


class SeqGAN():
    def __init__(self, sess, discriminator, generator, reconstructor,
                 learning_rate=0.001):
        self.sess = sess
        self.D = discriminator
        self.G = generator
        self.R = reconstructor

        # hyperparameters
        self.lr = learning_rate
        self.learn_phase = True

        # training parameter
        self.time = 0
        self.batch_size = 32
        self.vocab_size = 400
        self.output_max_length = 100

        # input placeholder
        # self.inputs = tf.placeholder(dtype=tf.int32, shape=(self._batch_size, None), name='inputs_text')
        # self.inputs_length = tf.reduce_sum(tf.to_int32(tf.not_equal(self.inputs, 1)), -1)
        # self.latent = tf.placeholder(dtype=tf.float32, shape=(self._batch_size, self._num_latent), name='inputs_latent')
        self.sample_pl = tf.placeholder(dtype='bool', shape=(), name='sample')
        self._time = tf.Variable(0, name='time')

    def build_gan(self):
        g_scores, g_seq = self.G.build_model()
        r_seq = tf.placeholder(tf.float32, name='real_seq',
                               shape=[self.batch_size, MAX_SUMMARY_LEN, self.vocab_size])
        r_logits, r_preds = self.D.build_model(r_seq, reuse=False)
        f_logits, f_preds = self.D.build_model(g_scores, reuse=True)

        d_weights = get_scope_variables('discriminator')
        g_weights = get_scope_variables('generator')

        dis_op = self.discriminator_op(r_logits, f_logits, d_weights)
        gen_op = self.generator_op(g_seq, f_logits, g_scores, g_weights)

        step_op = self._time.assign(self.time + 1)

        if self.learn_phase is None:
            gan_train_op = tf.group(gen_op, dis_op)
        else:
            gan_train_op = tf.cond(
                tf.equal(tf.mod(self.time, self.learn_phase), 0),
                lambda: gen_op,
                lambda: dis_op)

        self.train_op = tf.group(gan_train_op, step_op)
        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def discriminator_op(self, r_logits, f_logits, d_weights):
        with tf.variable_scope('loss/discriminator'):
            dis_optim = tf.train.AdamOptimizer(learning_rate=self.lr)

            r_loss = -tf.reduce_mean(r_logits)
            f_loss = tf.reduce_mean(f_logits)
            d_loss = r_loss + f_loss

            tf.summary.scalar('d_loss', d_loss)
            d_optim = dis_optim.minimize(d_loss, var_list=d_weights)

        return d_optim

    def generator_op(self, g_seq, f_logits, g_scores, g_weights):
        with tf.variable_scope('loss/generator'):
            gen_optim = tf.train.AdamOptimizer(learning_rate=self.lr)
            reward_op = tf.train.GradientDescentOptimizer(1e-3)

            g_seq = tf.one_hot(g_seq, self.vocab_size)
            g_scores = tf.clip_by_value(g_scores * g_seq, 1e-20, 1)

            expected_reward = tf.Variable(tf.zeros((self.output_max_length,)))
            reward = f_logits - expected_reward[:tf.shape(f_logits)[1]]
            mean_reward = tf.reduce_mean(reward)

            exp_reward_loss = tf.reduce_mean(tf.abs(reward))
            exp_op = reward_op.minimize(
                exp_reward_loss, var_list=[expected_reward])

            reward = tf.expand_dims(tf.cumsum(reward, axis=1, reverse=True), -1)
            gen_reward = tf.log(g_scores) * reward
            gen_reward = tf.reduce_mean(gen_reward)

            gen_loss = -gen_reward

            gen_op = gen_optim.minimize(gen_loss, var_list=g_weights)

            g_optim = tf.group(gen_op, exp_op)

        tf.summary.scalar('loss/expected_reward', exp_reward_loss)
        tf.summary.scalar('reward/mean', mean_reward)
        tf.summary.scalar('reward/generator', gen_reward)

        return g_optim


if __name__ == '__main__':
    discriminator = RNNDiscriminator(num_classes=2, vocab_size=400,
                                     embedding_units=64, hidden_units=64)
    generator = Seq2SeqGenerator(vocab_size=400, embedding_units=64,
                                 enc_units=256, dec_units=256, batch_size=32)
    reconstructor = Seq2SeqGenerator(vocab_size=400, embedding_units=64,
                                     enc_units=256, dec_units=256, batch_size=32)
    sess = None
    gan = SeqGAN(sess, discriminator, generator, reconstructor)
    gan.build_gan()
