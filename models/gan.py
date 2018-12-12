"""
https://github.com/sminocha/text-generation-GAN/blob/master/model.py
"""
import tensorflow as tf
from models.utils import dynamic_time_pad
from train import hyperparameter as H


def get_scope_variables(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)


class SeqGAN:
    def __init__(self, sess, discriminator, generator, reconstructor, emb_scope,
                 d_lr = 0.00001, g_lr = 0.001, r_lr = 0.001, alpha = 0.001):
        self.sess = sess
        self.D = discriminator
        self.G = generator
        self.R = reconstructor

        # hyperparameters
        self.d_lr = d_lr
        self.g_lr = g_lr
        self.r_lr = r_lr
        self.alpha = alpha

        # training parameter        
        #self.batch_size = H.batch_size
        self.vocab_size = H.vocab_size
        self.max_output_length = H.max_summary_len

        # manage scope
        self.emb_scope = emb_scope
        
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

        # build generator
        g_real_logits, g_real_seq = self.G.build_model(self.labeled_text, self.labeled_text_lengths,
                                                       reuse=False, emb_reuse=False)
        g_fake_logits, g_fake_seq = self.G.build_model(self.unlabeled_text, self.unlabeled_text_lengths,
                                                       reuse=True, emb_reuse=True)
        
        self.generated_sequence = g_fake_seq
        
        # build discriminator
        d_real_logits, d_real_preds = self.D.build_model(self.real_summary, reuse=False)
        d_fake_logits, d_fake_preds = self.D.build_model(g_fake_seq, reuse=True)

        # build reconstructor
        r_real_logits, r_real_seq = self.R.build_model(g_real_seq, self.real_summary_length,
                                                       reuse=False, emb_reuse=True)
        r_target_logits, r_target_seq = self.R.build_model(self.real_summary, self.real_summary_length,
                                                           reuse=True, emb_reuse=True)
        r_fake_logits, r_fake_seq = self.R.build_model(g_fake_seq, self.real_summary_length,
                                                       reuse=True, emb_reuse=True)

        # get trainable parameters
        d_weights = get_scope_variables('discriminator') + get_scope_variables('embedding')
        r_weights = get_scope_variables('reconstructor') + get_scope_variables('embedding')
        g_weights = get_scope_variables('generator') + get_scope_variables('embedding')

        # loss: discriminator
        #d_r_loss = -tf.reduce_mean(tf.log(tf.clip_by_value(d_real_preds, 1e-7, 1. - 1e-7)))  # r_preds -> 1.
        #d_f_loss = -tf.reduce_mean(tf.log(1 - tf.clip_by_value(d_fake_preds, 1e-7, 1. - 1e-7)))  # g_preds -> 0.
        d_r_loss = -tf.reduce_mean(d_real_logits)
        d_f_loss = tf.reduce_mean(d_fake_logits)

        dis_loss = d_r_loss + d_f_loss

        # loss: reconstructor
        labeled_text_target = tf.math.argmax(self.labeled_text, axis=-1)
        unlabeled_text_target = tf.math.argmax(self.unlabeled_text, axis=-1)

        # pad output
#        r_real_logits = dynamic_time_pad(r_real_logits, H.max_text_len, batch_size)
        r_fake_logits = dynamic_time_pad(r_fake_logits, H.max_text_len, batch_size)
        r_target_logits = dynamic_time_pad(r_target_logits, H.max_text_len, batch_size)

#        r_r_loss = tf.contrib.seq2seq.sequence_loss(r_real_logits, labeled_text_target, weight_l_txt)
        r_f_loss = tf.contrib.seq2seq.sequence_loss(r_fake_logits, unlabeled_text_target, weight_u_txt)
        r_t_loss = tf.contrib.seq2seq.sequence_loss(r_target_logits, labeled_text_target, weight_l_txt)

        #rec_loss = r_r_loss + r_f_loss + r_t_loss
        rec_loss = r_t_loss + r_f_loss
        # loss: generator
        gen_loss = self.alpha * r_f_loss - d_f_loss
        
        self.dis_loss = dis_loss
        self.gen_loss = gen_loss
        self.rec_loss = rec_loss
                
        # optimization operator
        self.dis_op = self.train_operator(loss_scope='loss/discriminator', loss=dis_loss, weights=d_weights,
                                          lr = self.d_lr, regul = True)
        self.rec_op = self.train_operator(loss_scope='loss/reconstructor', loss=rec_loss, weights=r_weights,
                                          lr=self.r_lr, regul=True)
        self.gen_op = self.train_operator(loss_scope='loss/generator', loss=gen_loss, weights=g_weights,
                                          lr=self.g_lr, regul=True)

        # test operator
        # self.test_op = self.train_operator(loss_scope='loss/reconstructor', loss=r_t_loss, weights=r_weights)
        # self.out = [r_target_seq, labeled_text_target]
        # self.loss = r_t_loss

        # define train operator
        self.train_op = tf.group(self.gen_op, self.dis_op, self.rec_op)
        # self.train_op = tf.group(self.test_op)
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        print('build GAN model done!')

    def train_operator(self, loss_scope, loss, weights, lr, regul = None):
        with tf.variable_scope(loss_scope):
            # optimizer

            optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)

            if regul is not None:
                _loss = sum([tf.nn.l2_loss(w) for w in weights]) * 1e-4
                loss = loss + _loss

            gvs = optim.compute_gradients(loss, weights)
            capped_gvs = [(tf.clip_by_value(grad, -.25, .25), var) for grad, var in gvs if grad is not None]
            op_train = optim.apply_gradients(capped_gvs)

            # update
            # optim = optim.minimize(loss, var_list=weights)
            # tf.summary.scalar(loss_scope, loss)

        return op_train
    
'''
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
'''