import tensorflow as tf
from models.discriminator import RNNDiscriminator
from models.generator import Seq2SeqGenerator
from models.gan import SeqGAN
from train import hyperparameter as H
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sess = tf.Session()
scope = tf.get_variable_scope()
discriminator = RNNDiscriminator(emb_scope=scope, num_classes=2, vocab_size=H.vocab_size,
                                 embedding_units=H.embedding_units, hidden_units=H.rnn_units)
generator = Seq2SeqGenerator(emb_scope=scope, namescope='generator', vocab_size=H.vocab_size,
                             embedding_units=H.embedding_units, enc_units=H.rnn_units,
                             dec_units=H.rnn_units, batch_size=H.batch_size)
reconstructor = Seq2SeqGenerator(emb_scope=scope, namescope='reconstructor',
                                 vocab_size=H.vocab_size, embedding_units=H.embedding_units,
                                 enc_units=H.rnn_units, dec_units=H.rnn_units, batch_size=H.batch_size)
gan = SeqGAN(sess, discriminator, generator, reconstructor, emb_scope=scope)
gan.build_gan()

