from models.discriminator import RNNDiscriminator
from models.generator import Seq2SeqGenerator
from models.gan import SeqGAN
from train import hyperparameter as H

discriminator = RNNDiscriminator(num_classes=2, vocab_size=H.vocab_size,
                                 embedding_units=H.embedding_units, hidden_units=H.rnn_units)
generator = Seq2SeqGenerator(vocab_size=H.vocab_size, embedding_units=H.embedding_units,
                             enc_units=H.rnn_units, dec_units=H.rnn_units, batch_size=H.batch_size)
reconstructor = Seq2SeqGenerator(vocab_size=H.vocab_size, embedding_units=H.embedding_units,
                                 enc_units=H.rnn_units, dec_units=H.rnn_units, batch_size=H.batch_size)
sess = None
gan = SeqGAN(sess, discriminator, generator, reconstructor)
gan.build_gan()

