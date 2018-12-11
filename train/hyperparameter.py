import numpy as np
import utils.data_file as helper

path = 'E:/Abstractive/text_summarization/data'
vocab_file = path + '/vocab.csv'
abstractive_article_komoran = np.load(path + '/abstractive_article_komoran.npy')
abstractive_summary_komoran = np.load(path + '/abstractive_summary_komoran.npy')
extractive_article_komoran = np.load(path + '/extractive_article_komoran.npy')
extractive_summary_komoran = np.load(path + '/extractive_summary_komoran.npy')
unlabeled_article_komoran = np.load(path + '/unlabeled_article_komoran.npy')

labeled_article = np.concatenate((abstractive_article_komoran,extractive_article_komoran),axis=0)
labeled_summary = np.concatenate((abstractive_summary_komoran,extractive_summary_komoran),axis=0)
unlabeled_article = unlabeled_article_komoran

build_vocab = helper.Vocab(vocab_file)

vocab_size = build_vocab._count
embedding_units = 64
rnn_units = 128
batch_size = 32
max_summary_len = 200
max_text_len = 1000