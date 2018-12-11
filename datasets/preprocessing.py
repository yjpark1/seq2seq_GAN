import glob
import numpy as np
import pandas as pd
import re
from konlpy.tag import Komoran


def read_files(paths, encoding='cp949'):
    # <read labeled text files>
    data = []
    for p in paths:
        file = pd.read_csv(p, encoding=encoding, engine='python')
        data.append(file)
    data = pd.concat(data)

    return data


def RMVFilter(text):
    is_str = np.array([isinstance(x, str) for x in text])
    return np.where(is_str == False)


def remove_footnote(x):
    x_split = x.split('다.')
    isRemove = any([x in x_split[-1] for x in ['ⓒ', '@', '※', '.com']])

    if isRemove:
        x = '다. '.join(x_split[0:-1]) + '다. '

    return x


komoran = Komoran()
def MorphAnal(x):
    x_pos = komoran.pos(x)

    x_out = []
    for char, tag in x_pos:
        if tag in ['SN', 'NR']:
            char = '<num>'
        x_out.append(char)
    x_out = ' '.join(x_out)

    x_out = re.sub(r'(<num> )+', '<num> ', x_out)
    return x_out

# morphological analysis
def MorphAnalDocument(text):
    text_re = []
    for x in text:
        try:
            morph = MorphAnal(x)
        except:
            morph = 'error!'
            print(morph)
        text_re.append(morph)

    # preprocess number 2
    text = np.array([re.sub('<num> . <num>', '<num>', x) for x in text_re])
    return text


def preprocess_text(text):
    # remove string in [] or ()
    text = np.array([re.sub("[\(\[].*?[\)\]]", "", x) for x in text])
    # remove footnote
    text = np.array([remove_footnote(x) for x in text])

    # unified quotation mark
    rmv_charset = ['‘', '’', '“', '”', '\'']
    for rmv_chr in rmv_charset:
        text = np.array([re.sub(rmv_chr, '"', x) for x in text])

    # remove some characters
    text = np.array([re.sub('·', ', ', x) for x in text])
    text = np.array([re.sub('→', ', ', x) for x in text])
    text = np.array([re.sub('\n+', ' ', x) for x in text])
    text = np.array([re.sub('[!#Δ\`©]', ', ', x) for x in text])

    # remove multiple space
    text = np.array([re.sub('\s+', ' ', x) for x in text])

    return text


# read file path
paths_label = glob.glob('datasets/labeled data/*.csv')
paths_unlabel = glob.glob('datasets/unlabeled data/*.csv')
paths_add = glob.glob('datasets/add_labeled data/*.csv')

# load datasets
data_label = read_files(paths_label, encoding='cp949')
data_unlabel = read_files(paths_unlabel, encoding='utf8')
data_add = read_files(paths_add, encoding='utf8')

# split summary & text
# file 1
data_label_summ = data_label.body.values
data_label_text = data_label.origin.values
# file 2
data_unlabel_text = data_unlabel.body.values
# file 3
data_add_summ = data_add.body.values
data_add_text = data_add.origin.values

# filter non string & more than 100 characters
RMVFilter(data_label_summ)
RMVFilter(data_label_text)
RMVFilter(data_unlabel_text)

id1 = RMVFilter(data_add_text)[0]
id2 = RMVFilter(data_add_summ)[0]
id = np.concatenate((id1, id2))
id = np.unique(id)

data_add_text = np.delete(data_add_text, id)
data_add_summ = np.delete(data_add_summ, id)
data_unlabel_text = np.delete(data_unlabel_text, 995)

# preprocess summary
data_label_summ_prep = preprocess_text(data_label_summ)
data_add_summ_prep = preprocess_text(data_add_summ)

# preprocess text
data_label_text_prep = preprocess_text(data_label_text)
data_unlabel_text_prep = preprocess_text(data_unlabel_text)
data_add_text_prep = preprocess_text(data_add_text)

# combind datasets: Text With Summary
summary_origin = np.concatenate([data_label_summ, data_add_summ])
summary = np.concatenate([data_label_summ_prep, data_add_summ_prep])
text_summ_origin = np.concatenate([data_label_text, data_add_text])
text_summ = np.concatenate([data_label_text_prep, data_add_text_prep])

summary = MorphAnalDocument(summary)
text_summ = MorphAnalDocument(text_summ)

TextWithSummary = pd.DataFrame({'summary_origin': summary_origin,
                                'summary': summary,
                                'text_origin': text_summ_origin,
                                'text': text_summ,
                                })

# combind datasets: Text With Summary
text_unlabel = MorphAnalDocument(data_unlabel_text_prep)

TextWithoutSummary = pd.DataFrame({'text_origin': data_unlabel_text,
                                   'text': text_unlabel,
                                   })


idx = (TextWithSummary['summary'].str.len().values > 100) * (TextWithSummary['text'].str.len().values > 200)
TextWithSummary = TextWithSummary[idx]

idx = TextWithoutSummary['text'].str.len().values > 200
TextWithoutSummary = TextWithoutSummary[idx]

TextWithSummary.to_csv('datasets/TextWithSummary.csv', encoding='utf8')
TextWithoutSummary.to_csv('datasets/TextWithoutSummary.csv', encoding='utf8')
