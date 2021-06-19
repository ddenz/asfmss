import docx
import logging
import numpy as np
import pandas as pd
import os
import re
import spacy

from keras.preprocessing.sequence import pad_sequences

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors

from sklearn.preprocessing import LabelBinarizer

env = 'remote'

if env == 'local':
    NORM_DIR = '/Users/Andre/workspace/PycharmProjects/asfmss/Quest ASFMSS/Data/Audio/Normalized'
    CLEAN_DIR = '/Users/Andre/workspace/PycharmProjects/asfmss/Quest ASFMSS/Data/Audio/Cleaned'
    FEATS_DIR = '/Users/Andre/workspace/PycharmProjects/asfmss/Quest ASFMSS/Data/Audio/Cleaned/Features'
    CODING_DIR = '/Users/Andre/workspace/PycharmProjects/asfmss/Quest ASFMSS/Data/Coding'
    TEXT_DIR = '/Users/Andre/workspace/PycharmProjects/asfmss/Quest ASFMSS/Data/Transcripts/TXT/Cleaned'
else:
    CODING_DIR = './data'

GENSIM_DATA_DIR = '~/gensim-data/'

SEED = 7
UNK = '<UNKNOWN>'
MAX_LENGTH = 400

LABELS = ['t2_fmss_IS', 't2_fmss_war', 't2_fmss_eoi', 't2_rel', 't2_ee']

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

nlp = spacy.load('en_core_web_sm')


def word2txt():
    """
    Extract text from Word documents and save as text.
    :return:
    """
    pin = 'data/test.docx'
    pout = os.path.splitext(pin)[0] + '.txt'

    doc = docx.Document(pin)
    text = '\n'.join([p.text for p in doc.paragraphs])

    with open(pout, 'w') as fout:
        print(text, file=fout)
    fout.close()


def load_data_to_dataframe():
    """
    Load all data to a dataframe.
    :return:
    """
    df_coding = pd.read_excel(os.path.join(CODING_DIR, 'Quest_ASFMSS_coding.xlsx'))

    inds = df_coding[df_coding.columns[1:]].dropna(how='all').index
    df_coding = df_coding.loc[inds]
    # Remove 69
    df_coding = df_coding.loc[df_coding.idnum != 69]

    """
    Only retain useful features
    IS = initial statement
    war = warmth
    eoi = emotional over-involvement
    rel = rela
    """
    df_coding = df_coding[['idnum', 't2_fmss_IS', 't2_fmss_war', 't2_fmss_eoi', 't2_rel', 't2_ee']]

    # Insert text for each id
    id_map = {re.sub('(Q[0-9]+)[\_ ].+', '\g<1>', f): f for f in os.listdir(TEXT_DIR)}
    df_coding['text'] = df_coding.idnum.apply(
        lambda x: open(os.path.join(TEXT_DIR, id_map['Q' + str(x)]), 'r', encoding='latin-1').read())

    return df_coding.reset_index(drop=True)


def create_token_index_mappings(texts, sentence_tokenized_input=False):
    logging.info('Creating token-index mappings...')
    # create mappings of words to indices and indices to words
    UNK = '<UNKNOWN>'
    # PAD = '<PAD>'
    token_counts = {}

    if sentence_tokenized_input:
        for doc in texts:
            for sent in doc:
                for token in sent:
                    c = token_counts.get(token, 0) + 1
                    token_counts[token] = c
    else:
        for doc in texts:
            for token in doc:
                c = token_counts.get(token, 0) + 1
                token_counts[token] = c

    vocab = sorted(token_counts.keys())
    # start indexing at 1 as 0 is reserved for padding
    token2index = dict(zip(vocab, list(range(1, len(vocab) + 1))))
    token2index[UNK] = len(vocab) + 1
    # token2index[PAD] = len(vocab) + 2
    index2token = {value: key for (key, value) in token2index.items()}
    assert index2token[token2index['help']] == 'help'

    return token_counts, index2token, token2index


def load_data(test=False):
    if test:
        return pd.read_csv(CODING_DIR + '/asfmss_dummy_data.csv')
    return pd.read_csv(CODING_DIR + '/Quest_ASFMSS_all_data.csv')


def load_embeddings(emb_path):
    model = None
    w2v_path = os.path.splitext(emb_path)[0] + '_w2v.txt'
    glove2word2vec(glove_input_file=emb_path, word2vec_output_file=w2v_path)
    model = KeyedVectors.load_word2vec_format(w2v_path, binary=False)
    return model


def prepare_sequential(emb_path, sentence_tokenize=False, test=False):
    logging.info('Preparing sequential data (' + emb_path + ')...')

    df_data = load_data(test=test)

    texts = []

    for doc in nlp.pipe(df_data.text):
        texts.append(spacy_tokenize(doc, sentence_tokenize=sentence_tokenize))

    embedding_vectors = load_embeddings(emb_path)

    token_counts, index2token, token2index = create_token_index_mappings(texts, sentence_tokenized_input=sentence_tokenize)

    # create mapping of words to their embeddings
    emb_map = {}
    for w in embedding_vectors.key_to_index:
        emb_map[w] = embedding_vectors.get_vector(w)

    vocab_size = len(token_counts)
    embed_len = embedding_vectors['help'].shape[0]
    embedding_matrix = np.zeros((vocab_size + 1, embed_len))

    # initialize the embedding matrix
    logging.info('Initializing embeddings matrix...')
    for word, i in token2index.items():
        if i >= vocab_size:
            continue
        if word in embedding_vectors:
            embedding_vector = embedding_vectors.get_vector(word)
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    logging.info('Preparing labels...')

    lb = LabelBinarizer()

    for label in LABELS:
        df_data[label] = lb.fit_transform(df_data[label])

    logging.info('Preparing features...')

    if sentence_tokenize:
        x = [[[token2index.get(token, token2index[UNK]) for token in sent] for sent in doc] for doc in texts]
        x = [pad_sequences(sent, maxlen=MAX_LENGTH, padding='post') for sent in x]
    else:
        x = [[token2index.get(token, token2index[UNK]) for token in doc] for doc in texts]
        x = pad_sequences(x, maxlen=MAX_LENGTH, padding='post')

    return x, df_data[LABELS], embedding_matrix


def process_token(token):
    if token.like_url:
        return '__URL__'
    elif token.like_num:
        return '__NUM__'
    else:
        form = token.lower_
        form = re.sub('[\!\"#\$%&\(\)\*\+,\./:;<=>\?@\[\\]\^_`\{\|\}\~]+', '', form)
        form = re.sub('([^\-,]+)[\-,]', '\g<1>', form)
        form = re.sub('^([^\.]+)\.', '\g<1>', form)
        return form


def spacy_tokenize(doc, sentence_tokenize=False):
    if isinstance(doc, str):
        doc = nlp(doc)
    if sentence_tokenize:
        return [[process_token(token) for token in sent if (not token.is_punct or token.is_space)] for sent in doc.sents]
    return [process_token(token) for token in doc]


if __name__ == '__main__':
    #df = load_data_to_dataframe()
    #df.to_csv(CODING_DIR + '/Quest_ASFMSS_all_data.csv', encoding='utf-8', index=False)
    X, y, embedding_matrix = prepare_sequential(GENSIM_DATA_DIR + '/glove.6B/glove.6B.300d.txt', sentence_tokenize=False, test=True)
