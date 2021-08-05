import logging

import matplotlib.pyplot as plt

from tensorflow.keras.layers import Bidirectional, Dense, Embedding, Dropout, Input, LSTM, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from utils import prepare_sequential_features, load_sequential_features
from utils import SEED, LABELS, MAX_LENGTH
from sklearn.model_selection import train_test_split

OUTPUT_DIR = './output'


def build_model(params, emb_matrix, n_labels, audio=True, text=True):
    if not (text or audio):
        raise Exception('-- Please specify input type (text/audio).')

    logging.info('-- Loading audio:' + str(audio))
    logging.info('-- Loading text:' + str(text))

    tf_dim = params['text_feature_dim']
    af_dim = params['audio_feature_dim']
    nunits1 = params['text_lstm_nunits']
    nunits2 = params['audio_lstm_nunits']
    dropout = params['dropout']
    lr = params['lr']

    tf_inputs = None
    af_inputs = None
    tf_emb = None
    tf_bilstm = None
    af_lstm = None
    merged = None
    model = None

    if text:
        tf_inputs = Input(shape=(tf_dim,), name='tf_inputs')
        tf_emb = Embedding(input_dim=emb_matrix.shape[0], output_dim=emb_matrix[0].shape[0], input_length=tf_dim,
                           weights=[emb_matrix], trainable=False)(tf_inputs)
        tf_bilstm = Bidirectional(LSTM(nunits1, activation='sigmoid', recurrent_dropout=0.2, recurrent_activation='sigmoid',
                                       return_sequences=False))(tf_emb)
    if audio:
        af_inputs = Input(shape=af_dim, name='af_inputs')
        af_lstm = LSTM(nunits2, return_sequences=False)(af_inputs)

    if text and audio:
        merged = concatenate([tf_bilstm, af_lstm])
        merged = Dropout(dropout)(merged)
    elif text:
        merged = tf_bilstm
    elif audio:
        merged = af_lstm

    #f_is = Dense(1)(aftf_conc)
    #f_war = Dense(1)(aftf_conc)
    #f_eoi = Dense(1)(aftf_conc)
    #f_rel = Dense(1)(aftf_conc)
    output = Dense(n_labels, activation='softmax')(merged)

    if text and audio:
        model = Model(inputs=[tf_inputs, af_inputs], outputs=[output], name='ee_full_bimodal')
    elif text:
        model = Model(inputs=[tf_inputs], outputs=[output], name='ee_full_bimodal')
    elif audio:
        model = Model(inputs=[af_inputs], outputs=[output], name='ee_full_bimodal')

    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy')

    model.summary()

    return model


if __name__ == '__main__':
    pred_label = 't2_ee'
    n_labels = 3  # this will be the "theoretical" number of possible classes (e.g. for t2_ee: Low, High, Borderline = 3)

    # tf, af, y, emb_matrix = prepare_sequential_features('~/gensim-data/glove.6B/glove.6B.300d.txt', sentence_tokenize=False, test=False, save=True)
    tf, af, y, embedding_matrix = load_sequential_features()

    y = y[pred_label]  # select only the label we want to predict
    
    from pprint import pprint
    pprint(y)

    assert len(tf) == len(af) == len(y)

    # 60-20-20 split
    tf_train, tf_test, af_train, af_test, y_train, y_test = train_test_split(tf, af, y, test_size=0.2, random_state=SEED)
    # tf_dev, tf_test, af_dev, af_test, y_test, y_dev = train_test_split(X_train, y_train, test_size=0.25, random_state=SEED)

    """
    param_grid = {
        'lstm_nunits': [25, 50, 100, 150],
        'dropout': [0.1, 0.2, 0.5],
        'lr': [0.1, 0.01, 0.001],
        'epochs': [10, 20, 30, 50]
    }
    """

    parameters = {
        'text_feature_dim': MAX_LENGTH,
        'audio_feature_dim': af[0].shape,
        'text_lstm_nunits': 150,
        'audio_lstm_nunits': 300,
        'dropout': 0.2,
        'lr': 0.1
    }

    at_model = build_model(parameters, embedding_matrix, n_labels, text=True, audio=False)

    at_model.fit(
        {'tf_inputs': tf_train, 'af_inputs': af_train},
        {'dense': y_train},
        epochs=20,
        batch_size=32
    )

    """
    models = {}
    m = build_model(emb_matrix)
    for label in LABELS:
        history = m.fit(X_train, y_train[label], epochs=10)
        for metric in history.history.keys():
            plt.plot(history.history[metric])
            fname = 'model_' + label + '.png'
            pout = OUTPUT_DIR + '/' + fname
            plt.savefig()
            print('Plot:', pout)
    """