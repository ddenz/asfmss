import matplotlib.pyplot as plt

from keras.layers import Bidirectional, Dense, Embedding, Dropout, Input, LSTM, concatenate
from keras.models import Model
from keras.optimizers import Adam
from utils import prepare_sequential_features
from utils import SEED, LABELS, MAX_LENGTH
from sklearn.model_selection import train_test_split

OUTPUT_DIR = './output'


def build_model(embedding_matrix):
    inputs = Input(shape=(MAX_LENGTH,))
    emb = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix[0].shape[0],
                    input_length=MAX_LENGTH, weights=[embedding_matrix], trainable=False)(inputs)
    bilstm = Bidirectional(LSTM(300, activation='sigmoid', recurrent_dropout=0.2, recurrent_activation='sigmoid',
                                                return_sequences=False))(emb)
    do1 = Dropout(0.2)(bilstm)
    output = Dense(1)(do1)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy')

    return model


if __name__ == '__main__':
    tf, af, y, emb_matrix = prepare_sequential_features('~/gensim-data/glove.6B/glove.6B.300d.txt', sentence_tokenize=False, test=True)

    assert len(tf) == len(af) == len(y)

    # 60-20-20 split
    tf_train, tf_test, af_train, af_test, y_train, y_test = train_test_split(tf, af, y, test_size=0.2, random_state=SEED)
    # X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.25, random_state=SEED)

    tf_inputs = Input(shape=(MAX_LENGTH,), name='tf_inputs')
    af_inputs = Input(shape=af[0].shape, name='af_inputs')

    tf_emb = Embedding(input_dim=emb_matrix.shape[0], output_dim=emb_matrix[0].shape[0], input_length=MAX_LENGTH,
                       weights=[emb_matrix], trainable=True)(tf_inputs)
    tf_bilstm = Bidirectional(LSTM(300, activation='sigmoid', recurrent_dropout=0.2, recurrent_activation='sigmoid',
                                   return_sequences=True))(tf_emb)

    af_lstm = LSTM(600, return_sequences=False)(af_inputs)

    aftf_conc = concatenate([tf_bilstm, af_lstm])

    aftf_conc = Dropout(0.2)(aftf_conc)

    #f_is = Dense(1)(aftf_conc)
    #f_war = Dense(1)(aftf_conc)
    #f_eoi = Dense(1)(aftf_conc)
    #f_rel = Dense(1)(aftf_conc)
    f_ee = Dense(1)(aftf_conc)

    model = Model(inputs=[tf_inputs, af_inputs], outputs=[f_ee], name='ee_full_bimodal')
    model.compile(optimizer=Adam(lr=0.1), loss='categorical_crossentropy')

    model.summary()

    model.fit(
        {'tf_inputs': tf_train, 'af_inputs': af_train},
        {'dense': y_train[['t2_ee']]},
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