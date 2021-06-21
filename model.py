import matplotlib.pyplot as plt

from keras.layers import Attention, Bidirectional, Conv1D, Dense, Embedding, Dropout, Input, LSTM, MaxPooling1D, TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
from utils import prepare_sequential
from utils import SEED, LABELS, MAX_LENGTH
from sklearn.model_selection import train_test_split


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
    X, y, emb_matrix = prepare_sequential('~/gensim-data/glove.6B/glove.6B.300d.txt', sentence_tokenize=False, test=True)

    # 60-20-20 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    #X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.25, random_state=SEED)

    models = {}
    m = build_model(emb_matrix)
    for label in LABELS:
        history = m.fit(X_train, y_train[label], epochs=10)
        for metric in history.history.keys():
            plt.plot(history.history[metric])
        fname = 'model_' + label + '.png'
        plt.savefig(fname)
        print('Plot:', fname)
