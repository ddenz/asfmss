from keras.layers import Attention, Bidirectional, Conv1D, Dense, Embedding, Dropout, Input, LSTM, MaxPooling1D, TimeDistributed
from keras.models import Sequential, Model
from keras.optimizers import Adam
from utils import prepare_sequential
from utils import SEED, LABELS
from sklearn.model_selection import train_test_split

MAX_LENGTH = 200


class SentenceEncoder(Sequential):
    def __init__(self, emb_matrix):
        super().__init__()
        self.emb_matrix = emb_matrix

    def build_model(self, optimizer=Adam(lr=0.001), loss='categorical_crossentropy'):
        # input_dim = vocab size, output_dim = embedding size, input_length = sentence length
        self.add(Embedding(input_dim=self.emb_matrix.shape[0], output_dim=self.emb_matrix[0].shape[0],
                           input_length=MAX_LENGTH, weights=[self.emb_matrix], trainable=False))
        self.add(Conv1D(filters=64, kernel_size=7))
        self.add(MaxPooling1D())
        self.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])


def build_model(embedding_matrix):
    inputs = Input(shape=(embedding_matrix.shape[0]))
    emb = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix[0].shape[0],
                    input_length=MAX_LENGTH, weights=[embedding_matrix], trainable=False)(inputs)
    bilstm = LSTM(300, activation='sigmoid', recurrent_dropout=0.2, recurrent_activation='sigmoid',
                  return_sequences=True)(emb)
    bilstm = Bidirectional()(bilstm)
    bilstm = TimeDistributed()(bilstm)
    do1 = Dropout(0.2)(bilstm)
    output = Dense(1)(do1)

    model = Model(inputs=inputs, outputs=output)

    return model


if __name__ == '__main__':
    X, y, emb_matrix = prepare_sequential('~/gensim-data/glove.6B/glove.6B.300d.txt', sentence_tokenize=False, test=True)

    # 60-20-20 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    #X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.25, random_state=SEED)

    m = build_model(emb_matrix)

    m.fit(X_train, y_train[LABELS[0]])
