import logging
import tensorflow as tf

from tensorflow.keras.layers import Bidirectional, Dense, Embedding, Dropout, Input, LSTM, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from utils import prepare_sequential_features, load_sequential_features
from utils import SEED, LABELS, MAX_LENGTH
from sklearn.model_selection import KFold, train_test_split

OUTPUT_DIR = './output'


class Attention(tf.keras.Model):
    """
    From https://matthewmcateer.me/blog/getting-started-with-attention-for-classification/
    """
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = tf.nn.tanh(
            self.W1(features) + self.W2(hidden_with_time_axis))
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


def build_model(params, emb_matrix, n_labels, audio=True, text=True):
    if not (text or audio):
        raise Exception('-- Please specify input type (text/audio).')

    logging.info('-- Loading audio:' + str(audio))
    logging.info('-- Loading text:' + str(text))

    tf_dim = params['text_feature_dim']
    af_dim = params['audio_feature_dim']
    nunits1 = params['text_lstm_nunits']
    nunits2 = 2 * nunits1
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

    output = Dense(n_labels, activation='softmax', name='output')(merged)

    if text and audio:
        model = Model(inputs=[tf_inputs, af_inputs], outputs=[output], name='ee_bimodal')
    elif text:
        model = Model(inputs=[tf_inputs], outputs=[output], name='ee_text')
    elif audio:
        model = Model(inputs=[af_inputs], outputs=[output], name='ee_audio')

    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy')

    return model


def build_text_model(params, emb_matrix, n_labels):
    logging.info('-- Loading text')

    tf_dim = params['text_feature_dim']
    nunits1 = params['text_lstm_nunits']
    dropout = params['dropout']
    lr = params['lr']

    tf_inputs = Input(shape=(tf_dim,), name='tf_inputs')
    tf_emb = Embedding(input_dim=emb_matrix.shape[0], output_dim=emb_matrix[0].shape[0], input_length=tf_dim,
                       weights=[emb_matrix], trainable=False)(tf_inputs)
    tf_bilstm = Bidirectional(LSTM(nunits1, activation='sigmoid', recurrent_dropout=0.2, recurrent_activation='sigmoid',
                                   return_sequences=True))(tf_emb)
    (lstm, forward_h, forward_c, backward_h, backward_c) = Bidirectional(
        LSTM(nunits1, return_sequences=True, return_state=True), name="bi_lstm_1")(tf_bilstm)
    state_h = concatenate()([forward_h, backward_h])
    state_c = concatenate()([forward_c, backward_c])
    context_vector, attention_weights = Attention(10)(lstm, state_h)
    dense1 = Dense(20, activation="relu")(context_vector)
    dropout = Dropout(0.05)(dense1)
    output = Dense(n_labels, activation="sigmoid")(dropout)

    model = Model(inputs=tf_inputs, outputs=output)
    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy')

    return model


if __name__ == '__main__':
    pred_label = 't2_ee'
    n_labels = 3  # this is the "theoretical" number of possible classes (e.g. for t2_ee: Low, High, Borderline = 3)

    # tf, af, y, emb_matrix = prepare_sequential_features('~/gensim-data/glove.6B/glove.6B.300d.txt', sentence_tokenize=False, test=False, save=True)
    # tf, af, y, embedding_matrix = load_sequential_features(audio_type='mfcc', text_type='glove.6B')
    tf, _, y, embedding_matrix = load_sequential_features(audio_type=None, text_type='glove.6B')

    y = y[pred_label]  # select only the label we want to predict
    
    #assert len(tf) == len(af) == len(y)

    # 60-20-20 split
    # tf_train, tf_test, af_train, af_test, y_train, y_test = train_test_split(tf, af, y, test_size=0.2, random_state=SEED)
    # tf_dev, tf_test, af_dev, af_test, y_test, y_dev = train_test_split(X_train, y_train, test_size=0.25, random_state=SEED)
    tf_train, tf_test, y_train, y_test = train_test_split(tf, y, test_size=0.2, random_state=SEED)

    print(tf_train.shape)
    # print(af_train.shape)
    print(y_train.shape)

    params = {
        'text_feature_dim': MAX_LENGTH,
        'audio_feature_dim': None,
        'text_lstm_nunits': 150,
        'dropout': 0.2,
        'lr': 0.001,
        'epochs': 1
    }

    #m = build_model(params, embedding_matrix, n_labels)
    m = build_text_model(params, embedding_matrix, n_labels)
    print(m.summary())
    m.fit(tf_train, y_train)