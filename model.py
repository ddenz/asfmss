from keras.layers import Embedding, Conv1D, Dropout, MaxPooling1D
from keras.models import Sequential
from keras.optimizers import Adam
from utils import prepare_sequential
from utils import SEED
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


if __name__ == '__main__':
    X, y, embedding_matrix = prepare_sequential('/Users/andre/gensim-data/glove.6B/glove.6B.300d.txt')

    # 60-20-20 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.25, random_state=SEED)


