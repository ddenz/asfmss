""""
CentOS Linux release 7.6.1810 (Core)

Python 3.8.10

keras                     2.4.3              pyhd8ed1ab_0    conda-forge
keras-hypetune            0.1.2                    pypi_0    pypi
numpy                     1.19.5           py38h9894fe3_2    conda-forge
scikit-learn              0.24.2           py38hdc147b9_0    conda-forge
tensorflow                2.4.1            py38h578d9bd_0    conda-forge
"""

import numpy as np

from kerashypetune import KerasGridSearchCV
from tensorflow.keras.layers import Bidirectional, Dense, Embedding, Dropout, Input, LSTM, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold


def build_model(params, emb_matrix, n_labels):
    nunits1 = params['lstm_nunits']
    nunits2 = 2 * nunits1
    dropout = params['dropout']
    lr = params['lr']

    text_inputs = Input(shape=(100,), name='text_inputs')
    tf_emb = Embedding(input_dim=emb_matrix.shape[0],
                       output_dim=emb_matrix[0].shape[0],
                       input_length=400,
                       weights=[emb_matrix],
                       trainable=False)(text_inputs)
    tf_bilstm = Bidirectional(LSTM(nunits1,
                                   activation='sigmoid',
                                   recurrent_dropout=0.2,
                                   recurrent_activation='sigmoid',
                                   return_sequences=False))(tf_emb)
    audio_inputs = Input(shape=(100, 13), name='audio_inputs')
    af_lstm = LSTM(nunits2, return_sequences=False)(audio_inputs)
    merged = concatenate([tf_bilstm, af_lstm])
    merged = Dropout(dropout)(merged)
    output = Dense(n_labels, activation='softmax', name='output')(merged)
    model = Model(inputs=[text_inputs, audio_inputs], outputs=[output])
    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy')

    model.summary()

    return model


tf = np.random.randint(1000, size=(100, 100))
af = np.random.rand(100, 100, 13)
y = np.random.randint(2, size=100)
embedding_matrix = np.random.rand(1000, 100)

parameters = {'lstm_nunits': 300,
              'dropout': 0.2,
              'lr': 0.01}

print(type(tf), type(af), type(y))

model = build_model(parameters, embedding_matrix, 1)

model.fit({'text_inputs': tf, 'audio_inputs': af}, {'output': y}, epochs=1)

param_grid = {
    'lstm_nunits': [25, 50, 150],
    'dropout': [0.1, 0.2, 0.5],
    'lr': [0.001, 0.0001],
    'epochs': [10, 25, 50]
}

hypermodel = lambda x: build_model(x, emb_matrix=embedding_matrix, n_labels=1)

cv = KFold(n_splits=3, random_state=42, shuffle=True)
kgs = KerasGridSearchCV(hypermodel, param_grid, monitor='val_loss', cv=cv, greater_is_better=False)

kgs.search({'text_inputs': tf, 'audio_inputs': af}, {'outputs': y})
"""
Traceback (most recent call last):
  File "example.py", line 78, in <module>
    kgs.search({'text_inputs': tf, 'audio_inputs': af}, {'outputs': y})
  File "/users/k1773308/anaconda3/envs/asfmss/lib/python3.8/site-packages/kerashypetune/kerashypetune.py", line 704, in search
    self._search(x=x, y=y,
  File "/users/k1773308/anaconda3/envs/asfmss/lib/python3.8/site-packages/kerashypetune/kerashypetune.py", line 503, in _search
    for fold, (train_id, val_id) in enumerate(self.cv.split(x, y, groups)):
  File "/users/k1773308/anaconda3/envs/asfmss/lib/python3.8/site-packages/sklearn/model_selection/_split.py", line 324, in split
    X, y, groups = indexable(X, y, groups)
  File "/users/k1773308/anaconda3/envs/asfmss/lib/python3.8/site-packages/sklearn/utils/validation.py", line 356, in indexable
    check_consistent_length(*result)
  File "/users/k1773308/anaconda3/envs/asfmss/lib/python3.8/site-packages/sklearn/utils/validation.py", line 319, in check_consistent_length
    raise ValueError("Found input variables with inconsistent numbers of"
ValueError: Found input variables with inconsistent numbers of samples: [2, 1]
"""
