from tensorflow.keras.utils import Sequence
from tensorflow.keras import Sequential, optimizers, preprocessing, initializers
from tensorflow.keras.layers import MaxPooling2D, Dense, BatchNormalization, Dropout, Flatten, InputLayer, Conv2D, Activation
import tensorflow as tf
from tensorflow.python.keras import backend as K
from sklearn import preprocessing, model_selection
from math import ceil
import pandas as pd
import numpy as np
from skimage import io
import os

import skopt
from skopt import gbrt_minimize, gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer

class BatchGenerator(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([preprocessing.StandardScaler().fit_transform(io.imread(os.path.join(PATH, file_name))).reshape(
            (400, 400, 1))
                         for file_name in batch_x]), np.array(batch_y)


def create_model(filters, learning_rate, activation, dropout0, dropout1):
    model = Sequential()
    model.add(Conv2D(filters[0], kernel_size=(5, 5), input_shape=(400, 400, 1),
                     kernel_initializer=initializers.GlorotNormal(), padding="same"))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(activation))

    for index, filter_ in enumerate(filters[1:-2]):
        model.add(Conv2D(filter_, kernel_size=(3, 3), activation=activation, padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Activation(activation))

    model.add(Flatten())
    model.add(Dense(filters[-2]))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout0))

    model.add(Dense(filters[-1]))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout1))

    model.add(Dense(4, activation='softmax'))

    opt = optimizers.Adam(learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['categorical_accuracy'])

    return model


layers = [[8, 16, 16, 8],
          [8, 16, 32, 16, 8],
          [8, 16, 32, 32, 64, 64, 16],
          [8, 16, 32, 64, 8],
          [8, 16, 32, 64, 32, 8],
          [8, 32, 64, 64, 32, 8],
          [16, 32, 64, 128, 64, 8],
          [32, 64, 128, 256, 128, 32],
          [16, 32, 64, 128, 256, 128, 32],
          [32, 64, 64, 128, 256, 128, 32]
          ]
dim_layers = Categorical(layers, name="layers")
dim_learning_rate = Real(low=0.0001, high=0.05, prior="log-uniform", name="learning_rate")
dim_dropout_0 = Real(low=0.1, high=0.3, prior="uniform", name="dropout0")
dim_dropout_1 = Real(low=0.1, high=0.3, prior="log-uniform", name="dropout1")
dim_activation = Categorical(["relu", "tanh", "sigmoid"], name="activation")
dim_batch_size = Categorical([32, 64, 128, 256, 512], "batch_size")

dimensions = [dim_layers, dim_learning_rate, dim_dropout_0,
              dim_dropout_1, dim_activation, dim_batch_size]


@use_named_args(dimensions=dimensions)
def fitness(learning_rate, filters, activation, batch_size, dropout0, dropout1):
    global X, Y
    batch_size = batch_size
    list_acc = []
    sss = model_selection.StratifiedShuffleSplit(n_splits=3, test_size=0.1)
    for train_index, test_index in sss.split(X, Y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = pd.get_dummies(Y[train_index]).values, pd.get_dummies(Y[test_index]).values
        train_batch_generator = BatchGenerator(x_train, y_train, batch_size=batch_size)
        test_batch_generator = BatchGenerator(x_test, y_test, batch_size=batch_size)
        model = create_model(filters, learning_rate, activation, dropout0, dropout1)
        history = model.fit(x=train_batch_generator, verbose=1, epochs=5,
                            validation_data=test_batch_generator, validation_freq=[5])
        list_acc.append(history.history['val_categorical_accuracy'])

    acc = np.array(list_acc).mean()
    print("Acc: {}".format(acc))

    del(model)
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    return -acc

PATH = os.path.join("dataset", "data_augmented")
df = pd.read_csv("dataset/name_data_augmented.csv", index_col=0)
X = df.filename.values
Y = df.label.values

gp_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            n_calls=12,
                            noise= 0.01,
                            n_jobs=-1,
                            kappa = 5,
                            x0=default_parameters)



