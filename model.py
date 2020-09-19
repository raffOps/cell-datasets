

import os
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
import skopt
from skopt import gbrt_minimize, gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


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
# layers = [
#         (8, 16, 16, 8),
#           (8, 16, 32, 16, 8),
#           (8, 16, 32, 32, 64, 64, 16),
#           (8, 16, 32, 64, 8),
#           (8, 16, 32, 64, 32, 8),
#           (8, 32, 64, 64, 32, 8),
#           (16, 32, 64, 128, 64, 8),
#           (32, 64, 128, 256, 128, 32),
#            (16, 32, 64, 128, 256, 128, 32),
#            (32, 64, 64, 128, 256, 128, 32)
#           ]


dim_layer0 = Categorical([8, 16], name="layer0")
dim_layer1 = Categorical([16, 32], name="layer1")
dim_layer2 = Categorical([32, 64], name="layer2")
dim_layer3 = Categorical([64, 128], name="layer3")
dim_layer4 = Categorical([128], name="layer4")
dim_layer5 = Categorical([128, 64], name="layer5")
dim_layer6 = Categorical([32, 16, 8], name="layer6")

dim_learning_rate = Real(low=0.001, high=0.05, prior="log-uniform", name="learning_rate")
dim_dropout_0 = Real(low=0.3, high=0.5, prior="log-uniform", name="dropout0")
dim_dropout_1 = Real(low=0.1, high=0.4, prior="log-uniform", name="dropout1")
dim_activation = Categorical(["relu", "tanh", "sigmoid"], name="activation")
dim_batch_size = Categorical([16, 32, 64], name="batch_size")

dimensions = [dim_layer0, dim_dropout_1, dim_layer2, dim_layer3, dim_layer4, dim_layer5, dim_layer6,
              dim_learning_rate, dim_dropout_0,
              dim_dropout_1, dim_activation, dim_batch_size]


@use_named_args(dimensions=dimensions)
def fitness(**kargs):
    global X, Y
    filters = [kargs[key] for key in kargs.keys() if key.startswith("layer")]
    learning_rate = kargs["learning_rate"]
    activation = kargs["activation"]
    dropout0 = kargs["dropout0"]
    dropout1 = kargs["dropout1"]
    batch_size = kargs["batch_size"]

    print("Filters: {}\nLearnig Rate: {}\nActivation: {}\nDropout 0: {}\nDropout 1: {}\nBatch size: {}\n".format(filters, learning_rate, activation, dropout0, dropout1, batch_size))


    list_acc = []
    sss = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=1)
    for train_index, test_index in sss.split(X, Y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = pd.get_dummies(Y[train_index]).values, pd.get_dummies(Y[test_index]).values
        train_batch_generator = BatchGenerator(x_train, y_train, batch_size=batch_size)
        test_batch_generator = BatchGenerator(x_test, y_test, batch_size=batch_size)
        model = create_model(filters, learning_rate, activation, dropout0, dropout1)
        history = model.fit(x=train_batch_generator, verbose=1, epochs=1,
                            validation_data=test_batch_generator, validation_freq=[1])
        list_acc.append(history.history['val_categorical_accuracy'])
        del(model)
        K.clear_session()
        tf.compat.v1.reset_default_graph()

    acc = np.array(list_acc).mean()
    print("\nAcc: {}\n\n".format(acc))

    return -acc

PATH = os.path.join("dataset", "data_augmented")
df = pd.read_csv("dataset/name_data_augmented.csv", index_col=0)
X = df.filename.values
Y = df.label.values
#x, x_test, y, y_test = train_test_split(X, Y, test_size=0.05)


gp_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            n_calls=50,
                            verbose=1,
                            noise=0.01)

