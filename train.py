
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout

import pandas as pd
import numpy as np

from util import learning_rate_reduction

# Read data
train = pd.read_csv('data/train.csv')
labels = train.ix[:,0].values.astype('int32')
X_train = (train.ix[:,1:].values).astype('float32')
X_test = (pd.read_csv('data/test.csv').values).astype('float32')

print('<---------INFO----------------------->')
print('tipo de train .........: ', type(train))
print('train.shape ...........: ', train.shape)
print('Labels ................: ', labels)
print('size Labels ...........: ', len(labels))
print('tipo de X_train........: ', type(X_train))
print('tipo de X_test.........: ', type(X_test))
print('-'*30)

# convert list of labels to binary class matrix
y_train = np_utils.to_categorical(labels)
print('convert list of labels to binary class matrix \n > size y_train = ', len(y_train), ' = ', y_train)

# pre-processing: divide by max and substract mean
scale = np.max(X_train)
X_train /= scale
X_test /= scale


input_dim = X_train.shape[1]
nb_classes = y_train.shape[1]

print('input_dim  = ', input_dim)
print('nb_classes  ', nb_classes)


def MLP(pesos='None'):
    # Here's a Deep Dumb MLP (DDMLP)
    print('Here s a Deep Dumb MLP (DDMLP)')
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dropout(0.15))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.15))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    if pesos:
        print('loading weigths ')
        model.load_weights(pesos)
    else:
        print('without weight loading')

    return model

if __name__ == '__main__':

    print('define model')
    model = MLP()

    # we'll use categorical xent for the loss, and RMSprop as the optimizer
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    print("Training...")
    model.fit(X_train, y_train, batch_size=16, nb_epoch=10, validation_split=0.1, verbose=1, callbacks=[learning_rate_reduction])  # show_accuracy=True,

    print("Generating test predictions...")
    preds = model.predict_classes(X_test, verbose=0)


    def write_preds(preds, fname):
        pd.DataFrame({"ImageId": list(range(1, len(preds) + 1)), "Label": preds}).to_csv(fname, index=False,
                                                                                         header=True)

    write_preds(preds, "keras-mlp.csv")

    print('save model')
    model.save_weights('model_MLP_Keras.hdf5')
















