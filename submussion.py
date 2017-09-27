
from train import MLP
from keras.utils import np_utils
import pandas as pd
import numpy  as np

# DATA

# Read data
train = pd.read_csv('data/train.csv')
labels = train.ix[:,0].values.astype('int32')
X_train = (train.ix[:,1:].values).astype('float32')
X_test = (pd.read_csv('data/test.csv').values).astype('float32')

# convert list of labels to binary class matrix
y_train = np_utils.to_categorical(labels)

# pre-processing: divide by max and substract mean
scale = np.max(X_train)
X_train /= scale
X_test /= scale

input_dim = X_train.shape[1]
nb_classes = y_train.shape[1]


print('<---------INFO----------------------->')
print('tipo de train .........: ', type(train))
print('train.shape ...........: ', train.shape)
print('Labels ................: ', labels)
print('size Labels ...........: ', len(labels))
print('tipo de X_train........: ', type(X_train))
print('shape X_train..........: ', X_train.shape)
print('tipo de y_train........: ', type(y_train))
print('shape y_train..........: ', y_train.shape)
print('tipo de X_test.........: ', type(X_test))
print('input_dim  = ', input_dim)
print('nb_classes  ', nb_classes)

print('-'*30)


#model = MLP('model_MLP_Keras.hdf5')
#model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
#out = model.predict(im)
#print np.argmax(out)

