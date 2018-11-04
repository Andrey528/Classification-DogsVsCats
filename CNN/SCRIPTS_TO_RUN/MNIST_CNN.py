from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Convolution2D, Dense, Dropout, Flatten
from keras.utils import np_utils
import numpy as np

batch_size = 32
num_epochs = 10
kernel_size = 3
pool_size = 2
conv_depth_1 = 15
conv_depth_2 = 20
conv_depth_3 = 25
drop_prob_1 = 0.25
hidden_size = 100

(X_train, y_train), (X_test, y_test) = mnist.load_data()

height, width, depth = 28, 28, 1
num_classes = np.unique(y_train).shape[0]

X_train = X_train.reshape(X_train.shape[0], depth, height, width)
X_test = X_test.reshape(X_test.shape[0], depth, height, width)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= np.max(X_train)
X_test /= np.max(X_train)

Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)

inp = Input(shape=(depth, height, width))
conv_1 = Convolution2D(conv_depth_1, kernel_size, kernel_size, border_mode='same', activation='relu')(inp)
conv_2 = Convolution2D(conv_depth_1, kernel_size, kernel_size, border_mode='same', activation='relu')(conv_1)
conv_3 = Convolution2D(conv_depth_3, kernel_size, kernel_size, border_mode='same', activation='relu')(conv_2)
drop_1 = Dropout(drop_prob_1)(conv_3)
flat = Flatten()(drop_1)
out = Dense(num_classes, activation='softmax')(flat)

model = Model(input=inp, output=out)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=num_epochs, verbose=1, validation_split=0.1)
model.evaluate(X_test, Y_test, verbose=1)

print("Точность работы загруженной сети на тестовых данных: %.2f%%" % (scores[1]*100))
