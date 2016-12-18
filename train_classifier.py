import numpy as np

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.layers import Flatten, Dense, Activation, Merge

import load_images

def build_model():
    model = Sequential()
    model.add(Convolution2D(10, 3, 3, input_shape=(64, 64, 1)))
    model.add(BatchNormalization())

    model.add(Convolution2D(10, 3, 3, border_mode="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Convolution2D(10, 3, 3, border_mode="same"))
    model.add(BatchNormalization())

    model.add(Convolution2D(10, 3, 3, border_mode="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Convolution2D(10, 3, 3, border_mode="same"))
    model.add(BatchNormalization())

    model.add(Convolution2D(10, 3, 3, border_mode="same"))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(optimizer='adadelta', loss='binary_crossentropy')

    return model

if __name__ == "__main__":
    model = build_model()
    images, target = load_images.load("train")
    model.fit(images, target, nb_epoch=10)
    model.save("car_model.h5")
