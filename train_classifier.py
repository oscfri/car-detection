import numpy as np

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.layers import Flatten, Dense, Activation, Merge
from keras.preprocessing.image import ImageDataGenerator

import load_images

def build_model():
    model = Sequential()
    model.add(Convolution2D(3, 3, 3, input_shape=(64, 64, 1)))
    model.add(BatchNormalization())
    model.add(Convolution2D(3, 3, 3))
    model.add(BatchNormalization())

    model.add(MaxPooling2D())

    model.add(Convolution2D(3, 3, 3))
    model.add(BatchNormalization())
    model.add(Convolution2D(3, 3, 3))
    model.add(BatchNormalization())

    model.add(MaxPooling2D())
    model.add(Convolution2D(3, 3, 3))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(optimizer='sgd', loss='binary_crossentropy')

    return model

if __name__ == "__main__":
    model = build_model()
    images, target = load_images.load("train")
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 rotation_range=10,
                                 shear_range=1,
                                 zoom_range=0.2)
    datagen.fit(images)
    model.fit_generator(datagen.flow(images, target),
                        samples_per_epoch=len(images),
                        nb_epoch=20)
    model.save("car_model.h5")
