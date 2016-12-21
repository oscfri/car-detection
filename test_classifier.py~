from keras.models import load_model

import load_images

if __name__ == "__main__":
    model = load_model("car_model.h5")
    images, target = load_images.load("test")
    print model.evaluate(images, target)
