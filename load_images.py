import os

import numpy as np
from numpy.random import random_integers

import cv2

data_folder = os.environ.get('CAR_DETECTION')
if data_folder is None:
    sys.stderr.write("Environment variable 'CAR_DETECTION' is not set\n")
    sys.exit()

def load_image_windows(image_name):
    full_path = os.path.join(data_folder, image_name)
    image = cv2.imread(full_path) / 255.0
    pts2 = np.float32([[0,0],[63,0],[0,63],[63,63]])
    if image.shape[0] > 200 or image.shape[1] > 200:
        for w in random_integers(64, 640, 50):
            x = random_integers(0, image.shape[0] - w)
            y = random_integers(0, image.shape[1] - w)
            pts1 = np.float32([[x,y],[x + w,y],[x,y + w],[x + w,y + w]])
            M = cv2.getPerspectiveTransform(pts1,pts2)
            dst = cv2.warpPerspective(image,M,(64,64))
            yield dst
    else:
        scaled_image = cv2.resize(image, (64, 64))
        yield scaled_image

def load(origin):
    train_files = open(origin, "r")
    non_vehicles_folder = os.path.join(data_folder, 'non-vehicles')
    vehicles_folder = os.path.join(data_folder, 'vehicles')

    images = []
    target = []
    for image_name in train_files:
        image_name = image_name.strip()
        for image in load_image_windows(image_name):
            images.append(image)
            if "non-vehicles" in image_name:
                target.append(0)
            else:
                target.append(1)

    images_formatted = np.stack(images, axis=0)
    target_formatted = np.asarray(target)

    train_files.close()

    return images_formatted, target_formatted
