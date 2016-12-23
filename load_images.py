import os

import numpy as np

import cv2

data_folder = os.environ.get('CAR_DETECTION')
if data_folder is None:
    sys.stderr.write("Environment variable 'CAR_DETECTION' is not set\n")
    sys.exit()

def load_image(image_name):
    full_path = os.path.join(data_folder, image_name)
    image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE) / 255.0
    scaled_image = cv2.resize(image, (64, 64))
    return scaled_image[:, :, np.newaxis]

def load(origin):
    train_files = open(origin, "r")
    non_vehicles_folder = os.path.join(data_folder, 'non-vehicles')
    vehicles_folder = os.path.join(data_folder, 'vehicles')

    images = []
    target = []
    for image_name in train_files:
        image_name = image_name.strip()
        image = load_image(image_name)
        images.append(load_image(image_name))
        if "non-vehicles" in image_name:
            target.append(0)
        else:
            target.append(1)

    images_formatted = np.stack(images, axis=0)
    target_formatted = np.asarray(target)

    train_files.close()

    return images_formatted, target_formatted
