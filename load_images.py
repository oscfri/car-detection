import os

import xml.etree.ElementTree as ET
import numpy as np
import math
from numpy.random import random_integers

import cv2

data_folder = os.environ.get('CAR_DETECTION')
if data_folder is None:
    sys.stderr.write("Environment variable 'CAR_DETECTION' is not set\n")
    sys.exit()

annotations_folder = os.path.join(data_folder, 'Annotations/Anno_XML')
images_folder = os.path.join(data_folder, 'Original')

def load_image_windows(image_name):
    full_path = os.path.join(data_folder, image_name)
    image = cv2.imread(full_path) / 255.0
    pts2 = np.float32([[0,0],[63,0],[0,63],[63,63]])
    if image.shape[0] > 200 or image.shape[1] > 200:
        for w in random_integers(64, 640, 5):
            x = random_integers(0, image.shape[0] - w)
            y = random_integers(0, image.shape[1] - w)
            pts1 = np.float32([[x,y],[x + w,y],[x,y + w],[x + w,y + w]])
            M = cv2.getPerspectiveTransform(pts1,pts2)
            dst = cv2.warpPerspective(image,M,(64,64))
            yield dst
    else:
        scaled_image = cv2.resize(image, (64, 64))
        yield scaled_image

def load_image_window(image, x, y, w):
    pts1 = np.float32([[x, y], [x + w, y], [x, y + w], [x + w, y + w]])
    pts2 = np.float32([[0,0],[63,0],[0,63],[63,63]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, M, (64, 64))

def load_image(filename):
    full_path = os.path.join(images_folder, filename)
    return cv2.imread(full_path) / 255.0

def overlapping(x1, y1, w1, windows):
    for x2, x2, w2 in windows:
        if x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + w2 and y1 + w1 > y2:
            return True
    return False

def load_from_annotation(filename):
    full_path = os.path.join(annotations_folder, filename)
    tree = ET.parse(full_path)
    root = tree.getroot()
    image = load_image(root.find('filename').text.strip())
    windows = []
    for obj in root.findall('object'):
        min_x = float('inf')
        min_y = float('inf')
        max_x = 0
        max_y = 0
        name = obj.find('name')
        if name != None and name.text == 'car':
            polygon = obj.find('polygon')
            for pt in polygon.find('pt'):
                x = int(pt.find('x').text)
                y = int(pt.find('y').text)
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
        w = max(max_x - min_x, max_y - min_y)
        windows.append((min_x, min_y, w))
        yield load_image_window(image, min_x, min_y, w), 1
    n_false = 0
    trials_left = 100
    while n_false < 5: 
        trials_left -= 1
        if trials_left < 0:
            break
        w = random_integers(64, 640)
        x = random_integers(0, image.shape[0] - w)
        y = random_integers(0, image.shape[1] - w)
        if not overlapping(x, y, w, windows):
            n_false += 1
            yield load_image_window(image, x, y, w), 0

def load2(origin):
    train_files = open(origin, "r")
    images = []
    targets = []
    weights = []
    for annotation in train_files:
        annotation = annotation.strip()
        for image, target, weight in load_from_annotation(annotation):
            images.append(image)
            targets.append(target)
            weights.append(weight)
    images_formatted = np.stack(images, axis=0)
    target_formatted = np.asarray(targets)
    weight_formatted = np.asarray(weights)

    train_files.close()

    return images_formatted, target_formatted, weight_formatted

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
