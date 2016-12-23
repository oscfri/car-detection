import os
import sys
import random

data_folder = os.environ.get('CAR_DETECTION')
if data_folder is None:
    sys.stderr.write("Environment variable 'CAR_DETECTION' is not set\n")
    sys.exit()

def write(train_file, test_file, string):
    if random.random() > 0.2:
        train_file.write(string + "\n")
    else:
        test_file.write(string + "\n")

if __name__ == "__main__":
    train_file = open("train", "w")
    test_file = open("test", "w")

    non_vehicles_folder = os.path.join(data_folder, 'non-vehicles')
    vehicles_folder = os.path.join(data_folder, 'vehicles')
    subfolders = [
        'non-vehicles/Far',
        'non-vehicles/Left',
        'non-vehicles/MiddleClose',
        'non-vehicles/Right',
        'non-vehicles/Set1Part1',
        'vehicles/Far',
        'vehicles/Left',
        'vehicles/MiddleClose',
        'vehicles/Right',
        'vehicles/cars128x128',
    ]

    for subfolder in subfolders:
        for image in os.listdir(os.path.join(data_folder, subfolder)):
            write(train_file, test_file, '%s/%s' % (subfolder, image))

    train_file.close()
    test_file.close()
