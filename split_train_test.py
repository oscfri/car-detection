import os
import sys
import random

data_folder = os.environ.get('CAR_DETECTION')
if data_folder is None:
    sys.stderr.write("Environment variable 'CAR_DETECTION' is not set\n")
    sys.exit()

def write(train_file, test_file, string):
    if random.random() > 0.5:
        train_file.write(string + "\n")
    else:
        test_file.write(string + "\n")

if __name__ == "__main__":
    train_file = open("train", "w")
    test_file = open("test", "w")

    non_vehicles_folder = os.path.join(data_folder, 'non-vehicles')
    vehicles_folder = os.path.join(data_folder, 'vehicles')
    orientations = [
        'Far',
        'Left',
        'MiddleClose',
        'Right',
    ]

    for orientation in orientations:
        non_vehicles_subfolder = os.path.join(non_vehicles_folder, orientation)
        vehicles_subfolder = os.path.join(vehicles_folder, orientation)
        for image in os.listdir(non_vehicles_subfolder):
            write(train_file, test_file, 'non-vehicles/%s/%s' % (orientation, image))
        for image in os.listdir(vehicles_subfolder):
            write(train_file, test_file, 'vehicles/%s/%s' % (orientation, image))

    train_file.close()
    test_file.close()
