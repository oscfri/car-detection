# Car Detection

## Training the classifier

### Download the dataset
Download the dataset https://www.gti.ssr.upm.es/data/Vehicle_database.html to a folder preferably outside the git repository. The location of the downloaded dataset should be set to the environment variable CAR_DETECTION. This is done by adding the below line to `~/.virtualenvs/cv/bin/postactivate` (provided that you're working on a virtual environment named 'cv').

```
export CAR_DETECTION=/path/to/car-detection-dataset/
```

### Split train and test

Execute

```
python split_train_test.py
```

to generate files `train` and `test`, which lists which input images shall be used for training and testing respectively.

### Train the classifier

Exectue

```
python train_classifier.py
```

to train the classifer on the images in the training set. This will generate the file `car_model.h5`.

### Test the classifier

Execute

```
python test_classifier.py
```

to test the classifier. This will load the trained model `car_model.h5` and test it on the images in the testing set.
