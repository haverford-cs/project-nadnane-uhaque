"""
Methods adapted from:
https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
with modifications by Nadine Adnane + Tanjuma Haque
"""
import os
import cv2
from PIL import Image
import numpy as np
np.random.seed(1000)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

import keras
from keras import optimizers
from keras.models import Sequential
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

# Define dataset directory and other properties
num_classes = 2
DATA_DIR = 'malaria/cell_images/'
SIZE = 64
dataset = []
label = []

# Process the infected images
infected_images = os.listdir(DATA_DIR + 'Infected/')
for i, image_name in enumerate(infected_images):
    try:
        if (image_name.split('.')[1] == 'png'):
            image = cv2.imread(DATA_DIR + 'Infected/' + image_name)
            image = Image.fromarray(image, 'RGB')
            image = image.resize((SIZE, SIZE))
            dataset.append(np.array(image))
            label.append(0)
    except Exception:
        print("Could not read image {} with name {}".format(i, image_name))

# Process the healthy images
healthy_images = os.listdir(DATA_DIR + 'Healthy/')
for i, image_name in enumerate(healthy_images):
    try:
        if (image_name.split('.')[1] == 'png'):
            image = cv2.imread(DATA_DIR + 'Healthy/' + image_name)
            image = Image.fromarray(image, 'RGB')
            image = image.resize((SIZE, SIZE))
            dataset.append(np.array(image))
            label.append(1)
    except Exception:
        print("Could not read image {} with name {}".format(i, image_name))

# Build the CNN!
def create_model():
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), input_shape = (SIZE, SIZE, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_last"))
    model.add(BatchNormalization(axis = -1))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_last"))
    model.add(BatchNormalization(axis = -1))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(activation = 'relu', units=512))
    model.add(BatchNormalization(axis = -1))
    model.add(Dropout(0.2))
    model.add(Dense(activation = 'relu', units=256))
    model.add(BatchNormalization(axis = -1))
    model.add(Dropout(0.2))
    model.add(Dense(activation = 'sigmoid', units=2))
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

# Split the dataset 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(dataset, to_categorical(np.array(label)), test_size = 0.20, random_state = 0)


# define the grid search parameters
model = KerasClassifier(build_fn=create_model, verbose=0)
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid = grid.fit(np.array(X_train), (y_train))
# summarize results
# Store and report the training score
print("Best params:", grid.best_params_)
train_score = grid.score(np.array(X_train), np.array(y_train))
train_acc_scores[i] = train_score
print("Training Score: ", train_score, "\n")

# Report the testing score
test_acc = grid.score(np.array(X_test), np.array(y_test))
print("Test Score/Acc:", test_acc)
