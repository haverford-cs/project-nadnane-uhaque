"""
NN training and testing.
Base code by Gracelyn Shi with modifications
by Nadine Adnane & Tanjuma Haque
Date: 12/5/19
"""

# Library Imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from imutils import paths
from PIL import Image, ImageOps

import os
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt

########################################################
# GLOBAL VARIABLES
# Set path to original dataset images
dataset_path = "malaria/cell_images"

# Initialize base path for training and testing data
base_path = "malaria"

# Define training, validation, and testing directories
train_path = os.path.sep.join([base_path, "training"])
val_path = os.path.sep.join([base_path, "validation"])
test_path = os.path.sep.join([base_path, "testing"])
 
# Define amount of data used for training
train_split = 0.8
 
# Define amount of validation data (percent of the training data)
val_split = 0.1

# Define the batch size
bs = 32

# initialize validation/testing data augmentation object
valAug = ImageDataGenerator(rescale=1 / 255.0)
########################################################

def split_data(image_paths):
    # Split the data into testing and training 
    i = int(len(image_paths) * train_split)
    trainPaths = image_paths[:i]
    testPaths = image_paths[i:]

    # set aside some of the training data for validation data 
    i = int(len(trainPaths) * val_split)
    valPaths = trainPaths[:i]
    trainPaths = trainPaths[i:]

    # define the training/validation/testing datasets 
    datasets = [
        ("training", trainPaths, train_path),
        ("validation", valPaths, val_path),
        ("testing", testPaths, test_path)
        ]

    # loop over the datasets
    for (dType, image_paths, baseOutput) in datasets:
        # show which data split we are creating
        print("[INFO] building '{}' split".format(dType))
     
        # if the output base output directory does not exist, create it
        if not os.path.exists(baseOutput):
            print("[INFO] 'creating {}' directory".format(baseOutput))
            os.makedirs(baseOutput)
     
        # loop over the input image paths
        for inputPath in image_paths:
            # extract the filename of the input image and its class label
            filename = inputPath.split(os.path.sep)[-1]
            label = inputPath.split(os.path.sep)[-2]
     
            # build the path to the label directory
            labelPath = os.path.sep.join([baseOutput, label])
     
            # if the label output directory does not exist, create it
            if not os.path.exists(labelPath):
                print("[INFO] 'creating {}' directory".format(labelPath))
                os.makedirs(labelPath)
     
            # construct the path to the destination image and then copy
            # the image itself
            p = os.path.sep.join([labelPath, filename])
            shutil.copy2(inputPath, p)

def mk_train():
    # initialize the training data augmentation object
    # randomly shifts, translats, and flips each training sample
    trainAug = ImageDataGenerator(
        rescale=1 / 255.0,
        rotation_range=20,
        zoom_range=0.05,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest")
    # initialize the training generator
    train_dset = trainAug.flow_from_directory(
        train_path,
        class_mode="categorical",
        target_size=(64, 64),
        color_mode="rgb",
        shuffle=True,
        batch_size=bs)
    return train_dset

def mk_val():
    # initialize the validation generator
    val_dset = valAug.flow_from_directory(
        val_path,
        class_mode="categorical",
        target_size=(64, 64),
        color_mode="rgb",
        shuffle=False,
        batch_size=bs)
    return val_dset

def mk_test():
    # initialize the testing generator
    test_dset = valAug.flow_from_directory(
        test_path,
        class_mode="categorical",
        target_size=(64, 64),
        color_mode="rgb",
        shuffle=False,
        batch_size=bs)
    return test_dset

def main():
    # # Shuffle the dataset
    # image_paths = list(paths.list_images(dataset_path))
    # random.seed(42)
    # random.shuffle(image_paths)
    # # Split the dataset images into folders
    # split_data(image_paths)

    # Prepare the training, validation, and testing data
    train_dset = mk_train()
    val_dset = mk_val()
    test_dset = mk_test()

    i = 0
    for inputs_batch, labels_batch in train_dset:
        features_batch = conv_base.predict(inputs_batch)
        train_features[i * batch_size : (i + 1) * batch_size] = features_batch
        train_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= nTrain:
            break          



main()