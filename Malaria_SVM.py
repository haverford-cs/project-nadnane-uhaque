"""
Using the malaria dataset, create an SVM and make test predictions!
by Nadine Adnane & Tanjuma Haque
"""

# Library Imports
import os
import cv2
from PIL import Image
import numpy as np
np.random.seed(1000)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

import keras
from keras import optimizers
from keras.models import Sequential
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

##==========================================================================
#|  GLOBAL VARIABLES
num_classes = 2
DATA_DIR = 'malaria/cell_images/'
SIZE = 64
dataset = []
label = []
##==========================================================================

##==========================================================================
#|  FUNCTIONS
# Define dataset directory and other properties
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


dataset = np.asarray(dataset)
print("PRE:", dataset.shape)
dataset = dataset.reshape(27558,12288)
print("POST:", dataset.shape)
# Split the dataset 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(dataset, (np.array(label)), test_size = 0.20, random_state = 0)

#constructs SVM
svm_clf = svm.SVC()
svm_clf.fit(X_train, y_train)
print("---")
print("SVM")
print("---")
#gets the train and test accuracies for this learner
print("Test_Accuracy: {:.2f}%".format(svm_clf.evaluate((X_test), (y_test))[1]*100))
##==========================================================================




