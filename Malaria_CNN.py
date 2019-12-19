"""
Methods adapted from:
https://www.kaggle.com/bhanotkaran22/keras-cnn-data-augmentation
https://github.com/EXJUSTICE/Keras-Malaria-Cell-Classification
Authors: Nadine Adnane + Tanjuma Haque
"""

import os
import cv2
from PIL import Image
import numpy as np
np.random.seed(1000)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from visualize import *

import keras
from keras import optimizers
from keras.models import Sequential
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout

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
print(model.summary())

# Split the dataset 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(dataset, to_categorical(np.array(label)), test_size = 0.20, random_state = 0)

# Train the model
history1 = model.fit(np.array(X_train), 
                         y_train, 
                         batch_size = 64, 
                         verbose = 2, 
                         epochs = 50, 
                         validation_split = 0.1,
                         shuffle = False)

# Plot the accuracy and loss
acc = history1.history['accuracy']
val_acc = history1.history['val_accuracy']
loss = history1.history['loss']
val_loss = history1.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'o', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Testing acc')
plt.xlabel("Epoch", fontsize = 16)
plt.ylabel("Accuracy", fontsize = 16)
title = 'Training and Testing accuracy'
plt.title(title)
plt.legend()
plt.savefig(title + '.png')

plt.figure()
plt.plot(epochs, loss, 'o', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Testing loss')
plt.xlabel("Epoch", fontsize = 16)
plt.ylabel("Loss", fontsize = 16)
title = 'Training and Testing loss'
plt.title(title)
plt.legend()
plt.savefig(title + '.png')

# Calculate and store testing accuracy
print("Test_Accuracy: {:.2f}%".format(model.evaluate(np.array(X_test), np.array(y_test))[1]*100))


# Augment the data to try and improve accuracy
train_generator = ImageDataGenerator(rescale = 1/255,
                                     zoom_range = 0.3,
                                     horizontal_flip = True,
                                     rotation_range = 30)

test_generator = ImageDataGenerator(rescale = 1/255)

train_generator = train_generator.flow(np.array(X_train),
                                       y_train,
                                       batch_size = 64,
                                       shuffle = False)

test_generator = test_generator.flow(np.array(X_test),
                                     y_test,
                                     batch_size = 64,
                                     shuffle = False)


# Train the model again and test on the augmented data
history2 = model.fit_generator(train_generator,
                                   steps_per_epoch = len(X_train)/64,
                                   epochs = 50,
                                   shuffle = False)

print("Test_Accuracy(after augmentation): {:.2f}%".format(model.evaluate_generator(test_generator, steps = len(X_test), verbose = 1)[1]*100))

# Check the model layers
print("model.layers = ", model.layers)
print("number of layers is: ", len(model.layers))
# Visualize the filters
# the name of the layer we want to visualize
print("Trying to visualize filters . . .")
for layer in model.layers
    # Visualize the current layer
    visualize_layer(model, layer)

# Plot the training vs testing accuracy and loss
aug_acc = history2.history['accuracy']
aug_loss = history2.history['loss']
epochs = range(1, len(aug_acc) + 1)


plt.figure()
plt.plot(epochs, acc, 'b', label='Original acc')
plt.plot(epochs, aug_acc, 'g', label='Augmented acc')
plt.xlabel("Epoch", fontsize = 16)
plt.ylabel("Accuracy", fontsize = 16)
title = 'Original vs Augmented Training Accuracy'
plt.title(title)
plt.legend()
plt.savefig(title + '.png')

plt.figure()
plt.plot(epochs, loss, 'b', label='Original loss')
plt.plot(epochs, aug_loss, 'g', label='Augmented loss')
plt.xlabel("Epoch", fontsize = 16)
plt.ylabel("Loss", fontsize = 16)
title = 'Original vs Augmented Training loss'
plt.title(title)
plt.legend()
plt.savefig(title + '.png')

# Generate the confusion matrix

Y_prediction = model.predict(test_generator)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_prediction, axis = 1) 
# Convert Testing observations to one hot vectors
Y_true = np.argmax(y_test, axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
plt.figure(figsize=(10,8))
sns.heatmap(confusion_mtx, 
            annot=True, 
            annot_kws = {'size':16, "ha": 'center', "va": 'center'},
            fmt="d", cbar = True);
plt.ylabel('True Label', fontsize = 18)
plt.xlabel('Predicted Label', fontsize = 18)
plt.title('Confusion Matrix', fontsize = 20)
plt.savefig("Confusion Matrix.png")

# Plot ROC curve
#compute the ROC-AUC values
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], Y_prediction[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), Y_prediction.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

#Plot ROC curve for the positive class
plt.figure(figsize=(20,10), dpi=300)
plt.plot(fpr[1], tpr[1], color='red',
         lw=2, label='ROC curve (area = %0.4f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize = 16)
plt.ylabel('True Positive Rate', fontsize = 16)
plt.title('ROC Curve', fontsize = 20)
plt.legend(loc="lower right", fontsize = 16)
plt.savefig("ROC Curve.png")

# Show some correctly classified images
correct = np.where(Y_pred_classes==Y_true)[0]
print("Found %d correct labels", len(correct))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.title("Predicted {}, Class {}".format(Y_pred_classes[correct], Y_true[correct]))
    plt.tight_layout()
    plt.imsave("correct.png", X_test[correct])

# Show hard to classify images
incorrect = np.where(Y_pred_classes!=Y_true)[0]
print("Found %d incorrect labels", len(incorrect))
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3,3,i+1)
    plt.title("Predicted {}, Class {}".format(Y_pred_classes[incorrect], Y_true[incorrect]))
    plt.tight_layout()
    plt.imsave("incorrect.png", X_test[incorrect])
