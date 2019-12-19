"""
Methods adapted from:
https://www.kaggle.com/bhanotkaran22/keras-cnn-data-augmentation
https://github.com/EXJUSTICE/Keras-Malaria-Cell-Classification
https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
Authors: Nadine Adnane + Tanjuma Haque
"""
# Custom imports
from visualize import *

# Library imports
import os
import cv2
from PIL import Image
import numpy as np
np.random.seed(1000)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

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

# Process the images in each folder, get the labels
def process_data():
    # Process the infected images
    infected_images = os.listdir(DATA_DIR + 'Infected/')
    for i, image_name in enumerate(infected_images):
        try:
            if (image_name.split('.')[1] == 'png'):
                image = cv2.imread(DATA_DIR + 'Infected/' + image_name)
                image = Image.fromarray(image, 'RGB')
                image = image.resize((SIZE, SIZE))
                dataset.append(np.array(image))
                label.append(1)
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
                label.append(0)
        except Exception:
            print("Could not read image {} with name {}".format(i, image_name))

# Make a CNN using the given optimizer opt, loss function l, and dropout rate d
# Return the model
def make_model(opt, l, d):
    # Build the CNN model!
    # 2 convolutional layers, flatten, and 3 dense layers!
    # using adam optimizer like in the lab
    # using 3 x 3 filters, pooling 2 x 2
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), input_shape = (SIZE, SIZE, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_last"))
    model.add(BatchNormalization(axis = -1))
    model.add(Dropout(d))

    model.add(Convolution2D(32, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_last"))
    model.add(BatchNormalization(axis = -1))
    model.add(Dropout(d))

    model.add(Flatten())

    model.add(Dense(activation = 'relu', units=512))
    model.add(BatchNormalization(axis = -1))
    model.add(Dropout(d))
    model.add(Dense(activation = 'relu', units=256))
    model.add(BatchNormalization(axis = -1))
    model.add(Dropout(d))
    model.add(Dense(activation = 'sigmoid', units=2))
    model.compile(optimizer = opt, loss = loss, metrics = ['accuracy'])
    return model

# Split the data by the given test split fraction ts
# Return the training and testing data
def split_data(ts):
    # Split the dataset 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(dataset, 
                                       to_categorical(np.array(label)),
                                       test_size = ts, random_state = 0)
    return X_train, X_test, y_train, y_test

# Train the model using X ad y train, # batches, # of epochs, & validation split
# Return model history for plotting
def train_model(model, X_train, y_train, b, e, val_split):
    # Train the model
    history1 = model.fit(np.array(X_train), 
                             y_train, 
                             batch_size = 64, 
                             verbose = 2, 
                             epochs = 50, 
                             validation_split = val_split,
                             shuffle = False)
    return history1

# Plot the accuracy and loss given the history object
# and whether or not this is augmented data ("aug")
# Return nothing since the plot is saved to disk as a png
def plot_acc_loss(history, augmented):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    # Plot and save the accuracy figure
    trn_accl = 'Training acc'
    trn_lossl = 'Training loss'
    tst_accl = 'Testing acc'
    tst_lossl = 'Testing loss'
    title1 = 'Training and Testing accuracy'
    title2 = 'Training and Testing loss'
    if augmented == True:
        trn_accl = 'Augmented ' + trn_accl
        trn_lossl = 'Augmented ' + trn_lossl
        tst_accl = 'Augmented ' + tst_accl
        tst_lossl = 'Augmented ' + tst_lossl
        title1 = 'Augmented ' + title1
        title2 = 'Augmented ' + title2

    plt.plot(epochs, acc, 'bo', label= trn_accl)
    plt.plot(epochs, val_acc, 'r', label= tst_accl)
    plt.xlabel("Epoch", fontsize = 16)
    plt.ylabel("Accuracy", fontsize = 16)
    plt.title(title1)
    plt.legend()
    plt.savefig(title1 + '.png')

    # Plot and save the loss figure
    plt.figure()
    plt.plot(epochs, loss, 'bo', label= trn_lossl)
    plt.plot(epochs, val_loss, 'r', label= tst_lossl)
    plt.xlabel("Epoch", fontsize = 16)
    plt.ylabel("Loss", fontsize = 16)
    plt.title(title2)
    plt.legend()
    plt.savefig(title2 + '.png')

# Augment the data using the data, batch size and # epochs
# Return a new history object and the test data generator
def augment_data(model, X_train, y_train, X_test, y_test, bs, e):
    # Augment the data to try and improve accuracy
    train_generator = ImageDataGenerator(rescale = 1/255,
                                         zoom_range = 0.3,
                                         horizontal_flip = True,
                                         rotation_range = 30)

    test_generator = ImageDataGenerator(rescale = 1/255)

    train_generator = train_generator.flow(np.array(X_train),
                                           y_train,
                                           batch_size = bs,
                                           shuffle = False)

    test_generator = test_generator.flow(np.array(X_test),
                                         y_test,
                                         batch_size = bs,
                                         shuffle = False)


    # Train the model again and test on the augmented data
    history = model.fit_generator(train_generator,
                                       steps_per_epoch = len(X_train)/bs,
                                       epochs = e,
                                       shuffle = False)
    return history, test_generator

# Plot and save the confusion matrix
# Return the predicted and true classes
def plot_cm(model, y_test, test_generator):
    # Generate the confusion matrix
    Y_prediction = model.predict(test_generator)
    # Convert predictions classes to one hot vectors 
    Y_pred_classes = np.argmax(Y_prediction, axis = 1) 
    # Convert Testing observations to one hot vectors
    Y_true = np.argmax(y_test, axis = 1) 
    # compute the confusion matrix
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
    plt.figure(figsize=(10,8))
    # Define confusion matrix quadrant labels
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ['{0:0.0f'.format(value) for value in confusion_mtx.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in confusion_mtx.flatten()/np.sum(confusion_mtx)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(confusion_mtx, annot=True, fmt="d", cbar = True);
    plt.ylabel('True Label', fontsize = 18)
    plt.xlabel('Predicted Label', fontsize = 18)
    plt.title('Confusion Matrix', fontsize = 20)
    plt.savefig("Confusion Matrix.png")

    return Y_pred_classes, Y_true

# Plot and save the ROC curve
def plot_ROC(y_test, Y_prediction):
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

# Output a plot of some correctly and incorrectly classified images
def save_examples(X_test, Y_pred_classes, Y_true):
    # Show some correctly classified images
    correct = np.where(Y_pred_classes==Y_true)[:18]
    print("Found %d correct labels", len(correct))
    for i, correct in enumerate(correct[:9]):
        plt.subplot(3,3,i+1)
        plt.title("Predicted {}, Class {}".format(Y_pred_classes[correct], Y_true[correct]))
        plt.tight_layout()
        plt.savefig("correct.png", X_test[correct])

    # Show hard to classify images
    incorrect = np.where(Y_pred_classes!=Y_true)[:18]
    print("Found %d incorrect labels", len(incorrect))
    for i, incorrect in enumerate(incorrect[:9]):
        plt.subplot(3,3,i+1)
        plt.title("Predicted {}, Class {}".format(Y_pred_classes[incorrect], Y_true[incorrect]))
        plt.tight_layout()
        plt.savefig("incorrect.png", X_test[incorrect])

# Visualize the conv layer filters of our model!
def visualize_filters(model):
    # Visualize the filters
    print("Trying to visualize filters . . .")
    visualize_layer(model, 'conv2d_1')
    visualize_layer(model, 'conv2d_2')

if __name__ == "__main__":
    # Go through the image folders and process the data
    process_data()

    # Build the CNN using the following optimizer, loss function, and dropout rate
    model = make_model('adam','categorical_crossentropy', 0.2)

    # Print a summary of the layers in our CNN
    print(model.summary())

    # Split the data specifying the test split
    X_train, X_test, y_train, y_test = split_data(0.2)

    # Train the model using 64 batches, 50 epochs, and 0.1 validation split
    # Store the training history for plotting
    history1 = train_model(model, X_train, y_train, 64, 50, 0.1)

    # Plot the training and testing accuracy and loss
    # using the history object and 'False' to indicate
    # that this data has not been augmented yet
    plot_acc_loss(history1, False)

    # Calculate and print testing accuracy
    print("Test_Accuracy: {:.2f}%".format(model.evaluate(np.array(X_test), np.array(y_test))[1]*100))


    # Augment the data to try and improve the accuracy
    # Give the model, batch size, and # of epochs
    history2, test_gen = augment_data(model, X_train, y_train, X_test, y_test, 64, 50)

    # Print the new and improved test accuracy after augmentation!
    print("Test_Accuracy(after augmentation): {:.2f}%".format(model.evaluate_generator(test_gen,
                                                                                       steps = len(X_test),
                                                                                       verbose = 1)[1]*100))

    # Plot the new augmented training and testing accuracies
    plot_acc_loss(history2, True)

    # Plot and save the confusion matrix
    plot_cm(model, y_test, test_gen)

    # Plot and save the ROC curve
    Y_pred_classes, Y_true = plot_ROC(y_test, Y_prediction)

    # Output a plot of some correctly and incorrectly classified images
    save_examples(X_test, Y_pred_classes, Y_true)

    # Visualize some of the model filters!
    visualize_filters(model)
    
main()