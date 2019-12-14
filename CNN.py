"""
Using the malaria dataset, create a CNN and make test predictions!
Nadine Adnane & Tanjuma Haque
Date: 12/10/19
"""

# Library Imports
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Activation, Dense, MaxPooling2D, Flatten, Dropout
from sklearn.metrics import accuracy_score

# My Imports
from preprocess import load_data, split_data
from util import resize, plot_ROC, plot_confusion

##==========================================================================
#|  GLOBAL VARIABLES
##==========================================================================
# Image Dimensions
img_x = 64
img_y = 64
channel = 3 #RGB

# Model Properties
num_classes = 2 
batch_size = 1
num_epoch = 60
##==========================================================================
#|  FUNCTIONS
##==========================================================================
def main():
    # Create the CNN
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(img_x,img_y,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Split data
    split_data()

    # Get the resized image data
    X_train, Y_train = load_data('train',img_x, img_y)
    print ("got here")
    X_test, Y_test = load_data('test',img_x, img_y)

    # Print the shape of the train data
    print(X_train.shape, Y_train.shape)

    # Train the model
    hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epoch, verbose=1, 
                     shuffle=True, validation_data=None)

    # Print the shape of the test data
    print(X_test.shape, Y_test.shape)

    # Make predictions
    print('-'*30)
    print('Predicting on the validation data...')
    print('-'*30)
    y_pred = model.predict(X_test, batch_size=batch_size, verbose=1)

    # compute the accuracy
    test_accuracy = accuracy_score(Y_test.argmax(axis=-1),y_pred.argmax(axis=-1))
    print("Test_Accuracy = ", test_accuracy)

    # Compute ROC values/plot curve
    plot_ROC(Y_test, y_pred, num_classes)

    # Compute and plot confusion matrix
    plot_confusion(Y_test, y_pred)
   
main()