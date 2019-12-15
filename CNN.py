"""
Using the malaria dataset, create a CNN and make test predictions!
Nadine Adnane & Tanjuma Haque
Date: 12/10/19
"""

# Library Imports
import os
import sys
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.preprocessing import image
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from IPython.display import SVG
from keras.layers import Dense, Conv2D, MaxPooling2D,Dropout,BatchNormalization,Flatten

infect_dir = 'malaria/cell_images/Infected/'
health_dir = 'malaria/cell_images/Healthy/'
infected = os.listdir(infect_dir)
healthy = os.listdir(health_dir)
img_dim = 150


def CNN_Model():
    model = Sequential()
    model.add(Conv2D(filters = 4, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'a11', input_shape = (img_dim, img_dim, 3)))  
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'a12'))
    model.add(BatchNormalization(name = 'a13'))
    #input = (128,128,4)
    model.add(Conv2D(filters = 8, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'a21'))   
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'a22'))
    model.add(BatchNormalization(name = 'a23'))
    #input = (64,64,8)
    model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'a31'))   
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'a32'))
    model.add(BatchNormalization(name = 'a33'))
    #input = (32,32,16)
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu', name = 'fc1'))
    model.add(Dense(1, activation = 'sigmoid', name = 'prediction'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def main():
    data = []
    label = []
    for i in infected:
        try:
            img =  image.load_img(infect_dir + i, target_size=(img_dim, img_dim))
            x = image.img_to_array(img)
            data.append(x)
            label.append(1)
        except:
            print("Can't add "+ i +" in the dataset")
    for h in healthy:
        try:
            img =  image.load_img(health_dir + h, target_size=(img_dim, img_dim))
            x = image.img_to_array(img)
            data.append(x)
            label.append(0)
        except:
            print("Can't add "+ h +" in the dataset")  

    data = np.array(data)
    label = np.array(label)

    print(sys.getsizeof(data))
    print(data.shape)

    # Shuffle and split the data!
    data = data/255
    x_train, x_test, y_train, y_test = train_test_split(data,label,test_size = 0.1,random_state=0)

    # Make the model and train it!
    model = CNN_Model()
    model.summary()

    # Fit the model and get the output
    output = model.fit(x_train, y_train, epochs=1, batch_size=1000)

    # Make predictions using the model
    preds = model.evaluate(x = x_test, y = y_test)
    print("Test Accuracy : %.2f%%" % (preds[1]*100))

    # Save the model diagram
    model_pic = plot_model(model, to_file='model.png')
    model_to_dot(model).create(prog='dot', format='png')
    
main()
