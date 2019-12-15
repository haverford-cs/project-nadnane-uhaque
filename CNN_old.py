"""
Using the malaria dataset, create a CNN and make test predictions!
Raw image CNN blueprint code by Adrian Yijie Xu
with modifications by Nadine Adnane & Tanjuma Haque
Date: 12/10/19
"""

# Library Imports
import tensorflow.keras
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, layers, models
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import os

##==========================================================================
#|  GLOBAL VARIABLES
##==========================================================================
# Define directories
base_dir ='malaria/cell_images/'
work_dir =  "work/"
# I = infected;  H = healthy
base_dir_I ='malaria/cell_images/Infected/'
base_dir_H ='malaria/cell_images/Healthy/'

work_dir_I = "work/I/"
work_dir_H = "work/H/"

train_dir = os.path.join(work_dir, 'train')
validation_dir = os.path.join(work_dir, 'validation')
test_dir = os.path.join(work_dir, 'test')

train_pos_dir = os.path.join(train_dir, 'pos')
train_neg_dir = os.path.join(train_dir, 'neg')
validation_pos_dir = os.path.join(validation_dir, 'pos')
validation_neg_dir = os.path.join(validation_dir, 'neg')
test_pos_dir = os.path.join(test_dir, 'pos')
test_neg_dir = os.path.join(test_dir, 'neg')

# Define training, validation, testing split
total_images = 9000
num_train = 3000
num_validation = 1000
num_test = 500
img_dim = 150

##==========================================================================
#|  FUNCTIONS
##==========================================================================

def mk_dirs():
    # Make all needed directories
    os.mkdir(work_dir)
    os.mkdir(work_dir_I)
    os.mkdir(work_dir_H)
    os.mkdir(train_dir)
    os.mkdir(validation_dir)
    os.mkdir(test_dir)
    print("New directories for train, validation, and test created")

    # Make pos/neg directories for train/validation/test
    os.mkdir(train_pos_dir)
    os.mkdir(train_neg_dir)
    os.mkdir(validation_pos_dir)
    os.mkdir(validation_neg_dir)
    os.mkdir(test_pos_dir)
    os.mkdir(test_neg_dir)
    print("Train, Validation, and Test folders made for both Infected (I) and Healthy (H) datasets")
    
    # Rename "Infected" images to correspond with their class
    i = 0
    for filename in os.listdir(base_dir_I): 
           dst ="pos" + str(i) + ".jpg"
           src =base_dir_I + filename 
           dst =work_dir_I + dst 
           # rename all the files 
           shutil.copy(src, dst) 
           i += 1
    # Rename "Healthy" images to correspond with their class
    j = 0  
    for filename in os.listdir(base_dir_H): 
           dst ="neg" + str(j) + ".jpg"
           src =base_dir_H + filename 
           dst =work_dir_H + dst 
           # rename all the files 
           shutil.copy(src, dst) 
           j += 1     

def split_data():
    # Copy images to their proper directories
    # Start with infected "pos" images
    fnames = ['pos{}.jpg'.format(i) for i in range(num_train)]
    for fname in fnames:
        src = os.path.join(work_dir_I, fname)
        dst = os.path.join(train_pos_dir, fname)
        shutil.copyfile(src, dst)

    v = (num_train + num_validation)
    fnames = ['pos{}.jpg'.format(i) for i in range(num_train, v)]
    for fname in fnames:
        src = os.path.join(work_dir_I, fname)
        dst = os.path.join(validation_pos_dir, fname)
        shutil.copyfile(src, dst)

    t = v + num_test
    fnames = ['pos{}.jpg'.format(i) for i in range(v, t)]
    for fname in fnames:
        src = os.path.join(work_dir_I, fname)
        dst = os.path.join(test_pos_dir, fname)
        shutil.copyfile(src, dst)

    # Now work on healthy "neg" images
    fnames = ['neg{}.jpg'.format(i) for i in range(num_train)]
    for fname in fnames:
        src = os.path.join(work_dir_H, fname)
        dst = os.path.join(train_neg_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['neg{}.jpg'.format(i) for i in range(num_train, v)]
    for fname in fnames:
        src = os.path.join(work_dir_H, fname)
        dst = os.path.join(validation_neg_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['neg{}.jpg'.format(i) for i in range(v, t)]
    for fname in fnames:
        src = os.path.join(work_dir_H, fname)
        dst = os.path.join(test_neg_dir, fname)
        shutil.copyfile(src, dst)
    print("Train, validation, and test datasets split and ready for use")

def mk_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(img_dim, img_dim, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-5), metrics=['acc'])
    print("Model created")
    return model

def normalize():
    # Normalizes pixel intensities to be between 0 and 1
    # Converts JPEG content into floating-point tensor maps of each image
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_gen = train_datagen.flow_from_directory(
            train_dir,
            target_size=(img_dim, img_dim),
            batch_size=20,
            class_mode='binary')
    val_gen = test_datagen.flow_from_directory(
            validation_dir,target_size=(img_dim, img_dim),
            batch_size=20,
            class_mode='binary')
    print("Image preprocessing complete")

    return (train_datagen, test_datagen, train_gen, val_gen)

def train(model, train_gen, val_gen):
    history = model.fit_generator(
        train_gen,
        steps_per_epoch=100,
        epochs=30,
        validation_data=val_gen,
        validation_steps=200)
    #model.save('basic_malaria_pos_neg_v1.h5')
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    return (model, acc, epochs, val_acc, loss, val_loss)

def plot_trn_data(epochs, acc, val_acc, loss, val_loss):
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    # Save the figure to a file
    plt.savefig(fname = "Training_vs_Val_Loss.png")

def predict(model):
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
            test_dir,target_size=(img_dim, img_dim),
            batch_size=20,
            class_mode='binary')
    test_gen.reset()    
    pred = model.predict_generator(test_gen,num_validation,verbose=1)

    return (model, test_gen, pred)

def print_data_stats():
    print('total training pos images:', len(os.listdir(train_pos_dir)))
    print('total training neg images:', len(os.listdir(train_neg_dir)))
    print('total validation pos images:', len(os.listdir(validation_pos_dir)))
    print('total validation neg images:', len(os.listdir(validation_neg_dir)))
    print('total test pos images:', len(os.listdir(test_pos_dir)))
    print('total test meg images:', len(os.listdir(test_neg_dir)))
    print("Images have been copied to working directories, renamed to I & H + num")

def main():
    # Create all needed directories
    if(os.path.isdir(work_dir) == False):
        mk_dirs()
        # Split data into training, validation, and testing data
        split_data()
    
    # Sanity check
    print_data_stats()

    # Normalize the data!
    (train_datagen, test_datagen, train_gen, val_gen) = normalize()

    # Create the CNN model!
    model = mk_model()

    # Train the model
    (model, acc, epochs, val_acc, loss, val_loss) = train(model, train_gen, val_gen)
        
    # Plot the training and validation accuracy
    plot_trn_data(epochs, acc, val_acc, loss, val_loss)

    # Use the model on the testing dataset
    (model, test_gen, pred) = predict(model)
    print("Predictions finished!~")

    # Visualize the model
   # plot_model(model, to_file='model.png')

    # Output visual!~
    # for index, probability in enumerate(pred):
    #     image_path = test_dir + "/" + test_gen.filenames[index]
    #     img = mpimg.imread(image_path)
        
    #     #plt.imshow(img)
    #     print(test_gen.filenames[index])
    #     if probability > 0.5:
    #         plt.title("%.2f" % (probability[0]*100) + "% H")
    #     else:
    #         plt.title("%.2f" % ((1-probability[0])*100) + "% I")
    #     #plt.show()
main()
