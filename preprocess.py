# Library Imports
import cv2
import numpy as np
import os
from keras.utils import np_utils
import shutil
###############################################################################
# Image Directories
train_dir = 'malaria/cell_images/train'
test_dir = 'malaria/cell_images/test'
infect_dir = 'malaria/cell_images/Infected'
health_dir = 'malaria/cell_images/Healthy'
###############################################################################
# Define train/test split and image sizes
num_train = 22045 # 80% of total data
num_test = 5513   # 20% of total data
num_classes = 2

def split_data():
    # PREPARE TRAIN DATA
    # Rename "Infected" images to correspond with their class
    i = 0
    fnames = os.listdir(infect_dir)[:int(num_train/2)]
    for filename in fnames: 
           src = infect_dir + '/' + filename 
           print("filename is: ", filename)
           dst = train_dir + '/Infected/' + filename
           # rename all the files 
           shutil.copy(src, dst) 
           i += 1
    # Rename "Healthy" images to correspond with their class
    j = 0  
    fnames = os.listdir(health_dir)[:int(num_train/2)]
    for filename in fnames: 
           src = health_dir + '/' + filename 
           dst = test_dir + '/Healthy/' + filename
           # rename all the files 
           shutil.copy(src, dst) 
           j += 1

    # PREPARE TEST DATA
      # Rename "Infected" images to correspond with their class
    k = 0
    end = int((num_train/2) + (num_test/2))
    fnames = os.listdir(infect_dir)[int(num_train/2) : end]
    for filename in fnames: 
           dst = "infected" + str(i) + ".png"
           src = infect_dir + filename 
           dst = train_dir + 'Infected' + dst 
           # rename all the files 
           shutil.copy(src, dst) 
           i += 1
    # Rename "Healthy" images to correspond with their class
    j = 0  
    fnames = os.listdir(health_dir)[int(num_train/2) : end]
    for filename in os.listdir(health_dir): 
           dst ="healthy" + str(j) + ".png"
           src = health_dir + filename 
           dst = test_dir + 'Healthy' + dst 
           # rename all the files 
           shutil.copy(src, dst) 
           j += 1


def load_data(typ, img_x, img_y):
    # Load training images
    if typ == 'train':
        dir_name = train_dir
        n = num_train
    elif typ == 'test':
        dir_name = test_dir
        n = num_test
    
    X_data = np.ndarray((n, img_x, img_y, 3), dtype=np.uint8)
    Y_data = np.zeros((n,), dtype='uint8')
    labels = os.listdir(dir_name)
    total = len(labels)

    i = 0
    print('-'*30)
    print('Creating images...')
    print('-'*30)
    j = 0
    for label in labels:
        img_names = os.listdir(os.path.join(dir_name, label))
        total = len(img_names)
        print(label, total)
        for img_name in img_names:
            img = cv2.imread(os.path.join(train_dir, label, img_name), cv2.IMREAD_COLOR)
            img = np.array([img])
            X_data[i] = img
            Y_data[i] = j

            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, total))
            i += 1
        j += 1    
    print(i)                
    print('Loading done.')
    
    print('Transform targets to keras compatible format.')
    Y_data = np_utils.to_categorical(Y_data[:n], num_classes)

    np.save('imgs_'+ typ + '.npy', X_data, Y_data)
    return X_data, Y_data