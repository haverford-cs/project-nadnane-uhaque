# Library Imports
from PIL import Image
import numpy as np
import os
from keras.utils import np_utils
import shutil
np.set_printoptions(threshold=np.inf)
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
    # Rename training images to correspond with their class
    fnamesI = os.listdir(infect_dir)[:int(num_train/2)]
    fnamesH = os.listdir(health_dir)[:int(num_train/2)]
    print("end index of train data ", int(num_train/2))
    for i in range(int(num_train/2)): 
           filename1 = fnamesI[i]
           src1 = infect_dir + '/' + filename1 
           dst1 = train_dir + '/Infected/' + filename1

           filename2 = fnamesH[i]
           src2 = health_dir + '/' + filename2 
           dst2 = train_dir + '/Healthy/' + filename2
           # rename all the files 
           shutil.copy(src1, dst1) 
           shutil.copy(src2, dst2)

    # PREPARE TEST DATA
    # Rename testing images to correspond with their class
    end = int((num_train/2) + (num_test/2))
    fnamesI = os.listdir(infect_dir)[int(num_train/2) : end]
    fnamesH = os.listdir(health_dir)[int(num_train/2) : end]
    for i in range(int(num_test/2)): 
           filename1 = fnamesI[i]
           src1 = infect_dir + '/' + filename1 
           dst1 = test_dir + '/Infected/' + filename1

           filename2 = fnamesH[i]
           src2 = health_dir + '/' + filename2 
           dst2 = test_dir + '/Healthy/' + filename2
           # rename all the files 
           shutil.copy(src1, dst1) 
           shutil.copy(src2, dst2)


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
        #print(label, total)
        for img_name in img_names:
            path = dir_name + '/' + label + '/' + img_name
            img = Image.open(path)
            img = np.array(img)
            if i == 5513:
              print("the shape of X is: " , X_data.shape)
              print("i is currently: ", i)
            X_data[i] = img
            Y_data[i] = j

            i += 1
        j += 1                   
    print('Loading done.')
    
    print('Transform targets to keras compatible format.')
    Y_data = np_utils.to_categorical(Y_data[:n], num_classes)

    np.save('imgs_'+ typ + '.npy', X_data, Y_data)
    return X_data, Y_data