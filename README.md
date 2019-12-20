# CS360 Final Project - Malaria CNN Classification

# Description:
For our final project, we will be creating a convolutional neural network to classify images of malaria cells. The dataset we are working with was provided by the NIH, and was downloaded from Kaggle. 

The dataset can be found at the following link:
https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria

or on the NIH official website:
https://ceb.nlm.nih.gov/repositories/malaria-datasets/

After reading the desciption of the dataset on the NIH website and doing some additional research, we learned that the NIH included a set of custom models that can be run on the data to achieve an accuracy of about 95%, though we were unable to run the models ourselves. We briefly looked at these custom models and began to wonder if we could match the accuracy of 95% by building a simple CNN model much like the one we created in the last lab.

# Inspiration:
Nadine is a pre-med computer science major, and Tanjuma is hoping to study computational biology in the future, so we were both very interested in working with a biological or health related dataset. We were also very inspired by Sara's biological machine learning work, and were hoping to see how we could apply the machine learning techniques we've learned so far to a real dataset to help solve a realworld problem. We also wanted to take this project as an opportunity to challenge ourselves! We wanted to take another shot at using Tensorflow considering the difficulties we had with the last lab, and to enrich our understanding of convolutional neural networks.


# Instructions:
To run our project, navigate to the root of the project folder (project-nadnane-uhaque), and run the following command in your terminal:
python Malaria_CNN.py

If you'd like to leave the model running in the background:
nohup python Malaria_CNN.py &

For Bryn Mawr students without VPN access to the haverford lab computers, you can SSH into one of the regular haverford lab computers, and then from there SSH into one of the GPU machines. Then download the project as you normally would using git clone, and run as described above.

# Notes:
- The SVM and grid search files were a work in progress and do not currently run.

# Progress Log:

12/1/19
- Downloaded dataset
- Read some background information on Malaria and the background of the dataset
    -Turns out the patients were from Bangladesh, Tanjuma's home country! :)
- Scoped out images, realized they are all different sizes

12/3/19
- Read in all dataset images
- Running into some weird dimension errors

12/7/19
- Working on padding to all images, uniform size is now 400 x 400
- Still can't access GPU machines off campus

12/8/19
- Reading about difference between padding and resizing, pros and cons
- Still getting some errors with padding function
- Takes a long time to pad all images, starting to consider using a subset of the data
- Researching different CNN models built for this dataset
- seems like using 2 conv layers is the standard?

12/9/19
- Added git ignore to avoid uploading giant dataset
    - Tanjuma's laptop can't handle the full dataset
- Trying to use cv2 to preprocess
- Getting error "cannot find module cv2 when using OpenCV"
    - even though it is installed on my laptop
- Tried running over ssh, still same error
    - permission denied to install package
    - looking for a work around

12/10/19
- Tanjuma's laptop won't turn on anymore ;-(

12/11/19
- Finally fixed pre-processing dimension error!
- reduced dataset to 200 images temporarily to run on my computer (CPU)
- 200 images on Nadine's laptop -> ~ 30-40 mins
- started off using 30 epochs, accuracy isn't great -> about 60%
- started working on SVM code for comparison
- Decided to use dropout and batch normalization to improve results

12/12/19
- Changed dataset size to 8000 to make things more manageable
- Debugging is becoming very difficult since CPU takes long time

12/13/19
- Changed preprocessing methods; resizing images to 64 x 64
    - manually resized a set of images from both folders, quality is less but important features still very distinct

12/14/19
- Adjusted model params
- hopefully fixed loss and accuracy plot function, need to rerun
- finally fixed data splitting out of bounds error
- thinking about trying to implement KNN for comparison

12/15/19
- Finally able to run CNN on GPU through double SSHing! :D
- decided we will use the full dataset
- added confusion matrix function using heatmap!
    - running into errors though
- CNN accuracy is looking great - 95%!

12/16/19
- lots of debugging
- added ROC curve function
- decided to try data augmentation to improve accuracy!

12/17/19
- adjusted plot fonts/colors
- started looking into filter visualization

12/18/19
- Finally fixed buggy plot outputs
- fixing some silly bugs, re-ran CNN several times
- trying to fix filter visualization ;-(

12/19/19
- cleaned up and reorganized code, added better comments
- swapped labels to be more intuitive
- ran out of time to fix alignment of confusion matrix labels + rerun
- Finally got to visualize filters!!
- First layer seems to be learning colors
- Second layer shows some more detailed lines/edges
- Hope to visualize more in the future for fun!

# Challenges:
We had a lot of fun working on this project and applying what we've learned in class to a real, meaningful dataset. The biggest challenges we faced were during the preprocessing phase, and with debugging due to the long runtimes. We also encountered soem permission denial errors when trying to install necessary packages over SSH, but we eventually discovered that this could be fixed using pip install --user (insert package name here). We also ran into some memory errors even on the GPU computers during the pre-processing phase, but this is likely because we were originally processing the data manually in the code. What I mean by this is that we were resizing the physical images on the disk, instead of just reading in the image data as-is and then resizing the dimensions in the code from there, leaving the images on disk unmanipulated. Once we changed our pre-processing approach, our project ran much faster and didn't use as much memory.


# References:
https://pillow.readthedocs.io/en/3.1.x/reference/Image.html

https://towardsdatascience.com/detecting-malaria-using-deep-learning-fd4fdcee1f5a

https://www.kaggle.com/bhanotkaran22/keras-cnn-data-augmentation

https://github.com/EXJUSTICE/Keras-Malaria-Cell-Classification

https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea

https://lhncbc.nlm.nih.gov/publication/pub9932

https://machinelearningmastery.com/image-augmentation-deep-learning-keras/

https://rpubs.com/Sharon_1684/454441
