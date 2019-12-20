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

12/11/19
- Finally fixed pre-processing dimension error!
- reduced dataset to 200 images temporarily to run on my computer (CPU)
- 200 images on Nadine's laptop -> ~ 30-40 mins
- started off using 30 epochs, accuracy isn't great -> about 60%

12/12/19
- Changed dataset size to 8000 to make things more manageable
- Debugging is becoming very difficult since CPU takes long time



# References:
https://pillow.readthedocs.io/en/3.1.x/reference/Image.html

https://towardsdatascience.com/detecting-malaria-using-deep-learning-fd4fdcee1f5a

https://www.kaggle.com/bhanotkaran22/keras-cnn-data-augmentation

https://github.com/EXJUSTICE/Keras-Malaria-Cell-Classification

https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea