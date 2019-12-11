# CS360 Machine Learning Final Project:
# Classifying Malaria Cells


# Progress Log:

12/1/19
- Downloaded dataset
- Scoped out images, realized they are all different sizes


12/3/19
- Read in all dataset images

12/7/19
- Added padding to all images, uniform size is now 400 x 400

12/10/19
- Filled out VPN form, hopefully will have GPU access soon!
- Got stuck on CNN data dimension error involving max pooling

12/11/19
- Decided to change dataset folder organization
- Renamed Parasitic to Infected
- Renamed Uninfected to Healthy
- Scrapped previous code attempt
- Fixed dimensionality error! Something went wrong in pre-processing
- Ran CNN for the first time (yay!)
    - no errors (so far), but laptop couldn't handle the intense power of ML XD
- Investigated SVMs for comparison
- Still need to run CNN on full dataset using GPU to get accuracies + fine tune parameters


# References:
https://pillow.readthedocs.io/en/3.1.x/reference/Image.html

https://towardsdatascience.com/detecting-malaria-using-deep-learning-fd4fdcee1f5a

https://medium.com/gradientcrescent/building-a-malaria-classifier-with-keras-background-implementation-d55c32773afa