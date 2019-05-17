# RandomForest

Classifying the [MNIST handwritten digits](http://yann.lecun.com/exdb/mnist/) with a random forest in C++.

Default parameters:
- 500 decision trees
- 1 000 random training images per tree
- 28 random pixels per tree (root of the number of pixels per image)

This results in an average accuracy of __91%__. Training with all 60 000 images takes around 180s, classifying takes about 0.16s per image.

The pixels are clamped to 0 (where the original color is 0) and 1 (where the original color is above 0).

----

Parser from https://github.com/wichtounet/mnist
