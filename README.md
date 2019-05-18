# RandomForest

Classifying the [MNIST handwritten digits](http://yann.lecun.com/exdb/mnist/) with a random forest in C++. Multithreaded!

## Default parameters
- 500 decision trees
- 1 000 random training images per tree
- 28 random pixels per tree (root of the number of pixels per image)

This results in an average accuracy of __91%__.

## Performance
- Training 60 000 images: __40s__
- Classifying 10 000 images: __340s__

(tested with an i7-4820K)

----

Parser from https://github.com/wichtounet/mnist
