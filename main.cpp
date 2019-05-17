#include <iostream>
#include <cmath>
#include <list>
#include "mnist_reader.hpp"
#include "DecisionTree.h"
#include "DecisionForest.h"

void clean_images(std::vector<std::vector<uint8_t>> &images) {
    for (auto &image : images) {
        for (uint8_t &pixel : image) {
            if (pixel > 0) {
                pixel = 1;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
            mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(
                    "/home/karl/Data/Technikum/SEM4/MLE/RandomForest/Datasets");

    clean_images(dataset.test_images);
    clean_images(dataset.training_images);

    int attributesPerImage = dataset.test_images[0].size();
    int numberOfTrees = 500;
    int samplesPerTree = 1000;
    int attributesPerTree = sqrt(attributesPerImage);

    DecisionForest forest;

    clock_t trainBegin = clock();
    forest.train(dataset, numberOfTrees, samplesPerTree, attributesPerTree);
    clock_t trainEnd = clock();

    auto elapsedTime = double(trainEnd - trainBegin) / 1000000;
    std::cout << "Training took " << elapsedTime << "s" << std::endl;

    clock_t classifyBegin = clock();

    // Classify!
    int goodDecisions = 0;
    int badDecisions = 0;

    std::map<std::string, std::map<std::string, int>> guessExpectMatrix;


    for (int i = 0; i < dataset.test_images.size(); i++) {
        std::string outcome = forest.classify(dataset.test_images[i]);
        std::string expected = std::to_string(dataset.test_labels[i]);

        guessExpectMatrix[outcome][expected]++;

        if (outcome == expected) {
            goodDecisions += 1;
        } else {
            badDecisions += 1;
        }
    }

    clock_t classifyEnd = clock();
    elapsedTime = double(classifyEnd - classifyBegin) / 1000000;
    std::cout << "Classifying took " << elapsedTime << "s, that's " << elapsedTime / dataset.test_images.size()
              << "s per image" << std::endl;

    // Print the guessExpectMatrix
    for (int x = 0; x < 10; x++) {
        for (int y = 0; y < 10; y++) {
            std::cout << guessExpectMatrix[std::to_string(x)][std::to_string(y)] << "\t";
        }

        std::cout << std::endl;
    }

    double accuracy = double(goodDecisions) / (goodDecisions + badDecisions);
    std::cout << "Accuracy: " << accuracy << std::endl;

    return 0;
}
