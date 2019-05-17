#include <iostream>
#include <cmath>
#include <list>
#include "mnist_reader.hpp"
#include "DecisionTree.h"

int main(int argc, char* argv[]) {
    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
            mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("/home/karl/Data/Technikum/SEM4/MLE/RandomForest/Datasets");

    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

    // Normalize the image data
    for (auto &image : dataset.training_images) {
        for (uint8_t &pixel : image) {
            if (pixel > 0) {
                pixel = 1;
            }
        }
    }
    for (auto &image : dataset.test_images) {
        for (uint8_t &pixel : image) {
            if (pixel > 0) {
                pixel = 1;
            }
        }
    }

    clock_t trainBegin = clock();

    int attributesPerImage = dataset.test_images[0].size();
    int numberOfTrees = 500;
    int samplesPerTree = 1000;
    int attributesPerTree = sqrt(attributesPerImage);


    std::list<DecisionTree> forest;

    srand(time(nullptr));

    for (int i = 0; i < numberOfTrees; i++) {
        DecisionTree tree = DecisionTree();

        std::vector<std::vector<std::string>> dataForTree;
        std::vector<int> attributes;

        // Check which attributes this tree is looking at
        for (int attr = 0; attr < attributesPerTree; attr++) {
            attributes.push_back(std::rand() % attributesPerImage);
        }

        // These attributes make up the first row of the tree
        std::vector<std::string> attributesStrings;
        for (auto attr : attributes) {
            attributesStrings.push_back(std::to_string(attr));
        }
        attributesStrings.emplace_back("outcome");
        dataForTree.push_back(attributesStrings);

        for (int j = 0; j < samplesPerTree; j++) {
            std::vector<std::string> row;

            // Choose a random training data entry
            int entryIndex = std::rand() % dataset.training_images.size();
            auto entry = dataset.training_images[entryIndex];

            for (int attribute : attributes) {
                row.push_back(std::to_string(entry[attribute]));
            }

            // Add the classification
            row.push_back(std::to_string(dataset.training_labels[entryIndex]));

            dataForTree.push_back(row);
        }

        tree.build(dataForTree);
        forest.push_back(tree);
    }

    clock_t trainEnd = clock();
    auto elapsedTime = double(trainEnd - trainBegin) / 1000000;

    std::cout << "Training took " << elapsedTime << "s" << std::endl;

    clock_t classifyBegin = clock();

    // Classify!
    int goodDecisions = 0;
    int badDecisions = 0;

    for (int i = 0; i < dataset.test_images.size(); i++) {
        std::map<std::string, std::string> entry;

        for (int j = 0; j < dataset.test_images[i].size(); j++) {
            entry[std::to_string(j)] = std::to_string(dataset.test_images[i][j]);
        }

        std::map<std::string, int> decision;

        for (auto tree : forest) {
            decision[tree.classify(entry)]++;
        }

        std::string highestDecision;
        int highestDecisionCount = 0;

        for (auto outcome : decision) {
            if (outcome.second > highestDecisionCount) {
                highestDecisionCount = outcome.second;
                highestDecision = outcome.first;
            }
        }

        if (highestDecision == std::to_string(dataset.test_labels[i])) {
            goodDecisions += 1;
        } else {
            badDecisions += 1;
        }
    }

    double accuracy = double(goodDecisions) / (goodDecisions + badDecisions);

    clock_t classifyEnd = clock();
    elapsedTime = double(classifyEnd - classifyBegin) / 1000000;

    std::cout << "Classifying took " << elapsedTime << "s, that's " << elapsedTime / dataset.test_images.size() << "s per image" << std::endl;

    std::cout << "Good decisions: " << goodDecisions << std::endl;
    std::cout << "Bad decisions: " << badDecisions << std::endl;

    std::cout << "Accuracy: " << accuracy << std::endl;

    return 0;
}
