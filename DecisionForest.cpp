//
// Created by karl on 17.05.19.
//

#include <mnist_reader.hpp>
#include <thread>
#include <mutex>
#include <random>
#include <time.h>
#include "DecisionForest.h"

std::mutex forestMutex;

// Thread safe random numbers
// From https://stackoverflow.com/questions/21237905/how-do-i-generate-thread-safe-uniform-random-numbers
int intRand(const int & min, const int & max) {
    static thread_local std::mt19937* generator = nullptr;
    if (!generator) {
        std::hash<std::thread::id> hasher;
        generator = new std::mt19937(clock() + hasher(std::this_thread::get_id()));
    }
    std::uniform_int_distribution<int> distribution(min, max);
    return distribution(*generator);
}

static void trainOneTree(mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> &dataset,
                         int &samplesPerTree, int &attributesPerTree, int &attributesPerImage, std::list<DecisionTree> &forest) {
    DecisionTree tree = DecisionTree();

    // Randomly choose the attributes this tree will be looking at
    std::vector<std::vector<std::string>> dataForTree;
    std::vector<int> attributes;
    attributes.reserve(attributesPerTree);

    for (int attr = 0; attr < attributesPerTree; attr++) {
        attributes.push_back(intRand(0, attributesPerImage - 1));
    }

    // These attributes make up the first row of the tree (it expects a csv-like format)
    std::vector<std::string> attributesStrings;
    attributesStrings.reserve(attributes.size());

    for (auto attr : attributes) {
        attributesStrings.push_back(std::to_string(attr));
    }
    attributesStrings.emplace_back("outcome");
    dataForTree.push_back(attributesStrings);

    // Train the tree with random samples from the training data
    for (int j = 0; j < samplesPerTree; j++) {
        std::vector<std::string> row;

        // Choose a random image
        int entryIndex = intRand(0, dataset.training_images.size() - 1);
        auto entry = dataset.training_images[entryIndex];

        // Extract the attributes we're looking at with this tree
        for (int attribute : attributes) {
            row.push_back(std::to_string(entry[attribute]));
        }

        // Add the expected outcome (label) on the right
        row.push_back(std::to_string(dataset.training_labels[entryIndex]));

        dataForTree.push_back(row);
    }

    tree.build(dataForTree);

    forestMutex.lock();
    forest.push_back(tree);
    forestMutex.unlock();
}

void DecisionForest::train(mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset, int numberOfTrees,
                           int samplesPerTree, int attributesPerTree) {
    int attributesPerImage = dataset.test_images[0].size();

    std::vector<std::thread> threads;
    threads.reserve(numberOfTrees);

    // Create multiple decision trees
    for (int i = 0; i < numberOfTrees; i++) {
        threads.emplace_back(std::thread(trainOneTree, std::ref(dataset), std::ref(samplesPerTree), std::ref(attributesPerTree), std::ref(attributesPerImage), std::ref(forest)));
        //trainOneTree(std::ref(dataset), std::ref(samplesPerTree), std::ref(attributesPerTree), std::ref(attributesPerImage), std::ref(forest));
    }

    // Join threads
    for (int i = 0; i < numberOfTrees; i++) {
        threads[i].join();
    }
}

std::string DecisionForest::classify(std::vector<uint8_t> image) {
    // Transform the image into the format the trees need to classify
    std::map<std::string, std::string> entry;

    for (int j = 0; j < image.size(); j++) {
        entry[std::to_string(j)] = std::to_string(image[j]);
    }

    // Let all trees in the forest classify the image; remember the amount of times each class was chosen
    std::vector<std::thread> threads;
    threads.reserve(forest.size());

    std::map<std::string, int> decision;
    for (auto tree : forest) {
        decision[tree.classify(entry)]++;
    }

    // Get the most common decision
    std::string highestDecision;
    int highestDecisionCount = 0;

    for (const auto &outcome : decision) {
        if (outcome.second > highestDecisionCount) {
            highestDecisionCount = outcome.second;
            highestDecision = outcome.first;
        }
    }

    return highestDecision;
}
