#include <iostream>
#include <cmath>
#include <list>
#include <thread>
#include <mutex>
#include <iomanip>
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

std::mutex classifyListMutex;

static void classifyOne(DecisionForest &forest,  mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> &dataset, std::map<std::string, std::map<std::string, int>> &guessExpectMatrix, int index, int &goodDecisions, int &badDecisions) {
    std::string outcome = forest.classify(dataset.test_images[index]);
    std::string expected = std::to_string(dataset.test_labels[index]);

    classifyListMutex.lock();

    guessExpectMatrix[outcome][expected]++;

    if (outcome == expected) {
        goodDecisions += 1;
    } else {
        badDecisions += 1;
    }

    classifyListMutex.unlock();
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

    auto start = std::chrono::system_clock::now();
    forest.train(dataset, numberOfTrees, samplesPerTree, attributesPerTree);
    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Training took " << elapsed_seconds.count() << " seconds" << std::endl;

    // Classify!
    int goodDecisions = 0;
    int badDecisions = 0;

    std::map<std::string, std::map<std::string, int>> guessExpectMatrix;

    std::vector<std::thread> threads;
    threads.reserve(numberOfTrees);

    start = std::chrono::system_clock::now();
    for (int i = 0; i < dataset.test_images.size(); i++) {
        threads.emplace_back(std::thread(classifyOne, std::ref(forest), std::ref(dataset), std::ref(guessExpectMatrix), i, std::ref(goodDecisions), std::ref(badDecisions)));
        //classifyOne(forest, dataset, guessExpectMatrix, i, goodDecisions, badDecisions);
    }

    // Join threads
    for (int i = 0; i < dataset.test_images.size(); i++) {
        threads[i].join();
    }
    end = std::chrono::system_clock::now();

    elapsed_seconds = end - start;
    std::cout << "Validating took " << elapsed_seconds.count() << " seconds" << std::endl;
    std::cout << std::endl;

    // Print the guessExpectMatrix
    for (int x = 0; x < 10; x++) {
        for (int y = 0; y < 10; y++) {
            std::cout << std::setw(10) << guessExpectMatrix[std::to_string(x)][std::to_string(y)];
        }

        std::cout << std::endl;
    }

    double accuracy = double(goodDecisions) / (goodDecisions + badDecisions);
    std::cout << "Accuracy: " << accuracy << std::endl;

    return 0;
}
