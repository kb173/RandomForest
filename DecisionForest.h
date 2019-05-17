//
// Created by karl on 17.05.19.
//

#ifndef RANDOMFOREST_DECISIONFOREST_H
#define RANDOMFOREST_DECISIONFOREST_H


#include <list>
#include "DecisionTree.h"
#include "mnist_reader.hpp"

class DecisionForest {

public:
    DecisionForest() = default;

    void train(mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset, int numberOfTrees,
               int samplesPerTree, int attributesPerTree);

    std::string classify(std::vector<uint8_t> image);

private:
    std::list<DecisionTree> forest;

};


#endif //RANDOMFOREST_DECISIONFOREST_H
