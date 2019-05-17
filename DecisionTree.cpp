#include <utility>

#include <utility>
#include <iostream>

//
// Created by karl on 05.05.19.
//

#include "DecisionTree.h"
#include "FrequencyTable.h"

void DecisionTreeNode::addNode(const std::string& attributeVal, std::shared_ptr<DecisionTreeNode> next) {
    nextNodes[attributeVal] = std::move(next);
}

const std::map<std::string, std::shared_ptr<DecisionTreeNode>> &DecisionTreeNode::getNextNodes() const {
    return nextNodes;
}

const std::string &DecisionTreeNode::getAttribute() const {
    return attribute;
}

void DecisionTree::build(const std::vector<std::vector<std::string>>& data) {
    root = buildRec(data);
}

void DecisionTree::print() {
    printRec(root, 0);
}

void DecisionTree::printRec(const std::shared_ptr<DecisionTreeNode>& currentNode, int depth) {
    auto nodes = currentNode->getNextNodes();

    if (nodes.empty()) { // This is a leaf node
        std::cout << "Decision: " << currentNode->getAttribute() << std::endl;
        return;
    }

    std::cout << currentNode->getAttribute() << "?" << std::endl;
    for (const auto& nextNode : nodes) {
        for (int i = 0; i < depth; i++) {
            std::cout << "\t";
        }
        std::cout << "-> " << nextNode.first << ": ";

        printRec(nextNode.second, depth + 1);
    }
}

std::shared_ptr<DecisionTreeNode> DecisionTree::buildRec(const std::vector<std::vector<std::string>> &data) {
    // Pure decision?
    std::string firstOutcome = data[1].back();
    bool pure = true;

    for (int i = 1; i < data.size(); i++) {
        if (data[i].back() != firstOutcome) {
            pure = false;
            break;
        }
    }

    if (pure) {
        // Make leaf node here (store the outcome in the attribute of the node)
        return std::make_shared<DecisionTreeNode>(DecisionTreeNode(firstOutcome));
    }

    // No more attributes?
    if (data[0].size() == 1) { // Only the 'Play' column left
        // Check which outcome is more common
        std::map<std::string, int> outcomes;

        for (int i = 1; i < data.size(); i++) {
            outcomes[data[i][0]] += 1;
        }

        std::string highestKey;
        int highestOccurance = 0;

        for (const auto& outcome : outcomes) {
            if (outcome.second > highestOccurance) {
                highestOccurance = outcome.second;
                highestKey = outcome.first;
            }
        }

        return std::make_shared<DecisionTreeNode>(DecisionTreeNode(highestKey));
    }

    int colNumber = data.front().size();

    // Check where the gain would be the highest
    double highestGain = -1000; // TODO: Initialize properly
    int colWithHighestGain = 0;
    std::unique_ptr<FrequencyTable> frequencyTableWithHighestGain;

    for (int col = 0; col < colNumber - 1; col++) {
        FrequencyTable ft = FrequencyTable(data, col);
        double gain = ft.getGain();

        if (gain > highestGain) {
            highestGain = gain;
            colWithHighestGain = col;
            frequencyTableWithHighestGain = std::make_unique<FrequencyTable>(ft);
        }
    }

    // Build this node accordingly
    std::shared_ptr<DecisionTreeNode> currentNode =
            std::make_shared<DecisionTreeNode>(DecisionTreeNode(data.front()[colWithHighestGain]));

    for (const auto &attribute : frequencyTableWithHighestGain->getAttributes()) {
        // Delete the column we chose, and only take all entries with the current attribute
        // for the new table
        // TODO: Could be made more efficient by removing the column we chose before this loop
        std::vector<std::vector<std::string>> new_data;
        std::vector<std::string> firstLine = data[0];
        firstLine.erase(firstLine.begin() + colWithHighestGain);

        new_data.push_back(firstLine);

        for (int i = 1; i < data.size(); i++) {
            // If the attributes match, add this line to the new data
            if (data[i][colWithHighestGain] == attribute) {
                std::vector<std::string> line = data[i];
                line.erase(line.begin() + colWithHighestGain);

                new_data.push_back(line);
            }
        }

        std::shared_ptr<DecisionTreeNode> nodeToInsert = buildRec(new_data);

        // If the tree continues, stick it to this node (at this attribute)
        if (nodeToInsert != nullptr) {
            currentNode->addNode(attribute, nodeToInsert);
        }
    }

    return currentNode;
}

std::string DecisionTree::classify(std::map<std::string, std::string> attributes) {
    // Walk down the tree until we arrive at a leaf node
    std::shared_ptr currentNode = root;
    std::map<std::string, std::shared_ptr<DecisionTreeNode>> nextNodes = currentNode->getNextNodes();

    do {
        if (nextNodes[attributes[currentNode->getAttribute()]] == nullptr) {
            break; // FIXME why can this be the case?
        }
        currentNode = nextNodes[attributes[currentNode->getAttribute()]];
        nextNodes = currentNode->getNextNodes();
    } while(!nextNodes.empty());

    // Return the attribute, where leaf nodes store the classification
    return currentNode->getAttribute();
}
