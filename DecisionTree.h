#include <utility>

//
// Created by karl on 05.05.19.
//

#ifndef DECISIONTREE_DECISIONTREE_H
#define DECISIONTREE_DECISIONTREE_H


#include <string>
#include <vector>
#include <memory>
#include <map>


// A node for an attribute. Depending on the attribute value, it leads to other nodes.
// If it's a leaf node, the outcome is stored in the attribute (and the nextNodes are empty)
class DecisionTreeNode {
public:
    explicit DecisionTreeNode(std::string _attribute) : attribute(std::move(_attribute)) {};

    void addNode(const std::string &attributeVal, std::shared_ptr<DecisionTreeNode> next);

    const std::map<std::string, std::shared_ptr<DecisionTreeNode>> &getNextNodes() const;

    const std::string &getAttribute() const;

private:
    std::map<std::string, std::shared_ptr<DecisionTreeNode>> nextNodes;

    std::string attribute;
};


class DecisionTree {
public:
    DecisionTree() = default;

    void build(const std::vector<std::vector<std::string>>& data);

    void print();

    std::string classify(std::map<std::string, std::string>);

private:
    std::shared_ptr<DecisionTreeNode> root;

    std::shared_ptr<DecisionTreeNode> buildRec(const std::vector<std::vector<std::string>> &data);

    void printRec(const std::shared_ptr<DecisionTreeNode>& currentNode, int depth);
};


#endif //DECISIONTREE_DECISIONTREE_H
