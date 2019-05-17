//
// Created by karl on 05.05.19.
//

#ifndef DECISIONTREE_FREQUENCYTABLE_H
#define DECISIONTREE_FREQUENCYTABLE_H


#include <string>
#include <vector>
#include <map>
#include <list>

class FrequencyTable {
public:
    FrequencyTable(const std::vector<std::vector<std::string>> &data, int column);

    double getGain();

    std::list<std::string> getAttributes();

private:
    // Maps attribute values to the frequencies of an outcome
    std::map<std::string, std::map<std::string, int>> attributeFrequencies;

    int attributeCount;
};


#endif //DECISIONTREE_FREQUENCYTABLE_H
