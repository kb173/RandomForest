//
// Created by karl on 05.05.19.
//

#include <numeric>
#include "FrequencyTable.h"
#include "EntropyCalculator.h"
#include <map>
#include <iostream>


FrequencyTable::FrequencyTable(const std::vector<std::vector<std::string>> &data, int column) {

    bool first = true; // TODO: Ugly

    // Build the frequency table by counting how many times the positive or negative result
    // is reached for all possible attribute values
    for (const auto &line : data) {
        if (first) {
            first = false;
            continue;
        }

        std::string attributeVal = line[column];
        std::string resultVal = line.back();

        attributeFrequencies[attributeVal][resultVal] += 1;
    }
}

double FrequencyTable::getGain() {
    EntropyCalculator ec = EntropyCalculator();

    double gain = 0;

    for (const auto &attribute : attributeFrequencies) {
        std::string attributeName = attribute.first;
        std::map<std::string, int> countAtOutcome = attribute.second;

        std::vector<int> line;
        line.reserve(countAtOutcome.size());

        for (const auto &count : countAtOutcome) {
            line.push_back(count.second);
        }

        // Total number of occurrences of this attribute in data
        int sum = std::accumulate(line.begin(), line.end(), 0);

        // Here, the actual formula is applied
        gain -= (double(sum) / attributeCount) * ec.getEntroy(line, sum);
    }

    return gain;
}

std::list<std::string> FrequencyTable::getAttributes() {
    auto attributes = std::list<std::string>();

    for (const auto &attribute : attributeFrequencies) {
        attributes.push_back(attribute.first);
    }

    return attributes;
}
