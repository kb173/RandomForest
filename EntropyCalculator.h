//
// Created by karl on 05.05.19.
//

#ifndef DECISIONTREE_ENTROPYCALCULATOR_H
#define DECISIONTREE_ENTROPYCALCULATOR_H

#include <vector>


class EntropyCalculator {
public:
    EntropyCalculator() = default;

    double getEntroy(const std::vector<int> &input, int sum);
};


#endif //DECISIONTREE_ENTROPYCALCULATOR_H
