//
// Created by karl on 05.05.19.
//

#include <cmath>
#include "EntropyCalculator.h"

double EntropyCalculator::getEntroy(const std::vector<int> &input, int sum) {
    double entropy = 0;

    for (int val : input) {
        if (val != 0) {
            double relVal = (double(val) / sum);
            entropy += relVal * std::log2(relVal);
        }
    }

    return -entropy;
}
