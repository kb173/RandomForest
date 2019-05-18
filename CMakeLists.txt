cmake_minimum_required(VERSION 3.14)
project(RandomForest)

set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

include_directories(mnist)

add_executable(RandomForest
        main.cpp
        DecisionTree.cpp
        DecisionTree.h
        EntropyCalculator.cpp
        EntropyCalculator.h
        FrequencyTable.cpp
        FrequencyTable.h DecisionForest.cpp DecisionForest.h)