//
// Created by Eugene Baek on 2021-04-16.
//

#ifndef NEURALNETWORK_TRAININGDRIVER_HPP
#define NEURALNETWORK_TRAININGDRIVER_HPP

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

class TrainingDriver {
public:
    TrainingDriver(const std::string filename);
    bool isEof(void) { return m_trainingDataFile.eof(); }
    void getTopology(std::vector<unsigned> &topology);

    // Returns the number of input values read from the file:
    unsigned getNextInputs(std::vector<double> &input_values);
    unsigned getTargetOutputs(std::vector<double> &target_output_values);

private:
    std::ifstream m_trainingDataFile;
};

#endif //NEURALNETWORK_TRAININGDRIVER_HPP
