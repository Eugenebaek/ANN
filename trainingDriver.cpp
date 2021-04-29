//
// Created by Eugene Baek on 2021-04-16.
//

#include "trainingDriver.hpp"

TrainingDriver::TrainingDriver(const std::string filename)
{
    m_trainingDataFile.open(filename.c_str());
}

void TrainingDriver::getTopology(std::vector<unsigned> &topology)
{
    std::string line;
    std::string label;

    getline(m_trainingDataFile, line);
    std::stringstream ss(line);
    ss >> label;
    if (this->isEof() || label.compare("topology:") != 0) {
        abort();
    }

    while (!ss.eof()) {
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }

    return;
}

unsigned TrainingDriver::getNextInputs(std::vector<double> &input_values)
{
    input_values.clear();

    std::string line;
    getline(m_trainingDataFile, line);
    std::stringstream ss(line);

    std::string label;
    ss>> label;
    if (label.compare("in:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            input_values.push_back(oneValue);
        }
    }

    return input_values.size();
}

unsigned TrainingDriver::getTargetOutputs(std::vector<double> &target_output_values)
{
    target_output_values.clear();

    std::string line;
    getline(m_trainingDataFile, line);
    std::stringstream ss(line);

    std::string label;
    ss>> label;
    if (label.compare("out:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            target_output_values.push_back(oneValue);
        }
    }

    return target_output_values.size();
}

