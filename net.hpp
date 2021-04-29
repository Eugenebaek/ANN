//
// Created by Eugene Baek on 2021-04-15.
//

#ifndef NEURALNETWORK_NET_HPP
#define NEURALNETWORK_NET_HPP

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

#include "neuron.hpp"

class Net {
public:
    //Layer is a vector of Neurons
    //the Net will comprise of a 2d vector that contains a vector of layers that each contain a vector of neurons
    typedef std::vector<Neuron> Layer;

    //Net constructor
    Net(const std::vector<unsigned> &topology);

    //inputValues are passed by reference, no need to copy entire argument
    //the argument is const because the values are only being read, not changed
    void feedForward(const std::vector<double> &input_values);

    //same reasoning for targetValues
    void backPropagate(const std::vector<double> &target_values);

    //passed in by reference however, not const since we will write to this vector
    void getResult(std::vector<double> &result_values) const;

    double getRecentAverageError(void) const { return m_recent_average_error; };
private:

    //"layers" or the net will be a vector of multiple layers
    std::vector<Layer> m_layers;

    double m_error;
    double m_recent_average_error;
    static double m_recent_average_smoothing_factor;
};


#endif //NEURALNETWORK_NET_HPP
