//
// Created by Eugene Baek on 2021-04-15.
//

#ifndef NEURALNETWORK_NEURON_HPP
#define NEURALNETWORK_NEURON_HPP

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

struct Connection {
    double weight;
    double delta_weight;
};

class Neuron {
public:
    //Layer is a vector of Neurons
    //the Net will comprise of a 2d vector that contains a vector of layers that each contain a vector of neurons
    typedef std::vector<Neuron> Layer;
    //to create the neuron's connections to next layer, it only needs to know how many neurons are in next layer
    Neuron(unsigned num_of_outputs, unsigned neuron_index);
    void setOutputValue(double value) { m_output_value = value; };
    double getOutputValue(void) const { return m_output_value; };
    void feedForward(const Layer &prev_layer);
    void calcOutputGradient(double target_value);
    void calcHiddenGradient(const Layer &next_layer);
    void updateInputWeight(Layer &prev_layer);


private:
    static double transferFunction(double sum);
    static double transferFunctionDerivative(double sum);
    double sumDOW(const Layer &next_layer) const;
    static double randWeight(void) {
        std::cout << "rand(): " << rand() << std::endl;
        std::cout << "double(RAND_MAX): " << double(RAND_MAX) << std::endl;
        return rand() / double(RAND_MAX); };
    //output value of neuron
    double m_output_value;
    //weight of each connection to neurons in next layer
    std::vector<Connection> m_output_weights;
    unsigned m_neuron_index;
    double m_gradient;
    static double eta; //[0, 1]
    static double alpha; //[0, n]

};


#endif //NEURALNETWORK_NEURON_HPP
