//
// Created by Eugene Baek on 2021-04-15.
//

#include "neuron.hpp"

double Neuron::eta = 0.15; //overall net learning rate
double Neuron::alpha = 0.5; //momentum

Neuron::Neuron(unsigned num_of_outputs, unsigned my_index) {

    //connect this neuron to each neuron in next layer
    for (unsigned num_of_connections = 0; num_of_connections < num_of_outputs; num_of_connections++) {
        //create each connection with random weight
        m_output_weights.push_back(Connection());
        m_output_weights.back().weight = randWeight();
    }

    m_neuron_index = my_index;
}

//double Neuron::randWeight() {
//    std::cout << "rand(): " << rand() << std::endl;
//    std::cout << "double(RAND_MAX): " << double(RAND_MAX) << std::endl;
//    return rand() / double(RAND_MAX);
//}
//
//void Neuron::setOutputValue(double value) {
//    output_value = value;
//}
//
//double Neuron::getOutputValue() const {
//    return output_value;
//}

void Neuron::feedForward(const Layer &prev_layer) {
    //output = f(sigma(isubi wsubi)
    double sum = 0.0;

    //access each neuron in prev layer
    for (unsigned neuron_num = 0; neuron_num < prev_layer.size(); neuron_num++) {
        sum += prev_layer[neuron_num].getOutputValue() *
               prev_layer[neuron_num].m_output_weights[m_neuron_index].weight;
    }
    m_output_value = Neuron::transferFunction(sum);
}

double Neuron::transferFunction(double sum) {
    //tanh is used
    //output range [-1, 1]

    return tanh(sum);
}

double Neuron::transferFunctionDerivative(double sum) {
    //very close approx for derivative of tanh
    return 1.0 - sum * sum;
}

void Neuron::calcOutputGradient(double target_value) {
    double delta = target_value - m_output_value;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_output_value);
}

void Neuron::calcHiddenGradient(const Layer &next_layer) {
    double dow = sumDOW(next_layer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_output_value);
}

double Neuron::sumDOW(const Layer &next_layer) const {
    double sum = 0;

    //sum our contributions of the errors at the nodes we feed
    for (unsigned neuron_num = 0; neuron_num < next_layer.size() - 1; neuron_num++) {
        sum += m_output_weights[neuron_num].weight * next_layer[neuron_num].m_gradient;
    }

    return sum;
}

void Neuron::updateInputWeight(Layer &prev_layer) {
    //update weights for each neuron in previous layer
    for (unsigned neuron_num = 0; neuron_num < prev_layer.size(); neuron_num++) {
        Neuron &neuron = prev_layer[neuron_num];
        double prev_delta_weight = neuron.m_output_weights[m_neuron_index].delta_weight;
        double new_delta_weight =
                //individual input
                eta    //overall learning rate
                * neuron.getOutputValue()   //previous layer neuron's output value
                * m_gradient     //neuron gradient
                //add momentum = fraction of prev delta weight
                + alpha     //momentum rate
                  * prev_delta_weight;

        neuron.m_output_weights[m_neuron_index].delta_weight = new_delta_weight;
        neuron.m_output_weights[m_neuron_index].weight += new_delta_weight;

    }
}