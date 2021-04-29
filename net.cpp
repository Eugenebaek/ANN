//
// Created by Eugene Baek on 2021-04-15.
//

#include "net.hpp"

double Net::m_recent_average_smoothing_factor = 100.0; // Number of training samples to average over

Net::Net(const std::vector<unsigned> &topology) {

    unsigned num_of_layers = topology.size();
    //create Net by populating with number of layers specified in topology
    for (unsigned layer_num = 0; layer_num < num_of_layers; layer_num++) {
        //push into layers a Layer
        m_layers.push_back(Layer());

        //if output layer, no outputs to that neuron
        //otherwise (if inner layer), set number of outputs to be the number of neurons NEXT layer
        unsigned num_of_outputs = layer_num == topology.size() - 1 ? 0 : topology[layer_num + 1];

        //push individual neurons into Layer, including the bias neuron
        for (unsigned neuron_num = 0; neuron_num <= topology[layer_num]; neuron_num++) {
            m_layers.back().push_back(Neuron(num_of_outputs, neuron_num));
            std::cout << "made neuron in layer " << layer_num << std::endl;
        }

        //force bias neuron output value 1
        m_layers.back().back().setOutputValue(1.0);
    }
}

void Net::feedForward(const std::vector<double> &input_values) {

    //make sure correct number of input neurons got created
    assert(input_values.size() == m_layers[0].size() - 1);

    //assign input values to input neurons
    for (unsigned input_num = 0; input_num < input_values.size(); input_num++) {
        m_layers[0][input_num].setOutputValue(input_values[input_num]);
    }

    //forward propagate
    for (unsigned layer_num = 1; layer_num < m_layers.size(); layer_num++) {
        Layer &prev_layer = m_layers[layer_num - 1];
        //access each neuron in each layer
        for (unsigned neuron_num = 0; neuron_num < m_layers[layer_num].size() - 1; neuron_num++) {
            //neuron will have its own feedForward to handle to math
            m_layers[layer_num][neuron_num].feedForward(prev_layer);
        }
    }

}

void Net::backPropagate(const std::vector<double> &target_values) {
    //Calculate overall net error (RMS -Root Mean Square of output neuron errors)
    Layer &output_layer = m_layers.back();
    m_error = 0.0;
    for (unsigned neuron_num = 0; neuron_num < output_layer.size() - 1; neuron_num++) {
        //difference (delta) is the target (actual value) - the output value
        double delta = target_values[neuron_num] - output_layer[neuron_num].getOutputValue();

        m_error += delta * delta;
    }

    m_error /= output_layer.size() - 1;   //get average
    m_error = sqrt(m_error);    //rms

    //Recent average error to see progression
    m_recent_average_error =
            (m_recent_average_error * m_recent_average_smoothing_factor + m_error)
            / (m_recent_average_smoothing_factor + 1.0);

    //Calculate output layer gradients
    for (unsigned neuron_num = 0; neuron_num < output_layer.size() - 1; neuron_num++) {
        output_layer[neuron_num].calcOutputGradient(target_values[neuron_num]);
    }

    //Calculate gradients on hidden layers
    for (unsigned layer_num = m_layers.size() - 2; layer_num > 0; layer_num--) {
        Layer &hidden_layer = m_layers[layer_num];
        Layer &next_layer = m_layers[layer_num + 1];

        for (unsigned neuron_num = 0; neuron_num < hidden_layer.size(); neuron_num++) {
            hidden_layer[neuron_num].calcHiddenGradient(next_layer);
        }
    }

    //Update connection weights off all neurons
    for (unsigned layer_num = m_layers.size() - 1; layer_num > 0; layer_num--) {
        Layer &curr_layer = m_layers[layer_num];
        Layer &prev_layer = m_layers[layer_num - 1];

        for (unsigned neuron_num = 0; neuron_num < m_layers.size() - 1; neuron_num++) {
            curr_layer[neuron_num].updateInputWeight(prev_layer);
        }
    }
}

void Net::getResult(std::vector<double> &result_values) const {
    result_values.clear();
    for (unsigned neuron_num = 0; neuron_num < m_layers.back().size() - 1; neuron_num++) {
        result_values.push_back(m_layers.back()[neuron_num].getOutputValue());
    }
}

//double Net::getRecentAverageError() {
//    return recent_average_error;
//}
