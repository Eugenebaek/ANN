#include <iostream>
#include <vector>
#include <cassert>

#include "net.hpp"
#include "trainingDriver.hpp"


void showVectorValues(std::string label, std::vector<double> &v)
{
    std::cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        std::cout << v[i] << " ";
    }

    std::cout << std::endl;
}

int main() {

    TrainingDriver trainData("trainingData.txt");

    //number of layers and number of neurons per layer in neural net passed into myNet object through topology
    // e.g., { 3, 2, 1 }
    std::vector<unsigned> topology;
    trainData.getTopology(topology); //pass in vector specifying number of input neurons, layers, output neurons

    Net myNet(topology);

    std::vector<double> input_values, target_values, result_values;
    int trainingPass = 0;

    while (!trainData.isEof()) {
        ++trainingPass;
        std::cout << std::endl << "Pass " << trainingPass;

        // Get new input data and feed it forward:
        if (trainData.getNextInputs(input_values) != topology[0]) {
            break;
        }
        showVectorValues(": Inputs:", input_values);
        myNet.feedForward(input_values);

        // Collect the net's actual output results:
        myNet.getResult(result_values);
        showVectorValues("Outputs:", result_values);

        // Train the net what the outputs should have been:
        trainData.getTargetOutputs(target_values);
        showVectorValues("Targets:", target_values);
        assert(target_values.size() == topology.back());

        myNet.backPropagate(target_values);

        // Report how well the training is working, average over recent samples:
        std::cout << "Net recent average error: "
             << myNet.getRecentAverageError() << std::endl;
    }

    std::cout << std::endl << "Done" << std::endl;

    return 0;
}
