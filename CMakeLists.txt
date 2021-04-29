cmake_minimum_required(VERSION 3.19)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 11)

add_executable(NeuralNetwork main.cpp net.cpp net.hpp neuron.cpp neuron.hpp trainingDriver.cpp trainingDriver.hpp)