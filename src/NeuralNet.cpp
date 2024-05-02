#include "NeuralNet.hpp"

#include <iostream>
#include <string>

NeuralNet::NeuralNet(const std::vector<uint32_t> &topology) 
  : layers_(topology.size())
{
    for (int i = 0; i < layers_.size(); ++i) {
        layers_[i].resize(topology[i]);
    }
}

void NeuralNet::FeedForward(const std::vector<double> &input_values)
{
    for (const double d : input_values) {
        std::cout << d << " ";
    }
    std::cout << "\n";
}

void NeuralNet::BackPropagation(const std::vector<double> &target_values)
{
    for (const double d : target_values) {
        std::cout << d << " ";
    }
    std::cout << "\n";
}

void NeuralNet::GetResults(std::vector<double> &result_values) const
{
    for (const double d : result_values) {
        std::cout << d << " ";
    }
    std::cout << "\n";
}
