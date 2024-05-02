#include "NeuralNet.hpp"

#include <iostream>
#include <string>

NeuralNet::NeuralNet(const std::vector<uint32_t> &topology) 
  : layers_(topology.size())
{
    for (int i = 0; i < layers_.size(); ++i) {

        const uint32_t num_neuron_outputs = (i == layers_.size()-1)
            ? 0 
            : topology[i+1];

        layers_[i] = Layer(
                topology[i] + 1, // Include Bias Neuron
                Neuron(num_neuron_outputs)
            );

        // Randomize weights in each connection for each neuron
        for (auto& neuron : layers_[i]) {
            for (auto& connection : neuron.output_weights_) {
                connection.weight_ = get_random_weight();
            }
        }
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
