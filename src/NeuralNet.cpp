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
        for (std::size_t n = 0; i < layers_[i].size(); ++n) {
            Neuron &neuron = layers_[i][n];
            neuron.index = n;
            for (auto& connection : neuron.output_weights_) {
                connection.weight_ = get_random_weight();
            }
        }
    }
}

void NeuralNet::FeedForward(const std::vector<double> &input_values)
{
    // assert inputs == input neruons - bias nueron
    assert(input_values.size() == layers_[0].size() - 1);

    // Assign (latch) the input values into the input neurons
    for (std::size_t i = 0; i < input_values.size(); ++i) {
        layers_[0][i].output_value_ = input_values[i];
    }

    // Forward Propegate
    for (std::size_t layer_idx = 1; layer_idx < layers_.size(); ++layer_idx) {
        Layer &prev_layer = layers_[layer_idx-1];

        for (std::size_t neuron_idx = 0; neuron_idx < layers_[layer_idx].size()-1; ++neuron_idx) {
            layers_[layer_idx][neuron_idx].FeedForward(prev_layer);
        }
    }
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
