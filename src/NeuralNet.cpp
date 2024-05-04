#include "NeuralNet.hpp"

#include <iostream>
#include <string>

std::random_device NeuralNet::rd;
std::mt19937 NeuralNet::gen(rd());
std::uniform_real_distribution<double> NeuralNet::dis(0.0f, 1.0f);

NeuralNet::NeuralNet(const std::vector<NNLayer::LayerConfig> &topology) 
  : layers_(topology.size()),
    error_(0.0f),
    recent_average_error_(0.0f),
    recent_average_smoothing_factor_(0.0f)
{
    for (int i = 0; i < layers_.size(); ++i) {

        const uint32_t num_neuron_outputs = (i == layers_.size()-1)
            ? 0 
            : topology[i+1].size_;

         layers_[i] = NNLayer(
                topology[i].size_ + 1, // Include Bias Neuron
                num_neuron_outputs,
                topology[i].activation_function_
            );
         

        // Randomize weights in each connection for each neuron
        for (std::size_t n = 0; n < layers_[i].output_weights_.size(); ++n) {
            for (auto& connection : layers_[i].output_weights_[n]) {
                connection.weight_ = get_random_weight();
            }
        }
    }
}

void NeuralNet::FeedForward(const std::vector<double> &input_values)
{
    // assert inputs == input neruons - bias nueron
    assert(input_values.size() == layers_[0].layer_size_ - 1);

    // Assign (latch) the input values into the input neurons
    for (std::size_t i = 0; i < input_values.size(); ++i) {
        layers_[0].output_values_[i] = input_values[i];
    }

    // Forward Propegate
    for (std::size_t layer_idx = 1; layer_idx < layers_.size(); ++layer_idx) {
        NNLayer &prev_layer = layers_[layer_idx-1];
        NNLayer &curr_layer = layers_[layer_idx];

        curr_layer.FeedForward(prev_layer);
    }
}

void NeuralNet::BackPropagation(const std::vector<double> &target_values)
{
    // Calculate the overall net errors (RMS)
    
    std::vector<double> &output_layer = layers_.back().output_values_;

    // Get sum of squares of error
    for (std::size_t n = 0; n < output_layer.size()-1; ++n) {
        double delta = target_values[n] - output_layer[n];
        error_ += delta * delta;
    }

    // Get average
    error_ /= output_layer.size() -1;

    // Get RMS
    error_ = sqrt(error_);
    
    // Recent average measurement

    recent_average_error_ = 
        (recent_average_error_ * recent_average_smoothing_factor_ + error_)
        / (recent_average_smoothing_factor_ + 1.0);

    // Calculate output layer gradients

    layers_.back().CalculateOutputGradients(target_values);

    // Calculate gradients on hidden layers

    for (std::size_t layer_idx = layers_.size()-2; layer_idx > 0; --layer_idx) {
        NNLayer &hidden_layer = layers_[layer_idx];
        NNLayer &next_layer = layers_[layer_idx+1];
        
        hidden_layer.CalculateHiddenGradients(next_layer);
    }
    
    // For all layers from outputs to first hidden layer, update connection weights
    
    for (std::size_t layer_idx = layers_.size()-1; layer_idx > 0; --layer_idx) {
        NNLayer &layer = layers_[layer_idx];
        NNLayer &prev_layer = layers_[layer_idx-1];

        layer.UpdateInputWeights(prev_layer);
    }
}

void NeuralNet::GetResults(std::vector<double> &result_values) const
{
    result_values.clear();

    const std::size_t n = layers_.back().layer_size_-1;
    result_values.reserve(n);

    const std::vector<double> &output_values = layers_.back().output_values_;

    for (std::size_t i = 0; i < n; ++i) {
        result_values.emplace_back(output_values[i]);
    }
}

double NeuralNet::get_recent_average_error() const {
    return recent_average_error_; 
}


inline double NeuralNet::get_random_weight() const {
    return NeuralNet::dis(NeuralNet::gen);
}
