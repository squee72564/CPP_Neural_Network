#include "NeuralNet.hpp"

#include <iostream>
#include <string>

std::random_device NeuralNet::rd;
std::mt19937 NeuralNet::gen(rd());
std::uniform_real_distribution<double> NeuralNet::dis(0.0f, 1.0f);

NeuralNet::NeuralNet(const std::vector<LayerConfig> &topology) 
  : layers_(topology.size()),
    error_(0.0f),
    recent_average_error_(0.0f),
    recent_average_smoothing_factor_(0.0f)
{
    for (int i = 0; i < layers_.size(); ++i) {

        const uint32_t num_neuron_outputs = (i == layers_.size()-1)
            ? 0 
            : topology[i+1].size_;

        layers_[i] = Layer(
                topology[i].size_ + 1, // Include Bias Neuron
                Neuron(num_neuron_outputs, topology[i].activation_function_)
            );

        
        // Randomize weights in each connection for each neuron
        for (std::size_t n = 0; n < layers_[i].size(); ++n) {
            layers_[i][n].neuron_index_ = n;
            for (auto& connection : layers_[i][n].output_weights_) {
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
        Layer &curr_layer = layers_[layer_idx];

        for (std::size_t neuron_idx = 0; neuron_idx < layers_[layer_idx].size()-1; ++neuron_idx) {
            curr_layer[neuron_idx].FeedForward(prev_layer);
        }
        
        if (curr_layer[0].activation_function_ == Neuron::SoftMax) {
            ApplySoftMax(curr_layer);
        }
    }
}

void NeuralNet::BackPropagation(const std::vector<double> &target_values)
{
    // Calculate the overall net errors (RMS)
    Layer &output_layer = layers_.back();

    // Get sum of squares of error
    for (std::size_t n = 0; n < output_layer.size()-1; ++n) {
        double delta = target_values[n] - output_layer[n].output_value_;
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

    for (std::size_t n = 0; n < output_layer.size()-1; ++n) {
        output_layer[n].CalculateOuputGradients(target_values[n]);
    }

    // Calculate gradients on hidden layers

    for (std::size_t layer_idx = layers_.size()-2; layer_idx > 0; --layer_idx) {
        Layer &hidden_layer = layers_[layer_idx];
        Layer &next_layer = layers_[layer_idx+1];

        for (std::size_t n = 0; n < hidden_layer.size(); ++n) {
            hidden_layer[n].CalculateHiddenGradients(next_layer);
        }
    }
    
    // For all layers from outputs to first hidden layer, update connection weights
    
    for (std::size_t layer_idx = layers_.size()-1; layer_idx > 0; --layer_idx) {
        Layer &layer = layers_[layer_idx];
        Layer &prev_layer = layers_[layer_idx-1];

        for (std::size_t n = 0; n < layer.size() -1; ++n) {
            layer[n].UpdateInputWeights(prev_layer);
        }
    }
}

void NeuralNet::GetResults(std::vector<double> &result_values) const
{
    result_values.clear();

    const std::size_t n = layers_.back().size()-1;
    result_values.reserve(n);

    for (std::size_t i = 0; i < n; ++i) {
        result_values.emplace_back(layers_.back()[i].output_value_);
    }
}

double NeuralNet::get_recent_average_error() const {
    return recent_average_error_; 
}


inline double NeuralNet::get_random_weight() const {
    return NeuralNet::dis(NeuralNet::gen);
}

void NeuralNet::ApplySoftMax(std::vector<Neuron>& curr_layer) {
    // Apply softmax activation
    double max_val = std::max_element(
                        curr_layer.begin(),
                        curr_layer.end(), 
                        [] (const auto &a, const auto &b ){ return a.output_value_ < b.output_value_; }
                    )->output_value_;

    double sum_exp = 0.0;

    for (size_t j = 0; j < curr_layer.size(); ++j) {
        curr_layer[j].output_value_ = std::exp(curr_layer[j].output_value_ - max_val); // Subtract max value for numerical stability
        sum_exp += curr_layer[j].output_value_;
    }

    for (size_t j = 0; j < curr_layer.size(); ++j) {
        curr_layer[j].output_value_ /= sum_exp;  // Normalize to obtain probabilities
    }

}
