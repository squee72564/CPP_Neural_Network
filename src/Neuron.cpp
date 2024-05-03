#include "Neuron.hpp"

inline double Neuron::eta_ = 0.25f;    // overall net learning
inline double Neuron::alpha_ = 0.3f;   // momentum, multiplier of last delta_weight

Neuron::Neuron(uint32_t num_outputs)
  : output_weights_(num_outputs),
    output_value_(0.0f),
    gradient_(0.0f),
    neuron_index_(0) {}

double Neuron::ActivationFunction(double x) {
    return tanh(x);
}

double Neuron::ActivationFunctionDerivative(double x) {
    return 1.0 - x * 2;
}

double Neuron::SumDOW(const Layer &next_layer) const {
    double sum = 0.0f;

    for (std::size_t n = 0; n < next_layer.size()-1; ++n) {
        sum += output_weights_[n].weight_ * next_layer[n].gradient_;
    }

    return sum;
}

void Neuron::FeedForward(const Layer& prev_layer) {
    double sum = 0.0f;

    for (const Neuron &neuron : prev_layer) {
        sum += neuron.output_value_ * neuron.output_weights_[neuron_index_].weight_;
    }

    output_value_ = Neuron::ActivationFunction(sum);
}

void Neuron::CalculateOuputGradients(const double target_value) {
    double delta = target_value - output_value_;
    gradient_ = delta * Neuron::ActivationFunctionDerivative(output_value_); 
}

void Neuron::CalculateHiddenGradients(const Layer &next_layer) {
    double dow = SumDOW(next_layer);
    gradient_ = dow * Neuron::ActivationFunctionDerivative(output_value_);
}

void Neuron::UpdateInputWeights(Layer &prev_layer) {
    // the weights to be updated in the Connection container
    // in the neruons in the preceding layer

    for (std::size_t n = 0; n < prev_layer.size(); ++n) {
        Neuron &neuron = prev_layer[n];
        double old_delta_weight = neuron.output_weights_[neuron_index_].delta_weight_;

        double new_delta_weight =
            // Individual input, magnified by the gradient and train rate:
            eta_
            * neuron.output_value_
            * gradient_
            // Also add momentum - a fraction of the previous delta weight
            + alpha_
            * old_delta_weight;

        neuron.output_weights_[neuron_index_].delta_weight_ = new_delta_weight;
        neuron.output_weights_[neuron_index_].weight_ += new_delta_weight;
    }
}
