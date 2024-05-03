#include "NNLayer.hpp"

inline double NNLayer::eta_ = 0.25f;    // overall net learning
inline double NNLayer::alpha_ = 0.3f;   // momentum, multiplier of last delta_weight

NNLayer::NNLayer()
  : output_weights_(),
    output_values_(),
	gradients_(),
	activation_function_(ActivationFunction::InputLayer),
	layer_size_(0)
{
}


NNLayer::NNLayer(uint32_t layer_size, uint32_t num_outputs, NNLayer::ActivationFunction activation_function)
  : output_weights_(layer_size, std::vector<Connection>(num_outputs)),
    output_values_(layer_size, 0.0f),
    gradients_(layer_size, 0.0f),
    activation_function_(activation_function),
    layer_size_(layer_size)
{
}

NNLayer::~NNLayer() {
	for (auto& neuron_connections : output_weights_) {
		neuron_connections.clear();
	}
	output_weights_.clear();
	
	output_values_.clear();
	gradients_.clear();
}

double NNLayer::ApplyActivationFunction(double x) {
    double ret = 0.0f;

    switch (activation_function_) {
        case NNLayer::TanH:
            ret = NNLayer::_TanH(x);
            break;
        case NNLayer::Sigmoid:
            ret = NNLayer::_Sigmoid(x);
            break;
        case NNLayer::Relu:
            ret = NNLayer::_Relu(x);
            break;
        default:
            ret = x;
            break;
    }

    return ret;
}

double NNLayer::ApplyActivationFunctionDerivative(double x)
{
    double ret = 0.0f;

    switch (activation_function_) {
        case NNLayer::TanH:
            ret = NNLayer::_TanHDerivative(x);
            break;
        case NNLayer::Sigmoid:
            ret = NNLayer::_SigmoidDerivative(x);
            break;
        case NNLayer::Relu:
            ret = NNLayer::_ReluDerivative(x);
            break;
        default:
            ret = x;
            break;
    }

    return ret;
}

double NNLayer::SumDOW(const uint32_t neuron_idx, const NNLayer &next_layer) const {
    double sum = 0.0f;

    for (std::size_t n = 0; n < next_layer.layer_size_-1; ++n) {
        sum += output_weights_[neuron_idx][n].weight_ * next_layer.gradients_[n];
    }

    return sum;
}

void NNLayer::FeedForward(const NNLayer& prev_layer) {

    for (int neuron_idx = 0; neuron_idx < layer_size_-1; ++neuron_idx) {
        double sum = 0.0f;
        
        for (int prev_neuron_idx = 0; prev_neuron_idx < prev_layer.layer_size_; ++prev_neuron_idx) {
            sum += 
                prev_layer.output_values_[prev_neuron_idx]
                * prev_layer.output_weights_[prev_neuron_idx][neuron_idx].weight_;
        }
        
        output_values_[neuron_idx] = ApplyActivationFunction(sum);
    }

    if (activation_function_ == NNLayer::SoftMax) {
        ApplySoftMax();
    }

}

void NNLayer::CalculateOutputGradients(const std::vector<double> &target_values) {
    assert(layer_size_-1 == target_values.size());

    for (int i = 0; i < layer_size_-1; ++i) {
        double delta = target_values[i] - output_values_[i];

        if (activation_function_ == NNLayer::SoftMax) {
            gradients_[i] = delta;
        } else {
            gradients_[i] = delta * ApplyActivationFunctionDerivative(output_values_[i]); 
        }
    }

}

void NNLayer::CalculateHiddenGradients(const NNLayer &next_layer) {
    for (int i = 0; i < layer_size_; ++i) {
        double dow = SumDOW(i, next_layer);
        gradients_[i] = dow * ApplyActivationFunctionDerivative(output_values_[i]); 
    }
}

void NNLayer::UpdateInputWeights(NNLayer &prev_layer) {

    for (std::size_t i = 0; i < layer_size_-1; ++i) {
        for (std::size_t n = 0; n < prev_layer.layer_size_; ++n) {
            double old_delta_weight = prev_layer.output_weights_[n][i].delta_weight_;

            double new_delta_weight =
                // Individual input, magnified by the gradient and train rate:
                eta_
                * prev_layer.output_values_[n]
                * gradients_[i]
                // Also add momentum - a fraction of the previous delta weight
                + alpha_
                * old_delta_weight;

            prev_layer.output_weights_[n][i].delta_weight_ = new_delta_weight;
            prev_layer.output_weights_[n][i].weight_ += new_delta_weight;
        }
    }
}

double NNLayer::_TanH(double x) {
    return tanh(x);
}

double NNLayer::_TanHDerivative(double x) {
    // Derivative of tanh(x) is sech^2(x)
    double sech_x = 1.0 / cosh(x);
    return sech_x * sech_x;
}

double NNLayer::_Sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double NNLayer::_SigmoidDerivative(double x) {
    // Derivative of sigmoid(x) is sigmoid(x) * (1 - sigmoid(x))
    double sigmoid_x = NNLayer::_Sigmoid(x);
    return sigmoid_x * (1.0 - sigmoid_x);
}

double NNLayer::_Relu(double x) {
    return (x > 0) ? x : 0;
}

double NNLayer::_ReluDerivative(double x) {
    // Derivative of relu(x) is 1 for x > 0, 0 otherwise
    return (x > 0) ? 1.0 : 0;
}

void NNLayer::ApplySoftMax() {
    // Apply softmax activation
    double max_val = *std::max_element(
                        output_values_.begin(),
                        output_values_.end(), 
                        [] (const auto &a, const auto &b ){ return a < b; }
                    );

    double sum_exp = 0.0;

    for (size_t j = 0; j < output_values_.size(); ++j) {
        output_values_[j] = std::exp(output_values_[j] - max_val); // Subtract max value for numerical stability
        sum_exp += output_values_[j];
    }

    for (size_t j = 0; j < output_values_.size(); ++j) {
        output_values_[j] /= sum_exp;  // Normalize to obtain probabilities
    }

}
