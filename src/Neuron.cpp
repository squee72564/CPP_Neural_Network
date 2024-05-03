#include "Neuron.hpp"

inline double Neuron::eta_ = 0.25f;    // overall net learning
inline double Neuron::alpha_ = 0.3f;   // momentum, multiplier of last delta_weight


Neuron::Neuron(uint32_t num_outputs, Neuron::ActivationFunction activation_function)
  : output_weights_(num_outputs),
    output_value_(0.0f),
    gradient_(0.0f),
    neuron_index_(0),
    activation_function_(activation_function)
{

}

double Neuron::ApplyActivationFunction(double x) {
    double ret = 0.0f;

    switch (activation_function_) {
        case Neuron::TanH:
            ret = Neuron::_TanH(x);
            break;
        case Neuron::Sigmoid:
            ret = Neuron::_Sigmoid(x);
            break;
        case Neuron::Relu:
            ret = Neuron::_Relu(x);
            break;
        default:
            ret = x;
            break;
    }

    return ret;
}

double Neuron::ApplyActivationFunctionDerivative(double x)
{
    double ret = 0.0f;

    switch (activation_function_) {
        case Neuron::TanH:
            ret = Neuron::_TanHDerivative(x);
            break;
        case Neuron::Sigmoid:
            ret = Neuron::_SigmoidDerivative(x);
            break;
        case Neuron::Relu:
            ret = Neuron::_ReluDerivative(x);
            break;
        default:
            ret = x;
            break;
    }

    return ret;
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
    
    for (const Neuron &prev_neuron : prev_layer) {
        sum += prev_neuron.output_value_ * prev_neuron.output_weights_[neuron_index_].weight_;
    }
    
    output_value_ = ApplyActivationFunction(sum);
}

void Neuron::CalculateOuputGradients(const double target_value) {
    double delta = target_value - output_value_;
    if (activation_function_ == Neuron::SoftMax) {
        gradient_ = delta;
    } else {
        gradient_ = delta * ApplyActivationFunctionDerivative(output_value_); 
    }
}

void Neuron::CalculateHiddenGradients(const Layer &next_layer) {
    double dow = SumDOW(next_layer);
    gradient_ = dow * ApplyActivationFunctionDerivative(output_value_); 
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

double Neuron::_TanH(double x) {
    return tanh(x);
}

double Neuron::_TanHDerivative(double x) {
    // Derivative of tanh(x) is sech^2(x)
    double sech_x = 1.0 / cosh(x);
    return sech_x * sech_x;
}

double Neuron::_Sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double Neuron::_SigmoidDerivative(double x) {
    // Derivative of sigmoid(x) is sigmoid(x) * (1 - sigmoid(x))
    double sigmoid_x = Neuron::_Sigmoid(x);
    return sigmoid_x * (1.0 - sigmoid_x);
}

double Neuron::_Relu(double x) {
    return (x > 0) ? x : 0;
}

double Neuron::_ReluDerivative(double x) {
    // Derivative of relu(x) is 1 for x > 0, 0 otherwise
    return (x > 0) ? 1.0 : 0;
}
