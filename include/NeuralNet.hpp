#ifndef NEURAL_NET_HPP
#define NEURAL_NET_HPP

#include <vector>
#include <iostream>
#include <random>
#include <ctime>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <cstdint>

class Connection;
class Neuron;
class NeuralNet;

typedef std::vector<Neuron> Layer;



struct Connection {
public:
    double weight_;
    double delta_weight_;

    Connection() : weight_(0), delta_weight_(0) {};
};

class Neuron {
public:
    std::vector<Connection> output_weights_;
    double output_value_;
    double gradient_;
    static double eta_;
    static double alpha_;
    size_t neuron_index_;

    Neuron (uint32_t num_outputs)
      : output_weights_(num_outputs),
        output_value_(0.0f),
        gradient_(0.0f),
        neuron_index_(0)
    {};

    static double ActivationFunction(double x) {
        return tanh(x);
    };

    static double ActivationFunctionDerivative(double x) {
        return 1.0 - x * 2;
    };

    double SumDOW(const Layer &next_layer) const {
        double sum = 0.0f;

        for (std::size_t n = 0; n < next_layer.size()-1; ++n) {
            sum += output_weights_[n].weight_ * next_layer[n].gradient_;
        }

        return sum;
    }

    void FeedForward(const Layer& prev_layer) {
        double sum = 0.0f;

        for (const Neuron &neuron : prev_layer) {
            sum += neuron.output_value_ * neuron.output_weights_[neuron_index_].weight_;
        }

        output_value_ = Neuron::ActivationFunction(sum);

    };

    void CalculateOuputGradients(const double target_value) {
        double delta = target_value - output_value_;
        gradient_ = delta * Neuron::ActivationFunctionDerivative(output_value_); 
    }

    void CalculateHiddenGradients(const Layer &next_layer) {
        double dow = SumDOW(next_layer);
        gradient_ = dow * Neuron::ActivationFunctionDerivative(output_value_);
    }

    void UpdateInputWeights(Layer &prev_layer) {
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
};

inline double Neuron::eta_ = 0.15f;    // overall net learning
inline double Neuron::alpha_ = 0.5f;   // momentum, multiplier of last delta_weight


class NeuralNet {
public:
    NeuralNet(const std::vector<uint32_t> &topology);

    void FeedForward(const std::vector<double> &input_vals);
    void BackPropagation(const std::vector<double> &target_vals);
    void GetResults(std::vector<double> &result_vals) const;
    double get_recent_average_error() const;
    inline double get_random_weight() const;
    static std::random_device rd;
    static std::mt19937 gen;
    static std::uniform_real_distribution<double> dis;

private:

    std::vector<Layer> layers_;
    double error_;
    double recent_average_error_;
    double recent_average_smoothing_factor_;
};

inline std::random_device NeuralNet::rd;
inline std::mt19937 NeuralNet::gen(rd());
inline std::uniform_real_distribution<double> NeuralNet::dis(0.0f, 1.0f);

#endif
