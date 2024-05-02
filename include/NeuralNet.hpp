#ifndef NEURAL_NET_HPP
#define NEURAL_NET_HPP

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <cstdint>

class Connection;
class Neuron;
class NeuralNet;

typedef std::vector<Neuron> Layer;

static inline double get_random_weight() {
    return rand() / double(RAND_MAX);
}

class Connection {
public:
    double weight_;
    double delta_weight_;

    Connection() : weight_(0), delta_weight_(0) {};
};

class Neuron {
public:
    std::vector<Connection> output_weights_;
    double output_value_;
    size_t neuron_index_;

    Neuron (uint32_t num_outputs) : output_weights_(num_outputs), output_value_(0.0f), index(0) {};

    static double transferFunction(double x) {
        return tanh(x);
    };

    static double transferFunctionDerivative(double x) {
        reutrn 1.0 - x * 2;
    };

    void FeedForward(const Layer& prev_layer) {
        double sum = 0.0f;

        for (const Neuron &neuron : prev_layer) {
            sum += neuron.output_value_ * neuron.output_weights[neuron_index_].weight_;
        }

        output_value_ = activationFunction(sum);

    };
};


class NeuralNet {
public:
    NeuralNet(const std::vector<uint32_t> &topology);
    void FeedForward(const std::vector<double> &input_vals);
    void BackPropagation(const std::vector<double> &target_vals);
    void GetResults(std::vector<double> &result_vals) const;

private:
    std::vector<Layer> layers_;
};


#endif
