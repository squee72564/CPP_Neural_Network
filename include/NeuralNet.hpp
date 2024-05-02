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

static inline double get_random_weight() {
    return rand() / double(RAND_MAX);
}

class NeuralNet {
public:
    struct Connection {
        Connection() : weight_(0), delta_weight_(0) {};
        double weight_;
        double delta_weight_;
    };

    struct Neuron {
        Neuron (uint32_t num_outputs) : output_weights_(num_outputs) {};
        std::vector<NeuralNet::Connection> output_weights_;

    };

    NeuralNet(const std::vector<uint32_t> &topology);
    void FeedForward(const std::vector<double> &input_vals);
    void BackPropagation(const std::vector<double> &target_vals);
    void GetResults(std::vector<double> &result_vals) const;

private:
    typedef std::vector<Neuron> Layer;
    std::vector<Layer> layers_;
};


#endif
