#ifndef NEURAL_NET_HPP
#define NEURAL_NET_HPP

#include <vector>
#include <iostream>
#include <random>
#include <cassert>
#include <cstdint>

#include "Neuron.hpp"

class Neuron;

typedef std::vector<Neuron> Layer;

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
