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

class Neuron {
public:
    Neuron () : x(0) {};
private:
    uint32_t x;

};

class NeuralNet {
public:
    NeuralNet(const std::vector<uint32_t> &topology);
    void FeedForward(const std::vector<double> &input_vals);
    void BackPropagation(const std::vector<double> &target_vals);
    void GetResults(std::vector<double> &result_vals) const;

private:
    typedef std::vector<Neuron> Layer;
    std::vector<Layer> layers_;
    
};


#endif
