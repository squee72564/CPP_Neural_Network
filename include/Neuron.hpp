#ifndef NEURON_HPP
#define NEURON_HPP

#include <vector>
#include <cmath>
#include <cstdint>

class Neuron;

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
    size_t neuron_index_;
    static double eta_;
    static double alpha_;

    Neuron(uint32_t num_outputs);

    static double ActivationFunction(double x);
    static double ActivationFunctionDerivative(double x);
    double SumDOW(const Layer &next_layer) const;
    void FeedForward(const Layer& prev_layer);
    void CalculateOuputGradients(const double target_value);
    void CalculateHiddenGradients(const Layer &next_layer);
    void UpdateInputWeights(Layer &prev_layer);
};

inline double Neuron::eta_ = 0.15f;    // overall net learning
inline double Neuron::alpha_ = 0.5f;   // momentum, multiplier of last delta_weight

#endif
