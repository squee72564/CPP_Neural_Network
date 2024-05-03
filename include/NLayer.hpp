#ifndef NLAYER_HPP
#define NLAYER_HPP

#include <vector>
#include <cmath>
#include <cassert>
#include <cstdint>
#include <functional>

struct Connection {
public:
    double weight_;
    double delta_weight_;

    Connection() : weight_(0), delta_weight_(0) {};
};

class NLayer {
public:
    
    enum ActivationFunction {
        TanH = 0,
        Sigmoid,
        Relu,
        SoftMax,
        InputLayer,
    };
	
	NLayer();
    NLayer(uint32_t layer_size, uint32_t num_outputs, ActivationFunction f);
	~NLayer();

    std::vector<std::vector<Connection>> output_weights_;
    std::vector<double> output_values_;
    std::vector<double> gradients_;
    ActivationFunction activation_function_;
    uint32_t layer_size_;
    
    static double eta_;
    static double alpha_;

    static double _TanH(double x);
    static double _TanHDerivative(double x);
    static double _Sigmoid(double x);
    static double _SigmoidDerivative(double x);
    static double _Relu(double x);
    static double _ReluDerivative(double x);

    double ApplyActivationFunction(double x);
    double ApplyActivationFunctionDerivative(double x);
    void ApplySoftMax();
    double SumDOW(const uint32_t curr_idx, const NLayer &next_layer) const;
    void FeedForward(const NLayer& prev_layer);
    void CalculateOutputGradients(const std::vector<double> &target_values);
    void CalculateHiddenGradients(const NLayer &next_layer);
    void UpdateInputWeights(NLayer &prev_layer);
};

#endif
