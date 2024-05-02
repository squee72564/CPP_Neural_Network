#include <iostream>
#include <vector>
#include <cstdint>

#include "NeuralNet.hpp"

int main() {
    
    std::vector<uint32_t> topology = {8, 128, 32, 10};
    std::vector<double> input_values = {1};
    std::vector<double> target_values = {2};
    std::vector<double> result_values = {3};

    NeuralNet myNet(topology);
    myNet.FeedForward(input_values);
    myNet.BackPropagation(target_values);
    

    myNet.GetResults(result_values);

    return 0;
}
