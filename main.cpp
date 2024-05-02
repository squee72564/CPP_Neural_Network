#include <iostream>
#include <vector>
#include <cstdint>

#include "NeuralNet.hpp"

int main() {
    
    std::vector<uint32_t> topology = {2, 2, 1};

    std::vector<std::vector<double>> input_values = {
        {0.0f, 0.0f},
        {1.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 1.0f},
        {0.0f, 0.0f},
        {0.0f, 0.0f},
        {0.0f, 0.0f},
        {1.0f, 0.0f},
        {0.0f, 0.0f},
    };

    std::vector<std::vector<double>> target_values = {
        {0.0f},
        {1.0f},
        {1.0f},
        {1.0f},
        {0.0f},
        {0.0f},
        {0.0f},
        {1.0f},
        {0.0f},
    };

    assert(input_values.size() == target_values.size());

    std::vector<double> result_values = {};

    NeuralNet myNet(topology);
    
    for ( int i = 0; i < input_values.size(); ++i) {

        // Get new input data and feed it forward
        assert(input_values[i].size() == topology[0]);
        myNet.FeedForward(input_values[i]);

        // Get the nets results
        myNet.GetResults(result_values);
        
        // Train the net what the outputs should have been
        assert(target_values[i].size() == topology.back());
        myNet.BackPropagation(target_values[i]);
        
        std::cout << "Error " << i << ": " << myNet.get_recent_average_error() << "\n";
        std:: cout << "\t-- " << target_values[i][0] << ", " << result_values[0] << "\n";
    }

    return 0;
}
