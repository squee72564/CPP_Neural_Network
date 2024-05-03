#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <cstdlib>

#include "NeuralNet.hpp"
#include "MNIST.hpp"

std::vector<std::vector<double>> normalize_image_data(
        std::vector<std::vector<unsigned char>> data,
        uint32_t num_images, uint32_t image_size
    );

std::vector<std::vector<double>> hot_encode_target_data(
        std::vector<unsigned char> data,
        uint32_t num_images
    );

void display_normalized_image(std::vector<double> &image_data);

int main(int argc, char** argv) {
    
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << "mnist_img_path mnist_labels_path\n";
        return 1;
    }
    
    std::vector<std::vector<unsigned char> > train_x;
    std::vector<unsigned char> train_y;
    uint32_t num_test_imgs = 0;
    uint32_t img_size = 0;
    uint32_t num_labels = 0;

    try {
        train_x = MNIST::read_mnist_images(argv[1], num_test_imgs, img_size);
        train_y = MNIST::read_mnist_labels(argv[2], num_labels);
    } catch (std::runtime_error error) {
        std::cerr << error.what() << std::endl;
        return -1;
    };

    std::cout << "test imgs: " << num_test_imgs << " , size : " << img_size << "\n";
    std::cout << "num labels: " << num_labels << "\n";

    std::vector<std::vector<double>> train_x_normalized =
            normalize_image_data(train_x, num_test_imgs, img_size);

    std::vector<std::vector<double>> train_y_encoded =
            hot_encode_target_data(train_y, num_test_imgs);
    
    std::vector<NNLayer::LayerConfig> topology = {
        {img_size, NNLayer::InputLayer},
        {22, NNLayer::TanH},
        {14, NNLayer::TanH},
        {10, NNLayer::SoftMax},
    };

    assert(train_x_normalized.size() == train_y_encoded.size());

    std::vector<double> result_values = {};

    NeuralNet myNet(topology);
    NNLayer::alpha_ = 0.314159265399999f;
    NNLayer::eta_ = 0.00314159265399999f;
   
    for (int epoch = 0; epoch <= 100; ++epoch) {
        int rand_index = rand() % train_x_normalized.size();

        for ( int i = 0; i < train_x_normalized.size(); ++i) {
            // Get new input data and feed it forward
            assert(train_x_normalized[i].size() == topology[0].size_);
            myNet.FeedForward(train_x_normalized[i]);

            // Get the nets results
            if (i == rand_index) {
                myNet.GetResults(result_values);
            }

            // Train the net what the outputs should have been
            assert(train_y_encoded[i].size() == topology.back().size_);
            myNet.BackPropagation(train_y_encoded[i]);

        }

        // Display error and a random tested image from this epoch
        display_normalized_image(train_x_normalized[rand_index]);
        std::cout << "Error " << epoch << ": " << myNet.get_recent_average_error() << "\n";

        assert(result_values.size() == train_y_encoded[rand_index].size());

        std::cout << "[ ";
        for (int j = 0; j < train_y_encoded[rand_index].size(); ++j) {
            std::cout << train_y_encoded[rand_index][j] << " "; 
        }
        std::cout << " ]\n";

        auto max_ele = std::max_element(result_values.begin(), result_values.end());
        std::cout << "[ ";
        for (auto it = result_values.begin(); it < result_values.end(); ++it) {
            const int t = (it == max_ele) ? 1 : 0;
            std::cout << t << " "; 
        }
        std::cout << " ]\n";

        std::cout << "[ ";
        for (int j = 0; j < result_values.size(); ++j) {
            std::cout << result_values[j] << " "; 
        }
        std::cout << " ]\n";

    }


    return 0;
}

std::vector<std::vector<double>> normalize_image_data(std::vector<std::vector<unsigned char>> data, uint32_t num_images, uint32_t image_size) {
    std::vector<std::vector<double>> normalizedData(num_images, std::vector<double>(image_size, 0.0f));

    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < image_size; ++j) {
            const double normalizedValue = static_cast<double>(data[i][j]) / 255.0f;
            normalizedData[i][j] = normalizedValue;
        }
    }

    return normalizedData;
}

std::vector<std::vector<double>> hot_encode_target_data(std::vector<unsigned char> data, uint32_t num_images) {
    std::vector<std::vector<double>> normalizedData(num_images, std::vector<double>(10, 0.0f));

    for (int i = 0; i < num_images; ++i) {
        normalizedData[i][data[i]] = 1.0f;
    }

    return normalizedData;

}

void display_normalized_image(std::vector<double> &image_data) {
    const char ascii_chars[] = {' ', '.', ',', ':', ';', '-', '=', '+', '*', '#', '%', '$', '@', '&'};
    const int num_chars = sizeof(ascii_chars) / sizeof(ascii_chars[0]);
    
    for (int i = 0; i < image_data.size(); ++i) {
        int intensity = image_data[i] * (num_chars-1);
        std::cout << ascii_chars[intensity] << " ";
        if ((i + 1) % 28 == 0) 
            std::cout << "\n";
    }
    std::cout << std::endl;
}
