#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <random>

#include "NeuralNet.hpp"

unsigned char** read_mnist_images(std::string full_path, uint32_t& number_of_images, uint32_t& image_size);
unsigned char* read_mnist_labels(std::string full_path, uint32_t& number_of_labels);
void display_mnist_image(unsigned char* image_data, uint32_t image_size);
void display_normalized_image(std::vector<double> &image_data);
std::vector<std::vector<double>> normalize_image_data(unsigned char** data, uint32_t num_images, uint32_t image_size);
std::vector<std::vector<double>> normalize_target_data(unsigned char* data, uint32_t num_images);

int main(int argc, char** argv) {
    
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << "mnist_img mnist_labels\n";
        return 1;
    }

    uint32_t num_test_imgs = 0;
    uint32_t img_size = 0;
    unsigned char** train_x = read_mnist_images(argv[1], num_test_imgs, img_size);

    uint32_t num_labels = 0;
    unsigned char* train_y = read_mnist_labels(argv[2], num_labels);

    std::cout << "test imgs: " << num_test_imgs << " , size : " << img_size << "\n";
    std::cout << "num labels: " << num_labels << "\n";

    std::vector<std::vector<double>> train_x_normalized = normalize_image_data(train_x, num_test_imgs, img_size);
    std::vector<std::vector<double>> train_y_normalized = normalize_target_data(train_y, num_test_imgs);

    for (int i = 0; i < num_test_imgs; ++i) {
        if (i % 1500 == 0) {
            display_normalized_image(train_x_normalized[i]);
            std::cout << static_cast<int>(train_y[i]) << ", ";
            std::cout << "[ ";
            for (const double& d : train_y_normalized[i]) {
                std::cout << d << " ";
            }
            std::cout << " ]\n";
        } 
    }

    for (int i = 0; i < num_test_imgs; ++i) {
        delete[] train_x[i];
    }

    delete[] train_x;

    delete[] train_y;
    
    std::vector<uint32_t> topology = {img_size, 32, 16, 10};

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0f, 1.0f);

    assert(train_x_normalized.size() == train_y_normalized.size());

    std::vector<double> result_values = {};

    NeuralNet myNet(topology);
    Neuron::alpha_ = 0.45f;
    Neuron::eta_ = 0.1f;
   
    for ( int i = 0; i < train_x_normalized.size(); ++i) {
        // Get new input data and feed it forward
        assert(train_x_normalized[i].size() == topology[0]);
        myNet.FeedForward(train_x_normalized[i]);

        // Get the nets results
        myNet.GetResults(result_values);
        
        // Train the net what the outputs should have been
        assert(train_y_normalized[i].size() == topology.back());
        myNet.BackPropagation(train_y_normalized[i]);

        if (i % 1000 == 0) {
            std::cout << "Error " << i << ": " << myNet.get_recent_average_error() << "\n";

            std::cout << "[ ";
            for (int j = 0; j < train_y_normalized[i].size(); ++j) {
                std::cout << train_y_normalized[i][j] << " "; 
            }
            std::cout << " ]\n";

            std::cout << "[ ";
            for (int j = 0; j < train_y_normalized[i].size(); ++j) {
                std::cout << result_values[i] << " "; 
            }
            std::cout << " ]\n";
        }
    }

    return 0;
}

unsigned char** read_mnist_images(std::string full_path, uint32_t& number_of_images, uint32_t& image_size) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    std::ifstream file(full_path, std::ios::binary);

    if(file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2051) throw std::runtime_error("Invalid MNIST image file!");

        file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        image_size = n_rows * n_cols;

        unsigned char** _dataset = new unsigned char*[number_of_images];
        for(int i = 0; i < number_of_images; i++) {
            _dataset[i] = new unsigned char[image_size];
            file.read((char *)_dataset[i], image_size);
        }
        return _dataset;
    } else {
        throw std::runtime_error("Cannot open file `" + full_path + "`!");
    }
}

unsigned char* read_mnist_labels(std::string full_path, uint32_t& number_of_labels) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    std::ifstream file(full_path, std::ios::binary);

    if(file.is_open()) {
        int magic_number = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2049) throw std::runtime_error("Invalid MNIST label file!");

        file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

        unsigned char* _dataset = new unsigned char[number_of_labels];
        for(int i = 0; i < number_of_labels; i++) {
            file.read((char*)&_dataset[i], 1);
        }
        return _dataset;
    } else {
        throw std::runtime_error("Unable to open file `" + full_path + "`!");
    }
}

void display_mnist_image(unsigned char* image_data, uint32_t image_size) {
    const char ascii_chars[] = {' ', '.', ',', ':', ';', '-', '=', '+', '*', '#', '%', '$', '@', '&'};
    const int num_chars = sizeof(ascii_chars) / sizeof(ascii_chars[0]);

    for (int i = 0; i < image_size; ++i) {
        int intensity = image_data[i] / (256 / num_chars);
        std::cout << ascii_chars[intensity] << " ";
        if ((i + 1) % 28 == 0) 
            std::cout << "\n";
    }
    std::cout << std::endl;
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

std::vector<std::vector<double>> normalize_image_data(unsigned char** data, uint32_t num_images, uint32_t image_size) {
    std::vector<std::vector<double>> normalizedData(num_images, std::vector<double>(image_size, 0.0f));

    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < image_size; ++j) {
            const double normalizedValue = static_cast<double>(data[i][j]) / 255.0;
            normalizedData[i][j] = normalizedValue;
        }
    }

    return normalizedData;
}

std::vector<std::vector<double>> normalize_target_data(unsigned char* data, uint32_t num_images) {
    std::vector<std::vector<double>> normalizedData(num_images, std::vector<double>(10, 0.0f));

    for (int i = 0; i < num_images; ++i) {
        normalizedData[i][data[i]] = 1.0f;
    }

    return normalizedData;

}
