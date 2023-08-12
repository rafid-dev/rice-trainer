#include "nn.h"
#include "types.h"
#include <fstream>
#include <iostream>
#include <omp.h>

// The forward pass of the network
const float NN::forward(Accumulator& accumulator, Features& features, Color stm) const {
    float output = hiddenBias.at(0); // Initialize with the bias

#pragma omp simd
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        accumulator[i] = accumulator[i + HIDDEN_SIZE] = inputBias[i];
    }

    for (int i = 0; i < features.n; i++) {
#pragma omp simd
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            accumulator[j] += inputFeatures[features.features[i][stm] * HIDDEN_SIZE + j];
            accumulator[j + HIDDEN_SIZE] += inputFeatures[features.features[i][!stm] * HIDDEN_SIZE + j];
        }
        
    }

    vectorized_ReLU<HIDDEN_SIZE>(accumulator.data());

#pragma omp simd reduction(+ : output)
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        output += hiddenFeatures[i] * accumulator[i];
    }

#pragma omp simd reduction(+ : output)
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        output += hiddenFeatures[i + HIDDEN_SIZE] * accumulator[i + HIDDEN_SIZE];
    }

    return output;
}

void NN::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);

    if (file) {
        file.read(reinterpret_cast<char*>(inputFeatures.data()), sizeof(inputFeatures));
        file.read(reinterpret_cast<char*>(inputBias.data()), sizeof(inputBias));
        file.read(reinterpret_cast<char*>(inputFeatures.data()), sizeof(inputFeatures));
        file.read(reinterpret_cast<char*>(hiddenBias.data()), sizeof(hiddenBias));
    } else {
        std::cout << "Couldn't read checkpoint file " << path << std::endl;
    }
}

void NN::save(const std::string& path) {
    std::ofstream file(path, std::ios::binary);

    if (file) {
        file.write(reinterpret_cast<char*>(inputFeatures.data()), sizeof(inputFeatures));
        file.write(reinterpret_cast<char*>(inputBias.data()), sizeof(inputBias));
        file.write(reinterpret_cast<char*>(inputFeatures.data()), sizeof(inputFeatures));
        file.write(reinterpret_cast<char*>(hiddenBias.data()), sizeof(hiddenBias));
    } else {
        std::cout << "Couldn't write checkpoint file " << path << std::endl;
    }
}