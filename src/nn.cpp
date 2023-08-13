#include "nn.h"
#include "types.h"
#include <fstream>
#include <iostream>
#include <omp.h>

// The forward pass of the network
const float NN::forward(Accumulator& accumulator, Features& features, Color stm) const {
    float output = hiddenBias[0]; // Initialize with the bias

    float* stmAccumulator = accumulator.data();
    float* nstmAccumulator = accumulator.data() + HIDDEN_SIZE;

    std::memcpy(stmAccumulator, inputBias.data(), sizeof(float) * HIDDEN_SIZE);
    std::memcpy(nstmAccumulator, inputBias.data(), sizeof(float) * HIDDEN_SIZE);

    for (int i = 0; i < features.n; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            stmAccumulator[j] += inputFeatures[features.features[i][stm] * HIDDEN_SIZE + j];
            nstmAccumulator[j] += inputFeatures[features.features[i][!stm] * HIDDEN_SIZE + j];
        }
    }
    
    for (int i = 0; i < HIDDEN_SIZE * 2; ++i){
        accumulator[i] = ReLU(accumulator[i]);
    }

    #pragma omp simd reduction(+:output)
    for (int i = 0; i < HIDDEN_SIZE; ++i){
        output += hiddenFeatures[i] * stmAccumulator[i];
    }

    #pragma omp simd reduction(+:output)
    for (int i = 0; i < HIDDEN_SIZE; ++i){
        output += hiddenFeatures[HIDDEN_SIZE + i] * nstmAccumulator[i];
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