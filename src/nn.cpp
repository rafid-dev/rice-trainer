#include "nn.h"
#include "types.h"
#include "quantize.h"
#include "dataloader.h"
#include <fstream>
#include <iostream>
#include <omp.h>

void NN::testFen(const std::string& fen){
    chess::Position pos{chess::Position::fromFen(fen)};

    Features features;

    chess::Bitboard        pieces = pos.piecesBB();

    const chess::Square ksq_White = pos.kingSquare(chess::Color::White);
    const chess::Square ksq_Black = pos.kingSquare(chess::Color::Black);

    for (chess::Square sq : pieces) {
        const chess::Piece piece      = pos.pieceAt(sq);
        const std::uint8_t pieceType  = static_cast<uint8_t>(piece.type());
        const std::uint8_t pieceColor = static_cast<uint8_t>(piece.color());

        const int featureW = inputIndex(pieceType, pieceColor, static_cast<int>(sq), static_cast<uint8_t>(chess::Color::White), static_cast<int>(ksq_White));
        const int featureB = inputIndex(pieceType, pieceColor, static_cast<int>(sq), static_cast<uint8_t>(chess::Color::Black), static_cast<int>(ksq_Black));

        features.add(featureW, featureB);
    }

    Accumulator accumulator;
    Color       stm = Color(pos.sideToMove());
    
    std::cout << "Score: " << forward(accumulator, features, stm) << std::endl;
}

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
        file.read(reinterpret_cast<char*>(hiddenFeatures.data()), sizeof(hiddenFeatures));
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
        file.write(reinterpret_cast<char*>(hiddenFeatures.data()), sizeof(hiddenFeatures));
        file.write(reinterpret_cast<char*>(hiddenBias.data()), sizeof(hiddenBias));
    } else {
        std::cout << "Couldn't write checkpoint file " << path << std::endl;
    }
}

void NN::quantize(const std::string& path, bool print){
    QuantizedNN qnn{*this, print};
    
    qnn.save(path);

    if (print){
        std::cout << "Quantized network saved to " << path << std::endl;
    }
}

void QuantizedNN::testFen(const std::string& fen){
    chess::Position pos{chess::Position::fromFen(fen)};

    Features features;

    chess::Bitboard        pieces = pos.piecesBB();

    const chess::Square ksq_White = pos.kingSquare(chess::Color::White);
    const chess::Square ksq_Black = pos.kingSquare(chess::Color::Black);

    for (chess::Square sq : pieces) {
        const chess::Piece piece      = pos.pieceAt(sq);
        const std::uint8_t pieceType  = static_cast<uint8_t>(piece.type());
        const std::uint8_t pieceColor = static_cast<uint8_t>(piece.color());

        const int featureW = inputIndex(pieceType, pieceColor, static_cast<int>(sq), static_cast<uint8_t>(chess::Color::White), static_cast<int>(ksq_White));
        const int featureB = inputIndex(pieceType, pieceColor, static_cast<int>(sq), static_cast<uint8_t>(chess::Color::Black), static_cast<int>(ksq_Black));

        features.add(featureW, featureB);
    }

    Accumulator accumulator;
    Color       stm = Color(pos.sideToMove());
    
    std::cout << "Score: " << forward(accumulator, features, stm) << std::endl;
}

const int32_t QuantizedNN::forward(Accumulator& accumulator, Features& features, Color stm) const{
    int32_t output = hiddenBias[0]; // Initialize with the bias

    auto* stmAccumulator = accumulator.data();
    auto* nstmAccumulator = accumulator.data() + HIDDEN_SIZE;

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
    
    return output / (Q1 * Q2);
}