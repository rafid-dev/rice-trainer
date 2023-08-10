#pragma once

#include <cstdint>
#include <array>
#include <algorithm>

template<typename T = float>
static inline const T ReLU(const T x){
    return std::max<T>(0, x);
} 

template<typename T = float>
static inline const T ReLUPrime(const T x){
    return x > 0 ? 1 : 0;
} 

template<typename T = float>
static inline const T sigmoid(const T x){
    return 1 / (1 + expf(-x));
}

template<typename T = float>
static inline const T sigmoidPrime(const T x){
    return sigmoid(x) * (1 - sigmoid(x));
}

static inline int inputIndex(uint8_t pieceType, uint8_t pieceColor, int square, uint8_t view, int kingSquare) {
    square = square ^ (56 * view);
    square = square ^ (7 * !!(kingSquare & 0x4));

    // clang-format off
    return square
           + pieceType * 64
           + !(pieceColor ^ view) * 64 * 6;
    // clang-format on
}