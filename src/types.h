#pragma once

#include <cstdint>
#include <array>
#include <random>
#include <cstring>
#include <sstream>
#include <chrono>
#include <iomanip>

constexpr int INPUT_SIZE = 64 * 6 * 2;
constexpr int HIDDEN_SIZE = 256;
constexpr int OUTPUT_SIZE = 1;

constexpr float EVAL_SCALE = 400.0f;
constexpr float EVAL_CP_RATIO = 0.7f;

constexpr int THREADS = 8;

constexpr std::size_t EPOCH_SIZE = 1e9;

static inline int inputIndex(uint8_t pieceType, uint8_t pieceColor, int square, uint8_t view, int kingSquare) {
    square = square ^ (56 * view);
    square = square ^ (7 * !!(kingSquare & 0x4));

    // clang-format off
    return square
           + pieceType * 64
           + !(pieceColor ^ view) * 64 * 6;
    // clang-format on
}