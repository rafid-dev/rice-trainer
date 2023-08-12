#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <random>
#include <sstream>

constexpr int INPUT_SIZE  = 64 * 6 * 2;
constexpr int HIDDEN_SIZE = 256;
constexpr int OUTPUT_SIZE = 1;

constexpr float EVAL_SCALE    = 400.0f;
constexpr float EVAL_CP_RATIO = 0.7f;

constexpr int THREADS = 6;

constexpr float BETA1   = 0.9f;
constexpr float BETA2   = 0.999f;
constexpr float EPSILON = 1e-8f;

constexpr std::size_t EPOCH_SIZE = 1e8;
constexpr std::size_t BATCH_SIZE = 16384;

struct Features {
    uint8_t                                n = 0;
    std::array<std::array<int16_t, 2>, 32> features;

    void add(int16_t featureWhite, int16_t featureBlack) {
        features[n][0] = featureWhite;
        features[n][1] = featureBlack;
        n++;
    }
    void clear() {
        n = 0;
    }
};

struct NN {
    using Accumulator = std::array<float, HIDDEN_SIZE * 2>;
    using Color       = uint8_t;

    alignas(32) std::array<float, INPUT_SIZE * HIDDEN_SIZE> inputFeatures;
    alignas(32) std::array<float, HIDDEN_SIZE>              inputBias;
    alignas(32) std::array<float, HIDDEN_SIZE * 2>          hiddenFeatures;
    alignas(32) std::array<float, OUTPUT_SIZE>              hiddenBias;

    NN() {
        std::random_device rd;
        std::mt19937       gen(rd());

        // Calculate the scaling factor for He initialization
        float scale_input  = std::sqrt(2.0 / static_cast<float>(INPUT_SIZE));
        float scale_hidden = std::sqrt(2.0 / static_cast<float>(HIDDEN_SIZE));

        std::normal_distribution<float> input_distribution(0.0, scale_input);
        std::normal_distribution<float> hidden_distribution(0.0, scale_hidden);

        for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
            inputFeatures[i] = input_distribution(gen);
        }

        for (int i = 0; i < HIDDEN_SIZE * 2; i++) {
            hiddenFeatures[i] = hidden_distribution(gen);
        }

        // Initialize biases with small random values
        std::normal_distribution<float> bias_distribution(0.0, 0.01); // You can adjust the scale
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            inputBias[i] = bias_distribution(gen);
        }
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            hiddenBias[i] = bias_distribution(gen);
        }
    }

    const float forward(Accumulator& accumulator, Features& features, Color stm) const;
    void        load(const std::string& path);
    void        save(const std::string& path);
};

static inline std::string generateRandomHexValue(int numDigits) {
    std::random_device              rd;
    std::mt19937                    gen(rd());
    std::uniform_int_distribution<> dis(0, 15);

    std::ostringstream hexValue;
    for (int i = 0; i < numDigits; ++i) {
        hexValue << std::hex << dis(gen);
    }

    return hexValue.str();
}

static inline std::uint64_t getTimeMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}