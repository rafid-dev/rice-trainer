#pragma once

#include <cstdint>
#include <array>
#include <random>
#include <cstring>
#include <sstream>
#include <chrono>

constexpr int INPUT_SIZE = 64 * 6 * 2;
constexpr int HIDDEN_SIZE = 256;
constexpr int OUTPUT_SIZE = 1;

constexpr float EVAL_SCALE = 400.0f;
constexpr float EVAL_CP_RATIO = 0.7f;

constexpr int THREADS = 6;

constexpr float BETA1 = 0.9f;
constexpr float BETA2 = 0.999f;
constexpr float EPSILON = 1e-8f;

constexpr std::size_t EPOCH_SIZE = 1e9;

struct Features
{
    uint8_t n = 0;
    std::array<std::array<int16_t, 2>, 32> features;

    void add(int16_t featureWhite, int16_t featureBlack)
    {
        features[n][0] = featureWhite;
        features[n][1] = featureBlack;
        n++;
    }
    void clear()
    {
        n = 0;
    }
};

struct NN {
    using Accumulator = std::array<float, HIDDEN_SIZE * 2>;
    using Color = uint8_t;

    std::array<float, INPUT_SIZE * HIDDEN_SIZE> inputFeatures;
    std::array<float, HIDDEN_SIZE> inputBias;
    std::array<float, HIDDEN_SIZE * 2> hiddenFeatures;
    std::array<float, OUTPUT_SIZE> hiddenBias;

    NN(){
        std::random_device rd;
        std::mt19937                    gen(rd());
        std::normal_distribution<float> input_distribution(0.0, std::sqrt(1.0 / static_cast<float>(32)));
        std::normal_distribution<float> hidden_distribution(0.0, std::sqrt(1.0 / static_cast<float>(HIDDEN_SIZE)));

        for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
            inputFeatures[i] = input_distribution(gen);
        }

        for (int i = 0; i < HIDDEN_SIZE * 2; i++) {
            hiddenFeatures[i] = hidden_distribution(gen);
        }

        std::memset(inputBias.data(), 0, sizeof(float) * HIDDEN_SIZE);
        std::memset(hiddenBias.data(), 0, sizeof(float) * OUTPUT_SIZE);
    }

    const float forward(Accumulator& accumulator, Features& features, Color stm) const;
    void load(const std::string& path);
    void save(const std::string& path);
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