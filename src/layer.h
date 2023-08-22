#pragma once

#include <cstdint>
#include <vector>

#include "nn.h"

struct Tensor {
    using Matrix = std::vector<float>;

    Matrix m_values;
    Matrix m_grad;

    Tensor(std::size_t _size) : m_values(_size), m_grad(_size){};
    Tensor() = default;

    constexpr void resize(std::size_t _size) {
        m_values.resize(_size);
        m_grad.resize(_size);
    }

    constexpr float* data() {
        return m_values.data();
    }

    constexpr float& operator[](std::size_t _index) {
        return m_values[_index];
    }

    constexpr const float& operator[](std::size_t _index) const {
        return m_values[_index];
    }
};

struct Layer {
    std::size_t m_size;

    explicit Layer(std::size_t _size) : m_size(_size){};

    virtual void forward();
    virtual void backward();
};

struct FeatureTransformer : Layer {
private:
    Tensor m_outputStm;
    Tensor m_outputNstm;

public:
    Features features;

    Tensor weights;
    Tensor bias;

    FeatureTransformer(std::size_t input_size, std::size_t hidden_half) : Layer(2 * hidden_half) {
        weights = Tensor(input_size * hidden_half);
        bias    = Tensor(hidden_half);

        m_outputStm.resize(hidden_half);
        m_outputNstm.resize(hidden_half);
    };

    void loadFeatures(const Features& _features) {
        features = _features;
    }

    void forward() override {
        std::memcpy(m_outputStm.data(), bias.data(), sizeof(float) * m_size);
        std::memcpy(m_outputNstm.data(), bias.data(), sizeof(float) * m_size);

        const uint8_t stm = features.stm;
        const auto hidden_half = m_size / 2;

        for (int i = 0; i < features.n; i++) {
            for (int j = 0; j < hidden_half; j++) {
                m_outputStm[j] += weights[features.features[i][stm] * hidden_half + j];
                m_outputNstm[j] += weights[features.features[i][!stm] * hidden_half + j];
            }
        }
    }

    void backward() override {
        
    }
};

struct Linear : Layer {
    void forward() override {
    }
    void backward() override {
    }
};
