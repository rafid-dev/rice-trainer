#pragma once

#include <algorithm>
#include <cmath>

class LearningRateScheduler {
public:
    float initial_learning_rate = 0;
};

class StepDecay : public LearningRateScheduler {
public:
    float decay_rate        = 0;
    float min_learning_rate = 0;
    int   step_size         = 0;
    float exponent_base     = 0;

    StepDecay(float initial_learning_rate = 0, float decay_rate = 0, int step_size = 0, float min_learning_rate = 0)
        : decay_rate(decay_rate), step_size(step_size), min_learning_rate(min_learning_rate), exponent_base(std::pow(decay_rate, 1.0f / step_size)) {
        this->initial_learning_rate = initial_learning_rate;
    }

    float get_learning_rate(int epoch) const {
        float decay_factor = std::pow(exponent_base, static_cast<float>(epoch / step_size));
        return std::max(initial_learning_rate * decay_factor, min_learning_rate);
    }

    float get_initial_learning_rate() const {
        return initial_learning_rate;
    }

    float get_min_learning_rate() const {
        return min_learning_rate;
    }
};

class CosineAnnealing : public LearningRateScheduler {
public:
    float max_epochs = 0;
    float min_learning_rate = 0;

    CosineAnnealing(float initial_learning_rate = 0, float max_epochs = 0, float min_learning_rate = 0)
        : max_epochs(max_epochs),
          min_learning_rate(min_learning_rate) {
        initial_learning_rate = initial_learning_rate;
    }

    float get_learning_rate(int epoch) const {
        constexpr float PI = 3.14159265358979323846f;
        float cos_factor = 0.5 * (1 + std::cos(PI * epoch / max_epochs));
        return min_learning_rate + (initial_learning_rate - min_learning_rate) * cos_factor;
    }

    float get_initial_learning_rate() const {
        return initial_learning_rate;
    }

    float get_min_learning_rate() const {
        return min_learning_rate;
    }
};