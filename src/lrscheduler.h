#pragma once

#include <algorithm>
#include <cmath>

namespace LearningRateScheduler {
    class LearningRateScheduler {
    public:
        float initial_learning_rate = 0;

        friend std::ostream& operator<<(std::ostream& os, const LearningRateScheduler& learning_rate_scheduler) {
            os << "LearningRateScheduler(initial_learning_rate=" << learning_rate_scheduler.initial_learning_rate << ")";
            return os;
        }
    };

    class StepDecay : public LearningRateScheduler {
    public:
        float decay_rate = 0;
        float min_learning_rate = 0;
        int step_size = 0;
        float exponent_base = 0;

        StepDecay(float _initial_learning_rate = 0, float _decay_rate = 0, int _step_size = 0, float _min_learning_rate = 0)
            : decay_rate(_decay_rate), step_size(_step_size), min_learning_rate(_min_learning_rate), exponent_base(std::pow(_decay_rate, 1.0f / _step_size)) {
            initial_learning_rate = _initial_learning_rate;
        }

        float get_learning_rate(int _epoch) const {
            float decay_factor = std::pow(exponent_base, static_cast<float>(_epoch / step_size));
            return std::max(initial_learning_rate * decay_factor, min_learning_rate);
        }

        float get_initial_learning_rate() const {
            return initial_learning_rate;
        }

        float get_min_learning_rate() const {
            return min_learning_rate;
        }

        friend std::ostream& operator<<(std::ostream& os, const StepDecay& step_decay) {
            os << "StepDecay(initial_learning_rate=" << step_decay.initial_learning_rate << ", decay_rate=" << step_decay.decay_rate << ", step_size=" << step_decay.step_size << ", min_learning_rate=" << step_decay.min_learning_rate << ")";
            return os;
        }
    };

    class CosineAnnealing : public LearningRateScheduler {
    public:
        float max_epochs = 0;
        float min_learning_rate = 0;

        CosineAnnealing(float _initial_learning_rate = 0, float _max_epochs = 0, float _min_learning_rate = 0)
            : max_epochs(_max_epochs), min_learning_rate(_min_learning_rate) {
            initial_learning_rate = _initial_learning_rate;
        }

        float get_learning_rate(int _epoch) const {
            constexpr float PI = 3.14159265358979323846f;
            float cos_factor = 0.5 * (1 + std::cos(PI * _epoch / max_epochs));
            return min_learning_rate + (initial_learning_rate - min_learning_rate) * cos_factor;
        }

        float get_initial_learning_rate() const {
            return initial_learning_rate;
        }

        float get_min_learning_rate() const {
            return min_learning_rate;
        }

        friend std::ostream& operator<<(std::ostream& os, const CosineAnnealing& cosine_annealing) {
            os << "CosineAnnealing(initial_learning_rate=" << cosine_annealing.initial_learning_rate << ", max_epochs=" << cosine_annealing.max_epochs << ", min_learning_rate=" << cosine_annealing.min_learning_rate << ")";
            return os;
        }
    };
} // namespace LearningRateScheduler