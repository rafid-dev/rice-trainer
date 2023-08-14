#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>

namespace LearningRateScheduler {
    class LearningRateScheduler {
    public:
        float initial_learning_rate = 0;
        int   steps                 = 0;

        friend std::ostream& operator<<(std::ostream& os, const LearningRateScheduler& learning_rate_scheduler) {
            os << "LearningRateScheduler(initial_learning_rate=" << learning_rate_scheduler.initial_learning_rate << ")";
            return os;
        }
    };

    class StepDecay : public LearningRateScheduler {
    public:
        float decay    = 0;
        int   interval = 0;
        StepDecay(float initial_learning_rate, float decay, int interval) {
            this->initial_learning_rate = initial_learning_rate;
            this->decay                 = decay;
            this->interval              = interval;
        }

        void step(float& learningRate) {
            steps++;
            learningRate = initial_learning_rate * std::pow(decay, std::floor(steps / interval));
        }

        friend std::ostream& operator<<(std::ostream& os, const StepDecay& step_decay) {
            os << "StepDecay(initial_learning_rate=" << step_decay.initial_learning_rate << ", decay=" << step_decay.decay << ", interval=" << step_decay.interval << ")";
            return os;
        }
    };

    class CosineAnnealing : public LearningRateScheduler {
    public:
        float max_epochs = 0;
        CosineAnnealing(float initial_learning_rate, float max_epochs) {
            this->initial_learning_rate = initial_learning_rate;
            this->max_epochs            = max_epochs;
        }

        void step(float& learningRate, int epoch) {
            steps++;
            constexpr float PI         = 3.14159265358979323846f;
            float           cos_factor = 0.5 * (1 + std::cos(PI * steps / max_epochs));
            learningRate               = initial_learning_rate * cos_factor;
        }

        friend std::ostream& operator<<(std::ostream& os, const CosineAnnealing& cosine_annealing) {
            os << "CosineAnnealing(initial_learning_rate=" << cosine_annealing.initial_learning_rate << ", max_epochs=" << cosine_annealing.max_epochs << ")";
            return os;
        }
    };

    class ExponentialDecay : public LearningRateScheduler {
    public:
        float decay = 0;
        ExponentialDecay(float initial_learning_rate, float decay) {
            this->initial_learning_rate = initial_learning_rate;
            this->decay                 = decay;
        }

        void step(float& learningRate) {
            steps++;
            learningRate = initial_learning_rate * std::exp(-decay * steps);
        }

        friend std::ostream& operator<<(std::ostream& os, const ExponentialDecay& exponential_decay) {
            os << "ExponentialDecay(initial_learning_rate=" << exponential_decay.initial_learning_rate << ", decay=" << exponential_decay.decay << ")";
            return os;
        }
    };

    class CyclicalLearningRates : public LearningRateScheduler {
    public:
        float base_lr    = 0;
        float max_lr     = 0;
        int   step_size  = 0;
        int   cycle_mult = 0;

        CyclicalLearningRates(float base_lr, float max_lr, int step_size, int cycle_mult = 1) {
            this->base_lr    = base_lr;
            this->max_lr     = max_lr;
            this->step_size  = step_size;
            this->cycle_mult = cycle_mult;
        }

        void step(float& learningRate) {
            steps++;
            int   cycle  = std::floor(1 + steps / (2 * step_size));
            float x      = std::abs(steps / step_size - 2 * cycle + 1);
            learningRate = base_lr + (max_lr - base_lr) * std::max(0.0f, (1 - x));
        }

        friend std::ostream& operator<<(std::ostream& os, const CyclicalLearningRates& clr) {
            os << "CyclicalLearningRates(base_lr=" << clr.base_lr << ", max_lr=" << clr.max_lr << ", step_size=" << clr.step_size << ", cycle_mult=" << clr.cycle_mult << ")";
            return os;
        }
    };
} // namespace LearningRateScheduler