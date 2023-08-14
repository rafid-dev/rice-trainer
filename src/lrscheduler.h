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
        float decay   = 0;
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
} // namespace LearningRateScheduler