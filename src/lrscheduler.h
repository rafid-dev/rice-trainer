#pragma once

#include <algorithm>
#include <cmath>

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
        int interval = 0;
        float learningRate = 0;
        float decay = 0.5;

        StepDecay(float initial_learning_rate, int interval, float decay) {
            this->initial_learning_rate = initial_learning_rate;
            this->learningRate = initial_learning_rate;
            this->interval = interval;
            this->decay = decay;
        }

        void step() {
            steps++;
            if (steps % interval == 0) {
                learningRate *= decay;
            }
        }

        float getLearningRate() {
            return learningRate;
        }
    };
} // namespace LearningRateScheduler