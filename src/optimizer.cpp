#include "optimizer.h"
#include "types.h"
#include <algorithm>

void adamUpdate(float& v, Gradient& grad, const float gsum, const float lr) {
    grad.M = BETA1 * grad.M + (1 - BETA1) * gsum;
    grad.V = BETA2 * grad.V + (1 - BETA2) * gsum * gsum;

    v -= lr * grad.M / (sqrt(grad.V) + EPSILON);
}

void adamaxUpdate(float& v, Gradient& grad, const float gsum, const float lr) {
    grad.M = BETA1 * grad.M + (1 - BETA1) * gsum;
    grad.V = std::max(BETA2 * grad.V, std::abs(gsum));

    v -= lr * grad.M / (grad.V + EPSILON);
}