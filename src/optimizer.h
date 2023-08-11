#pragma once

#include "gradient.h"

void adamUpdate(float& v, Gradient& grad, const float gsum, const float lr);
void adamaxUpdate(float& v, Gradient& grad, const float gsum, const float lr);