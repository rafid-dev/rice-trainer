#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <immintrin.h>

template <typename T = float>
static inline const T ReLU(const T x) {
    return std::max<T>(0, x);
}

template<int N>
static inline void vectorized_ReLU(float* accumulator) {
    constexpr int    vectorized_size = N / 8;
    __m256 zero            = _mm256_setzero_ps();

    __m256* acc_ptr  = (__m256*) accumulator;
    __m256* acc_ptr2 = (__m256*) (accumulator + N);

    for (int i = 0; i < vectorized_size; i++) {
        acc_ptr[i]  = _mm256_max_ps(acc_ptr[i], zero);
        acc_ptr2[i] = _mm256_max_ps(acc_ptr2[i], zero);
    }
}

template<int N>
static inline float vectorized_DotProduct(float* v1, float* v2) {
  const size_t width = sizeof(__m256) / sizeof(float);
  const size_t chunks = N / width;

  __m256 s0 = _mm256_setzero_ps();
  __m256 s1 = _mm256_setzero_ps();

  __m256* vector1 = (__m256*)v1;
  __m256* vector2 = (__m256*)v2;

  for (size_t j = 0; j < chunks; j += 2) {
    s0 = _mm256_add_ps(_mm256_mul_ps(vector1[j], vector2[j]), s0);
    s1 = _mm256_add_ps(_mm256_mul_ps(vector1[j + 1], vector2[j + 1]), s1);
  }

  const __m256 r8 = _mm256_add_ps(s0, s1);
  const __m128 r4 = _mm_add_ps(_mm256_castps256_ps128(r8), _mm256_extractf128_ps(r8, 1));
  const __m128 r2 = _mm_add_ps(r4, _mm_movehl_ps(r4, r4));
  const __m128 r1 = _mm_add_ss(r2, _mm_shuffle_ps(r2, r2, 0x1));
  return _mm_cvtss_f32(r1);
}

template <typename T = float>
static inline const T ReLUPrime(const T x) {
    return x > 0 ? 1 : 0;
}

template <typename T = float>
static inline const T sigmoid(const T x) {
    return 1 / (1 + expf(-x));
}

template <typename T = float>
static inline const T sigmoidPrime(const T x) {
    T sigmoid_x = sigmoid(x);
    return sigmoid_x * (1 - sigmoid_x);
}

static inline int inputIndex(uint8_t pieceType, uint8_t pieceColor, int square, uint8_t view, int kingSquare) {
    square = square ^ (56 * view);
    square = square ^ (7 * !!(kingSquare & 0x4));

    // clang-format off
    return square
           + pieceType * 64
           + !(pieceColor ^ view) * 64 * 6;
    // clang-format on
}