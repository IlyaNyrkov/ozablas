// src/common/crt_math.hpp
#pragma once

#include <cstdint>
#include <cmath>

// Safely allow both CPU tests and GPU kernels to use this math
#if defined(__CUDACC__) || defined(__HIPCC__)
    #define OZA_HOST_DEVICE __host__ __device__
#else
    #define OZA_HOST_DEVICE
#endif

namespace ozablas {
namespace crt {

/**
 * @brief A 256-bit unsigned integer represented by four 64-bit limbs.
 */
struct uint256_t {
    uint64_t data[4];

    OZA_HOST_DEVICE uint256_t() {
        data[0] = 0; data[1] = 0; data[2] = 0; data[3] = 0;
    }

    OZA_HOST_DEVICE uint256_t(uint64_t l0, uint64_t l1, uint64_t l2, uint64_t l3) {
        data[0] = l0; data[1] = l1; data[2] = l2; data[3] = l3;
    }

    // Construct from the constexpr arrays
    OZA_HOST_DEVICE uint256_t(const uint64_t arr[4]) {
        data[0] = arr[0]; data[1] = arr[1]; data[2] = arr[2]; data[3] = arr[3];
    }
};

// 256-bit safe subtraction (a = a - b)
OZA_HOST_DEVICE inline void sub_256(uint256_t& a, const uint256_t& b) {
    uint64_t borrow = 0;
    for(int i = 0; i < 4; i++) {
        uint64_t sub1 = a.data[i] - borrow;
        uint64_t b1 = (a.data[i] < borrow) ? 1 : 0;
        uint64_t diff = sub1 - b.data[i];
        uint64_t b2 = (sub1 < b.data[i]) ? 1 : 0;
        a.data[i] = diff;
        borrow = b1 | b2;
    }
}

// 256-bit safe addition (a = a + b)
OZA_HOST_DEVICE inline void add_256(uint256_t& a, const uint256_t& b) {
    uint64_t carry = 0;
    for(int i = 0; i < 4; i++) {
        uint64_t sum1 = a.data[i] + carry;
        uint64_t c1 = (sum1 < a.data[i]) ? 1 : 0;
        uint64_t sum = sum1 + b.data[i];
        uint64_t c2 = (sum < sum1) ? 1 : 0;
        a.data[i] = sum;
        carry = c1 | c2;
    }
}

// 256-bit greater-than-or-equal comparison
OZA_HOST_DEVICE inline bool cmp_ge_256(const uint256_t& a, const uint256_t& b) {
    for(int i = 3; i >= 0; i--) {
        if (a.data[i] > b.data[i]) return true;
        if (a.data[i] < b.data[i]) return false;
    }
    return true; // They are equal
}

// Convert 256-bit int to double
OZA_HOST_DEVICE inline double to_double_256(const uint256_t& a) {
    double res = 0.0;
    res += static_cast<double>(a.data[0]);
    res += ldexp(static_cast<double>(a.data[1]), 64);
    res += ldexp(static_cast<double>(a.data[2]), 128);
    res += ldexp(static_cast<double>(a.data[3]), 192);
    return res;
}

} // namespace crt
} // namespace ozablas