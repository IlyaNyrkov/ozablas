#pragma once

#include <cstdint>
#include "common/crt_tables.hpp"

namespace ozablas {
    namespace pipeline {

#if defined(__CUDACC__) || defined(__HIPCC__)
        // Define the constant memory arrays exactly once without 'extern'.
        // This ensures all kernels and memcpy functions reference the exact same memory block.
        __constant__ uint8_t  c_moduli_all[crt::MAX_SLICES];
        __constant__ uint64_t c_M_prod_20[crt::MAX_SLICES][4];
        __constant__ uint64_t c_M_half_20[crt::MAX_SLICES][4];
        __constant__ uint64_t c_partial_moduli_20[crt::MAX_SLICES][crt::MAX_SLICES][4];
        __constant__ uint64_t c_mod_inv_20[crt::MAX_SLICES][crt::MAX_SLICES];
#endif

    } // namespace pipeline
} // namespace ozablas