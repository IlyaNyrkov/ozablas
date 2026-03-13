#pragma once

#include <cstdint>
#include "common/crt_tables.hpp"

#if defined(__CUDACC__) || defined(__HIPCC__)
__constant__ uint8_t  OZA_c_moduli_all[ozablas::crt::MAX_SLICES];
__constant__ uint64_t OZA_c_M_prod_20[ozablas::crt::MAX_SLICES][4];
__constant__ uint64_t OZA_c_M_half_20[ozablas::crt::MAX_SLICES][4];
__constant__ uint64_t OZA_c_partial_moduli_20[ozablas::crt::MAX_SLICES][ozablas::crt::MAX_SLICES][4];
__constant__ uint64_t OZA_c_mod_inv_20[ozablas::crt::MAX_SLICES][ozablas::crt::MAX_SLICES];
#endif