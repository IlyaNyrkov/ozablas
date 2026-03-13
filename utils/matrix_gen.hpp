#pragma once

#include <vector>
#include <random>
#include <cmath>

namespace matrix_utils {
    namespace generation {

        // -----------------------------------------------------------------
        // Generate 1D Dense Matrix (Row-Major) using Ozaki Scheme II formula
        // a_ij, b_ij = (rand - 0.5) * exp(phi * randn)
        // -----------------------------------------------------------------
        template <typename T>
        std::vector<T> generate_ozaki_matrix(int rows, int cols, T phi = 0.5, int seed = 42) {
            std::mt19937 rng(seed);

            // std::uniform_real_distribution generates in [0, 1).
            // We do 1.0 - val to shift it strictly to (0, 1] as defined in Ozaki I&II papers.
            std::uniform_real_distribution<T> unif(0.0, 1.0);

            // Standard normal distribution (mean 0.0, stddev 1.0)
            std::normal_distribution<T> norm(0.0, 1.0);

            std::vector<T> mat(rows * cols);
            for (int i = 0; i < rows * cols; ++i) {
                T rand_val = static_cast<T>(1.0) - unif(rng);
                T randn_val = norm(rng);

                mat[i] = (rand_val - static_cast<T>(0.5)) * std::exp(phi * randn_val);
            }

            return mat;
        }

    } // namespace generation
} // namespace matrix_utils
