// Threshold calibration: pick theta as the (1 - contamination) quantile of
// full-ensemble scores over a reference sample, rather than a fixed 0.5 cut.
// Isolation Forest scores don't have a universal "anomalous" cutoff -- 0.5 is
// not meaningful in general -- so every real deployment (and scikit-learn's
// own IsolationForest) picks the threshold from an assumed contamination
// rate. See docs/02_baseline.md for the value used and why.
#pragma once
#include <vector>
#include <algorithm>
#include "isolation_forest.hpp"

inline double calibrate_theta(const IsolationForest& forest,
                               const std::vector<std::vector<float>>& reference_data,
                               double contamination, size_t max_sample = 20000) {
    size_t n = std::min(reference_data.size(), max_sample);
    std::vector<double> scores(n);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) // independent per row: same result, any thread count
#endif
    for (size_t i = 0; i < n; i++) {
        scores[i] = forest.score_full(reference_data[i]);
    }
    std::sort(scores.begin(), scores.end());
    size_t rank = static_cast<size_t>((1.0 - contamination) * static_cast<double>(n));
    rank = std::min(rank, n - 1);
    return scores[rank];
}
