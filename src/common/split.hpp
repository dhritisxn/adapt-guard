// Chronological train/stream split. creditcard.csv is already ordered by
// ascending Time, so a prefix/suffix split simulates "train on history,
// score the live stream" without shuffling (see docs/01_dataset.md).
#pragma once
#include <vector>
#include <cstddef>

struct SplitIndices {
    std::vector<int> train_pool;  // row indices used to draw tree subsamples from
    std::vector<int> stream;      // row indices replayed as the incoming stream
};

inline SplitIndices split_train_stream(size_t n_rows, double train_frac) {
    SplitIndices s;
    size_t n_train = static_cast<size_t>(n_rows * train_frac);
    s.train_pool.reserve(n_train);
    s.stream.reserve(n_rows - n_train);
    for (size_t i = 0; i < n_rows; i++) {
        if (i < n_train) s.train_pool.push_back(static_cast<int>(i));
        else s.stream.push_back(static_cast<int>(i));
    }
    return s;
}
