// Small helpers for timing and detection-quality metrics.
#pragma once
#include <chrono>
#include <vector>
#include <cstdio>

class Timer {
public:
    void start() { t0_ = std::chrono::steady_clock::now(); }
    double elapsed_ms() const {
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0_).count();
    }
private:
    std::chrono::steady_clock::time_point t0_;
};

struct ConfusionCounts {
    long tp = 0, fp = 0, fn = 0, tn = 0;
    double precision() const { return (tp + fp) == 0 ? 0.0 : static_cast<double>(tp) / (tp + fp); }
    double recall() const { return (tp + fn) == 0 ? 0.0 : static_cast<double>(tp) / (tp + fn); }
    double f1() const {
        double p = precision(), r = recall();
        return (p + r) == 0.0 ? 0.0 : 2 * p * r / (p + r);
    }
};

inline ConfusionCounts compute_confusion(const std::vector<int>& true_label, const std::vector<int>& pred_label) {
    ConfusionCounts c;
    for (size_t i = 0; i < true_label.size(); i++) {
        if (pred_label[i] == 1 && true_label[i] == 1) c.tp++;
        else if (pred_label[i] == 1 && true_label[i] == 0) c.fp++;
        else if (pred_label[i] == 0 && true_label[i] == 1) c.fn++;
        else c.tn++;
    }
    return c;
}

inline double compute_agreement(const std::vector<int>& a, const std::vector<int>& b) {
    long match = 0;
    for (size_t i = 0; i < a.size(); i++) if (a[i] == b[i]) match++;
    return static_cast<double>(match) / static_cast<double>(a.size());
}
