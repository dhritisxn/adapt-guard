// Dataset loading and preprocessing for the Kaggle "Credit Card Fraud Detection" CSV.
#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>

struct Dataset {
    std::vector<std::vector<float>> X; // rows x n_features (Time, V1..V28, Amount)
    std::vector<int> y;                // ground-truth Class label (0=normal, 1=fraud), used ONLY for evaluation
    std::vector<std::string> feature_names;
};

inline Dataset load_creditcard_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Could not open dataset file: " + path);
    }
    Dataset ds;
    std::string line;
    // header
    if (!std::getline(f, line)) throw std::runtime_error("Empty dataset file");
    {
        std::stringstream ss(line);
        std::string col;
        while (std::getline(ss, col, ',')) {
            // strip quotes
            if (!col.empty() && col.front() == '"' && col.back() == '"')
                col = col.substr(1, col.size() - 2);
            ds.feature_names.push_back(col);
        }
        ds.feature_names.pop_back(); // drop "Class" from feature list
    }
    size_t n_cols = ds.feature_names.size() + 1; // +1 for Class
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::vector<float> row;
        row.reserve(n_cols - 1);
        std::stringstream ss(line);
        std::string val;
        size_t idx = 0;
        int cls = 0;
        while (std::getline(ss, val, ',')) {
            if (!val.empty() && val.front() == '"' && val.back() == '"') {
                val = val.substr(1, val.size() - 2);
            }
            if (idx == n_cols - 1) {
                cls = static_cast<int>(std::stof(val));
            } else {
                row.push_back(std::stof(val));
            }
            idx++;
        }
        ds.X.push_back(std::move(row));
        ds.y.push_back(cls);
    }
    return ds;
}

// Standardize selected columns (by index) in-place: x <- (x - mean) / std.
// creditcard.csv's V1..V28 are already PCA-whitened; only Time and Amount need scaling.
inline void standardize_columns(std::vector<std::vector<float>>& X, const std::vector<int>& cols) {
    if (X.empty()) return;
    size_t n = X.size();
    for (int c : cols) {
        double mean = 0.0;
        for (size_t i = 0; i < n; i++) mean += X[i][c];
        mean /= static_cast<double>(n);
        double var = 0.0;
        for (size_t i = 0; i < n; i++) {
            double d = X[i][c] - mean;
            var += d * d;
        }
        var /= static_cast<double>(n);
        double sd = std::sqrt(var);
        if (sd < 1e-12) sd = 1.0;
        for (size_t i = 0; i < n; i++) {
            X[i][c] = static_cast<float>((X[i][c] - mean) / sd);
        }
    }
}

// Removes feature column `col` from every row (and from feature_names).
inline void drop_column(Dataset& ds, int col) {
    for (auto& row : ds.X) row.erase(row.begin() + col);
    ds.feature_names.erase(ds.feature_names.begin() + col);
}
