// Core Isolation Forest implementation shared by serial, OpenMP, and MPI binaries.
//
// Design notes (see docs/02_baseline.md and docs/04_adaptive_early_exit.md):
//  - Each tree is seeded independently as seed_base + tree_index, so tree i is
//    bit-identical no matter which thread/rank builds it or in what order.
//    This is what makes serial/OpenMP/MPI outputs directly comparable.
//  - Trees are stored as a flat vector<Node> (index-based left/right) rather
//    than pointer-linked nodes, for cache locality and trivial (de)serialization.
#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <cstdint>
#include <algorithm>
#include <limits>

struct IFNode {
    int feature = -1;     // -1 marks a leaf
    float split_value = 0.0f;
    int left = -1;
    int right = -1;
    int size = 0;         // number of training points that reached this node (leaves use this for the c(size) correction)
};

struct IFTree {
    std::vector<IFNode> nodes; // nodes[0] is the root
};

class IsolationForest {
public:
    int n_features = 0;
    int subsample_size = 256;   // psi
    int max_depth = 0;          // ceil(log2(psi)), the standard IF depth limit
    double c_psi = 1.0;         // normalization constant c(psi)
    std::vector<IFTree> trees;

    static double harmonic_approx(double n) {
        // H(n) ~= ln(n) + gamma  (Euler-Mascheroni), accurate to <1e-4 for n>=8,
        // matches the approximation used in the original Isolation Forest paper.
        static const double gamma = 0.5772156649015329;
        if (n <= 0) return 0.0;
        return std::log(n) + gamma;
    }

    static double c_factor(double n) {
        if (n <= 1.0) return 0.0;
        if (n == 2.0) return 1.0;
        return 2.0 * harmonic_approx(n - 1.0) - (2.0 * (n - 1.0) / n);
    }

    void init(int n_feat, int psi) {
        n_features = n_feat;
        subsample_size = psi;
        max_depth = static_cast<int>(std::ceil(std::log2(std::max(2, psi))));
        c_psi = c_factor(static_cast<double>(psi));
    }

    // Builds a single tree deterministically from `seed`. `data` is the full
    // dataset; a subsample of `subsample_size` row indices is drawn from it.
    IFTree build_one_tree(const std::vector<std::vector<float>>& data, uint64_t seed) const {
        std::mt19937_64 rng(seed);
        int n = static_cast<int>(data.size());
        int psi = std::min(subsample_size, n);
        std::vector<int> idx(n);
        for (int i = 0; i < n; i++) idx[i] = i;
        // partial Fisher-Yates: only need the first `psi` shuffled entries
        for (int i = 0; i < psi; i++) {
            std::uniform_int_distribution<int> d(i, n - 1);
            int j = d(rng);
            std::swap(idx[i], idx[j]);
        }
        idx.resize(psi);

        IFTree tree;
        tree.nodes.reserve(2 * psi); // upper bound on node count for a binary tree over psi points
        build_node(tree, data, idx, 0, rng);
        return tree;
    }

    // Recursively builds a node from the given subset of row indices and
    // returns its index in tree.nodes.
    int build_node(IFTree& tree, const std::vector<std::vector<float>>& data,
                    std::vector<int>& idx, int depth, std::mt19937_64& rng) const {
        int node_idx = static_cast<int>(tree.nodes.size());
        tree.nodes.emplace_back();
        tree.nodes[node_idx].size = static_cast<int>(idx.size());

        if (depth >= max_depth || idx.size() <= 1) {
            return node_idx; // leaf (feature stays -1)
        }

        // Pick a feature that actually varies in this subset (retry a bounded
        // number of times; if none vary, fall back to a leaf).
        std::uniform_int_distribution<int> feat_dist(0, n_features - 1);
        int feature = -1;
        float split_value = 0.0f;
        for (int attempt = 0; attempt < 8; attempt++) {
            int f = feat_dist(rng);
            float mn = std::numeric_limits<float>::max();
            float mx = std::numeric_limits<float>::lowest();
            for (int i : idx) {
                float v = data[i][f];
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
            if (mx > mn) {
                std::uniform_real_distribution<float> split_dist(mn, mx);
                feature = f;
                split_value = split_dist(rng);
                break;
            }
        }
        if (feature == -1) {
            return node_idx; // no feature varies -> can't split further, leaf
        }

        std::vector<int> left_idx, right_idx;
        left_idx.reserve(idx.size());
        right_idx.reserve(idx.size());
        for (int i : idx) {
            if (data[i][feature] < split_value) left_idx.push_back(i);
            else right_idx.push_back(i);
        }
        // Degenerate split (shouldn't normally happen since mx>mn) -> leaf.
        if (left_idx.empty() || right_idx.empty()) {
            return node_idx;
        }

        int left_child = build_node(tree, data, left_idx, depth + 1, rng);
        int right_child = build_node(tree, data, right_idx, depth + 1, rng);
        // tree.nodes may have reallocated; re-fetch reference by index.
        tree.nodes[node_idx].feature = feature;
        tree.nodes[node_idx].split_value = split_value;
        tree.nodes[node_idx].left = left_child;
        tree.nodes[node_idx].right = right_child;
        return node_idx;
    }

    // Pre-sizes trees so parallel callers (OpenMP/MPI) can write trees[t]
    // from independent iterations without any reallocation race.
    void prepare_trees(int n_trees) { trees.assign(n_trees, IFTree()); }

    // Builds trees [0, n_trees) into this->trees, each seeded by seed_base + i.
    // Callers doing parallel construction (OpenMP) just need to run the loop
    // body in parallel; the seeding scheme makes the result order-independent.
    void build_all(const std::vector<std::vector<float>>& data, int n_trees, uint64_t seed_base) {
        trees.assign(n_trees, IFTree());
        for (int t = 0; t < n_trees; t++) {
            trees[t] = build_one_tree(data, seed_base + static_cast<uint64_t>(t));
        }
    }

    double path_length(const IFTree& tree, const std::vector<float>& x) const {
        int node_idx = 0;
        int depth = 0;
        while (true) {
            const IFNode& node = tree.nodes[node_idx];
            if (node.feature == -1) {
                return depth + c_factor(static_cast<double>(node.size));
            }
            node_idx = (x[node.feature] < node.split_value) ? node.left : node.right;
            depth++;
        }
    }

    // Same traversal, but also accumulates a per-feature "isolation
    // contribution" as a byproduct (see docs/05_explainability.md): each
    // internal node visited adds weight 1/(depth+1) to contrib[feature],
    // so splits made earlier (shallower) on the path -- which isolate the
    // point faster and drive the anomaly score more -- count for more.
    double path_length_contrib(const IFTree& tree, const std::vector<float>& x,
                                std::vector<double>& contrib) const {
        int node_idx = 0;
        int depth = 0;
        while (true) {
            const IFNode& node = tree.nodes[node_idx];
            if (node.feature == -1) {
                return depth + c_factor(static_cast<double>(node.size));
            }
            contrib[node.feature] += 1.0 / static_cast<double>(depth + 1);
            node_idx = (x[node.feature] < node.split_value) ? node.left : node.right;
            depth++;
        }
    }

    double score_to_anomaly(double mean_path_length) const {
        return std::pow(2.0, -mean_path_length / c_psi);
    }

    // Full-ensemble score: average path length over ALL trees. This is the
    // reference computation every other mode is validated against.
    double score_full(const std::vector<float>& x) const {
        double sum = 0.0;
        for (const auto& tree : trees) sum += path_length(tree, x);
        return score_to_anomaly(sum / static_cast<double>(trees.size()));
    }

    struct AdaptiveResult {
        double score;
        int trees_used;
        int label;       // 1 = anomaly, 0 = normal
        bool early_exit;
    };

    // Sequential confidence-bound early exit (the project's novelty piece;
    // full derivation in docs/04_adaptive_early_exit.md):
    //   After each tree, update the running mean/variance of path length
    //   (Welford's algorithm). Once >= min_trees trees have been evaluated,
    //   compute a z*SE confidence band around the running score estimate.
    //   If the whole band lies on one side of `theta`, stop -- the label is
    //   already decided within the requested confidence, further trees are
    //   near-certain to be wasted computation on a clearly-normal (or
    //   clearly-anomalous) point.
    AdaptiveResult score_adaptive(const std::vector<float>& x, double theta,
                                   double z, int min_trees) const {
        double mean = 0.0, m2 = 0.0;
        int k = 0;
        int n_trees = static_cast<int>(trees.size());
        for (int t = 0; t < n_trees; t++) {
            double h = path_length(trees[t], x);
            k++;
            double delta = h - mean;
            mean += delta / k;
            double delta2 = h - mean;
            m2 += delta * delta2;

            if (k >= min_trees && k < n_trees) {
                double variance = m2 / (k - 1);
                double se = std::sqrt(variance / k);
                double score_upper = score_to_anomaly(mean - z * se); // optimistic (most anomalous) bound
                double score_lower = score_to_anomaly(mean + z * se); // pessimistic (least anomalous) bound
                if (score_lower > theta) {
                    return {score_to_anomaly(mean), k, 1, true};
                }
                if (score_upper < theta) {
                    return {score_to_anomaly(mean), k, 0, true};
                }
            }
        }
        double final_score = score_to_anomaly(mean);
        return {final_score, k, final_score > theta ? 1 : 0, false};
    }

    struct AdaptiveExplainResult {
        double score;
        int trees_used;
        int label;
        bool early_exit;
        std::vector<double> contrib; // per-feature contribution, normalized by trees_used
    };

    // Same early-exit logic as score_adaptive, but reuses the traversal to
    // accumulate feature contributions at (near) zero extra cost -- no
    // separate explanation pass is needed. This is the integrated path
    // measured in the explainability latency benchmark.
    AdaptiveExplainResult score_adaptive_explain(const std::vector<float>& x, double theta,
                                                  double z, int min_trees) const {
        double mean = 0.0, m2 = 0.0;
        int k = 0;
        int n_trees = static_cast<int>(trees.size());
        std::vector<double> contrib(n_features, 0.0);
        for (int t = 0; t < n_trees; t++) {
            double h = path_length_contrib(trees[t], x, contrib);
            k++;
            double delta = h - mean;
            mean += delta / k;
            double delta2 = h - mean;
            m2 += delta * delta2;

            bool stop = false;
            int label = 0;
            double cur_score = score_to_anomaly(mean);
            if (k >= min_trees && k < n_trees) {
                double variance = m2 / (k - 1);
                double se = std::sqrt(variance / k);
                double score_upper = score_to_anomaly(mean - z * se);
                double score_lower = score_to_anomaly(mean + z * se);
                if (score_lower > theta) { stop = true; label = 1; }
                else if (score_upper < theta) { stop = true; label = 0; }
            }
            if (stop) {
                for (auto& c : contrib) c /= static_cast<double>(k);
                return {cur_score, k, label, true, contrib};
            }
        }
        double final_score = score_to_anomaly(mean);
        for (auto& c : contrib) c /= static_cast<double>(k);
        return {final_score, k, final_score > theta ? 1 : 0, false, contrib};
    }

    // Naive post-hoc explanation: a FULL separate traversal of every tree,
    // purely to accumulate feature contributions, ignoring any early exit.
    // Used only to measure the latency cost of the "separate pass" approach
    // this project's design avoids. See docs/05_explainability.md.
    std::vector<double> explain_posthoc(const std::vector<float>& x) const {
        std::vector<double> contrib(n_features, 0.0);
        for (const auto& tree : trees) {
            path_length_contrib(tree, x, contrib);
        }
        for (auto& c : contrib) c /= static_cast<double>(trees.size());
        return contrib;
    }
};
