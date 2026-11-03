// Unit tests for the shared core (src/common/). Dependency-free: each CHECK
// failure is printed, and the exit code is the number of failures.
// Run with `make test`. Uses a small synthetic dataset, not creditcard.csv.
#include <cstdio>
#include <cmath>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "../src/common/isolation_forest.hpp"
#include "../src/common/calibrate.hpp"
#include "../src/common/dataset.hpp"
#include "../src/common/metrics.hpp"
#include "../src/common/split.hpp"
#include "../src/common/args.hpp"

static int g_failures = 0;
static int g_checks = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        g_checks++;                                                              \
        if (!(cond)) {                                                           \
            g_failures++;                                                        \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);          \
        }                                                                        \
    } while (0)

static bool near(double a, double b, double tol = 1e-9) { return std::fabs(a - b) <= tol; }

// 2,000 points from a standard normal cloud in 5-D, plus 10 far outliers.
static std::vector<std::vector<float>> make_data(int n_normal = 2000, int n_out = 10) {
    std::mt19937_64 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<std::vector<float>> X;
    for (int i = 0; i < n_normal; i++) {
        std::vector<float> r(5);
        for (auto& v : r) v = nd(rng);
        X.push_back(r);
    }
    for (int i = 0; i < n_out; i++) X.push_back(std::vector<float>(5, 8.0f + i));
    return X;
}

static IsolationForest make_forest(const std::vector<std::vector<float>>& X, int n_trees = 100) {
    IsolationForest f;
    f.init(5, 256);
    f.build_all(X, n_trees, 42);
    return f;
}

static void test_c_factor() {
    CHECK(near(IsolationForest::c_factor(1.0), 0.0));
    CHECK(near(IsolationForest::c_factor(2.0), 1.0));
    // c(256) = 2*(ln 255 + gamma) - 2*255/256 ~= 10.2448
    CHECK(near(IsolationForest::c_factor(256.0), 10.2448, 1e-3));
    IsolationForest f;
    f.init(5, 256);
    CHECK(f.max_depth == 8);
    CHECK(near(f.score_to_anomaly(f.c_psi), 0.5)); // E[h] = c(psi) -> score 0.5 by construction
}

static void test_tree_structure() {
    auto X = make_data();
    IsolationForest f = make_forest(X, 10);
    for (const auto& tree : f.trees) {
        CHECK(tree.nodes[0].size == 256); // root holds the whole subsample
        for (const auto& n : tree.nodes) {
            if (n.feature == -1) continue;
            CHECK(n.left > 0 && n.right > 0);
            CHECK(tree.nodes[n.left].size + tree.nodes[n.right].size == n.size);
        }
    }
}

static void test_deterministic_and_order_independent() {
    auto X = make_data();
    IsolationForest a = make_forest(X);
    // Build in reverse order into a pre-sized forest (what OpenMP/MPI do).
    IsolationForest b;
    b.init(5, 256);
    b.prepare_trees(100);
    for (int t = 99; t >= 0; t--) b.trees[t] = b.build_one_tree(X, 42 + t);
    bool same = true;
    for (int t = 0; t < 100; t++) {
        const auto& na = a.trees[t].nodes;
        const auto& nb = b.trees[t].nodes;
        if (na.size() != nb.size()) { same = false; break; }
        for (size_t k = 0; k < na.size(); k++) {
            if (na[k].feature != nb[k].feature || na[k].split_value != nb[k].split_value ||
                na[k].left != nb[k].left || na[k].right != nb[k].right || na[k].size != nb[k].size) same = false;
        }
    }
    CHECK(same);
    CHECK(a.score_full(X[0]) == b.score_full(X[0]));
}

static void test_outliers_score_higher() {
    auto X = make_data();
    IsolationForest f = make_forest(X);
    double normal_mean = 0.0;
    for (int i = 0; i < 2000; i++) normal_mean += f.score_full(X[i]);
    normal_mean /= 2000;
    for (int i = 2000; i < 2010; i++) CHECK(f.score_full(X[i]) > normal_mean + 0.15);
    CHECK(f.score_full(X[2009]) > 0.6);
}

static void test_adaptive_matches_full_when_disabled() {
    auto X = make_data();
    IsolationForest f = make_forest(X);
    double theta = calibrate_theta(f, X, 0.01);
    for (int i = 0; i < 2010; i += 37) {
        // min_trees == n_trees: early exit can never trigger
        auto r = f.score_adaptive(X[i], theta, 2.0, 100);
        CHECK(r.trees_used == 100);
        CHECK(!r.early_exit);
        // Welford's running mean rounds differently from sum/n, so equal only
        // to ~1 ulp (the binaries call score_full directly when not adaptive).
        CHECK(near(r.score, f.score_full(X[i]), 1e-12));
    }
}

static void test_adaptive_exits_early_and_agrees() {
    auto X = make_data();
    IsolationForest f = make_forest(X);
    double theta = calibrate_theta(f, X, 0.01);
    long used = 0;
    int agree = 0;
    for (const auto& x : X) {
        auto r = f.score_adaptive(x, theta, 2.0, 10);
        CHECK(r.trees_used >= 10 && r.trees_used <= 100);
        used += r.trees_used;
        agree += (r.label == (f.score_full(x) > theta ? 1 : 0));
    }
    CHECK(static_cast<double>(used) / X.size() < 40.0); // most points exit well before 100
    CHECK(static_cast<double>(agree) / X.size() > 0.98);
    // The far outlier should be confidently flagged as soon as allowed.
    auto r = f.score_adaptive(X[2009], theta, 2.0, 10);
    CHECK(r.label == 1 && r.early_exit);
}

static void test_explain_consistency() {
    auto X = make_data();
    IsolationForest f = make_forest(X);
    double theta = calibrate_theta(f, X, 0.01);
    auto a = f.score_adaptive(X[100], theta, 2.0, 10);
    auto e = f.score_adaptive_explain(X[100], theta, 2.0, 10);
    CHECK(a.score == e.score && a.trees_used == e.trees_used && a.label == e.label);
    CHECK(e.contrib.size() == 5);
    // With early exit disabled, integrated and post-hoc contributions agree.
    auto full = f.score_adaptive_explain(X[100], theta, 2.0, 100);
    auto post = f.explain_posthoc(X[100]);
    for (int k = 0; k < 5; k++) CHECK(near(full.contrib[k], post[k], 1e-12));
    // Root split alone contributes 1.0 per tree, so the sum is at least 1.
    double sum = 0.0;
    for (double c : post) sum += c;
    CHECK(sum >= 1.0);
}

static void test_calibrate_quantile() {
    auto X = make_data();
    IsolationForest f = make_forest(X);
    double theta = calibrate_theta(f, X, 0.05);
    int above = 0;
    for (const auto& x : X) above += f.score_full(x) > theta;
    double frac = static_cast<double>(above) / X.size();
    CHECK(frac <= 0.05 && frac > 0.04);
}

static void test_split_and_metrics() {
    SplitIndices s = split_train_stream(10, 0.3);
    CHECK(s.train_pool.size() == 3 && s.stream.size() == 7);
    CHECK(s.train_pool.back() == 2 && s.stream.front() == 3); // chronological, no shuffle

    ConfusionCounts c = compute_confusion({1, 1, 0, 0, 1}, {1, 0, 0, 1, 1});
    CHECK(c.tp == 2 && c.fn == 1 && c.fp == 1 && c.tn == 1);
    CHECK(near(c.precision(), 2.0 / 3) && near(c.recall(), 2.0 / 3) && near(c.f1(), 2.0 / 3));
    CHECK(near(compute_agreement({1, 0, 1, 0}, {1, 1, 1, 0}), 0.75));
}

static void test_dataset_loader(const std::string& tmp_path) {
    {
        std::ofstream f(tmp_path);
        f << "\"Time\",\"V1\",\"Amount\",\"Class\"\n";
        f << "0,1.5,10,\"0\"\n";
        f << "1,-0.5,30,\"1\"\n";
    }
    Dataset ds = load_creditcard_csv(tmp_path);
    CHECK(ds.feature_names.size() == 3 && ds.feature_names[0] == "Time");
    CHECK(ds.X.size() == 2 && ds.y[0] == 0 && ds.y[1] == 1);
    CHECK(near(ds.X[1][1], -0.5));
    standardize_columns(ds.X, {2});
    CHECK(near(ds.X[0][2], -1.0, 1e-6) && near(ds.X[1][2], 1.0, 1e-6));
    drop_column(ds, 0);
    CHECK(ds.feature_names.size() == 2 && ds.X[0].size() == 2 && ds.feature_names[0] == "V1");
    std::remove(tmp_path.c_str());
}

static void test_args() {
    const char* argv[] = {"prog", "--trees", "50", "--adaptive", "--z", "1.5", "--score-n", "-1"};
    Args a(8, const_cast<char**>(argv));
    CHECK(a.get_int("trees", 100) == 50);
    CHECK(a.has("adaptive") && !a.has("explain"));
    CHECK(near(a.get_double("z", 2.0), 1.5));
    CHECK(a.get_int("score-n", 0) == -1); // negative values are values, not flags
    CHECK(a.get_int("psi", 256) == 256);
}

int main() {
    test_c_factor();
    test_tree_structure();
    test_deterministic_and_order_independent();
    test_outliers_score_higher();
    test_adaptive_matches_full_when_disabled();
    test_adaptive_exits_early_and_agrees();
    test_explain_consistency();
    test_calibrate_quantile();
    test_split_and_metrics();
    test_dataset_loader("/tmp/adaptguard_test_dataset.csv");
    test_args();
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures;
}
