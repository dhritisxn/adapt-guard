// OpenMP Isolation Forest: parallel tree construction + parallel per-
// transaction scoring on a shared-memory node. Same core algorithm as the
// serial baseline (src/serial/main.cpp) -- only the two hot loops (tree
// build, transaction scoring) are parallelized. Deterministic per-tree
// seeding (seed_base + tree_index) makes the forest, and therefore every
// score, bit-identical to the serial run regardless of thread scheduling.
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <omp.h>

#include "../common/dataset.hpp"
#include "../common/isolation_forest.hpp"
#include "../common/args.hpp"
#include "../common/metrics.hpp"
#include "../common/split.hpp"
#include "../common/calibrate.hpp"

int main(int argc, char** argv) {
    Args args(argc, argv);
    std::string data_path = args.get("data", "data/creditcard.csv");
    int n_trees = args.get_int("trees", 100);
    int psi = args.get_int("psi", 256);
    uint64_t seed = static_cast<uint64_t>(args.get_int("seed", 42));
    double train_frac = args.get_double("train-frac", 0.5);
    int score_n = args.get_int("score-n", -1);
    bool adaptive = args.has("adaptive");
    bool theta_explicit = args.has("theta");
    double theta = args.get_double("theta", 0.5);
    double contamination = args.get_double("contamination", 0.01);
    double z = args.get_double("z", 2.0);
    int min_trees = args.get_int("min-trees", 10);
    std::string explain = args.get("explain", "none");
    std::string out_path = args.get("out", "");
    bool compare_full = args.has("compare-full");
    int repeat = std::max(1, args.get_int("repeat", 1));
    int threads = args.get_int("threads", 0);
    // Rolling retraining (docs/10_drift_and_tuning.md): every `retrain_every`
    // streamed transactions, rebuild the forest and re-calibrate theta on the
    // most recent `window` rows that have already arrived. Labels are never
    // read, so this stays fully unsupervised. 0 = off (static model).
    int retrain_every = args.get_int("retrain-every", 0);
    int window = args.get_int("window", 50000);
    bool drop_time = args.has("drop-time");
    if (threads > 0) omp_set_num_threads(threads);

    Dataset ds = load_creditcard_csv(data_path);
    int n_features = static_cast<int>(ds.feature_names.size());
    standardize_columns(ds.X, {0, n_features - 1});
    if (drop_time) {
        drop_column(ds, 0); // Time is monotonic, so stream values always lie outside the training range
        n_features = static_cast<int>(ds.feature_names.size());
    }

    SplitIndices split = split_train_stream(ds.X.size(), train_frac);

    std::vector<std::vector<float>> train_data;
    train_data.reserve(split.train_pool.size());
    for (int i : split.train_pool) train_data.push_back(ds.X[i]);

    std::vector<int> stream_idx = split.stream;
    if (score_n > 0 && score_n < static_cast<int>(stream_idx.size())) {
        stream_idx.resize(score_n);
    }
    std::vector<std::vector<float>> stream_data;
    std::vector<int> stream_labels;
    stream_data.reserve(stream_idx.size());
    stream_labels.reserve(stream_idx.size());
    for (int i : stream_idx) {
        stream_data.push_back(ds.X[i]);
        stream_labels.push_back(ds.y[i]);
    }

    IsolationForest forest;
    forest.init(n_features, psi);
    forest.prepare_trees(n_trees);

    auto build_forest = [&](const std::vector<std::vector<float>>& data, uint64_t seed_base) {
        #pragma omp parallel for schedule(dynamic)
        for (int t = 0; t < n_trees; t++) {
            forest.trees[t] = forest.build_one_tree(data, seed_base + static_cast<uint64_t>(t));
        }
    };

    Timer build_timer;
    build_timer.start();
    build_forest(train_data, seed);
    double build_ms = build_timer.elapsed_ms();

    if (!theta_explicit) {
        theta = calibrate_theta(forest, train_data, contamination);
    }

    size_t n_stream = stream_data.size();
    std::vector<double> scores(n_stream);
    std::vector<int> pred_label(n_stream);
    std::vector<int> trees_used(n_stream, n_trees);
    std::vector<std::vector<double>> contribs;
    if (explain != "none") contribs.assign(n_stream, std::vector<double>(n_features, 0.0));

    double score_ms = 0.0;
    double initial_theta = theta;
    double retrain_ms = 0.0;
    int n_retrains = 0;
    int effective_min_trees = adaptive ? min_trees : n_trees;

    auto score_range = [&](size_t begin, size_t end) {
        // dynamic scheduling matters here: with --adaptive, per-transaction
        // cost varies a lot (some points exit after min_trees, others run
        // all n_trees), so a static split would load-imbalance across threads.
        #pragma omp parallel for schedule(dynamic, 64)
        for (size_t i = begin; i < end; i++) {
            if (explain == "integrated") {
                auto res = forest.score_adaptive_explain(stream_data[i], theta, z, effective_min_trees);
                scores[i] = res.score; pred_label[i] = res.label; trees_used[i] = res.trees_used;
                contribs[i] = res.contrib;
            } else if (explain == "posthoc") {
                if (adaptive) {
                    auto res = forest.score_adaptive(stream_data[i], theta, z, min_trees);
                    scores[i] = res.score; pred_label[i] = res.label; trees_used[i] = res.trees_used;
                } else {
                    scores[i] = forest.score_full(stream_data[i]);
                    pred_label[i] = scores[i] > theta ? 1 : 0;
                    trees_used[i] = n_trees;
                }
                contribs[i] = forest.explain_posthoc(stream_data[i]);
            } else {
                if (adaptive) {
                    auto res = forest.score_adaptive(stream_data[i], theta, z, min_trees);
                    scores[i] = res.score; pred_label[i] = res.label; trees_used[i] = res.trees_used;
                } else {
                    scores[i] = forest.score_full(stream_data[i]);
                    pred_label[i] = scores[i] > theta ? 1 : 0;
                    trees_used[i] = n_trees;
                }
            }
        }
    };

    for (int r = 0; r < repeat; r++) {
        if (retrain_every <= 0) {
            Timer score_timer;
            score_timer.start();
            score_range(0, n_stream);
            score_ms += score_timer.elapsed_ms();
            continue;
        }
        if (r > 0) { // restore the initial model so every repeat does identical work
            build_forest(train_data, seed);
            theta = initial_theta;
        }
        n_retrains = 0;
        for (size_t begin = 0; begin < n_stream; begin += retrain_every) {
            size_t end = std::min(n_stream, begin + static_cast<size_t>(retrain_every));
            if (begin > 0) {
                Timer retrain_timer;
                retrain_timer.start();
                int g = stream_idx[begin]; // global row index of the chunk's first transaction
                int w0 = std::max(0, g - window);
                std::vector<std::vector<float>> window_data(ds.X.begin() + w0, ds.X.begin() + g);
                n_retrains++;
                build_forest(window_data, seed + static_cast<uint64_t>(n_retrains) * n_trees);
                if (!theta_explicit) {
                    size_t ref = std::min<size_t>(window_data.size(), 20000);
                    std::vector<std::vector<float>> recent(window_data.end() - ref, window_data.end());
                    theta = calibrate_theta(forest, recent, contamination);
                }
                retrain_ms += retrain_timer.elapsed_ms();
            }
            Timer score_timer;
            score_timer.start();
            score_range(begin, end);
            score_ms += score_timer.elapsed_ms();
        }
    }
    score_ms /= repeat;
    retrain_ms /= repeat;

    ConfusionCounts cc = compute_confusion(stream_labels, pred_label);
    long total_trees_used = 0;
    for (int t : trees_used) total_trees_used += t;
    double avg_trees_used = static_cast<double>(total_trees_used) / n_stream;

    double agreement_full = -1.0;
    double full_score_ms = -1.0;
    if (compare_full && adaptive && retrain_every <= 0) { // the reference would need the same per-chunk models
        std::vector<int> full_label(n_stream);
        Timer full_timer;
        full_timer.start();
        #pragma omp parallel for schedule(dynamic, 64)
        for (size_t i = 0; i < n_stream; i++) {
            double s = forest.score_full(stream_data[i]);
            full_label[i] = s > theta ? 1 : 0;
        }
        full_score_ms = full_timer.elapsed_ms();
        agreement_full = compute_agreement(pred_label, full_label);
    }

    double throughput = n_stream / (score_ms / 1000.0);
    int actual_threads = omp_get_max_threads();

    std::cout << "RESULT binary=openmp n_trees=" << n_trees << " psi=" << psi
              << " threads=" << actual_threads << " ranks=1 n_train=" << train_data.size()
              << " n_stream=" << n_stream << " adaptive=" << (adaptive ? 1 : 0)
              << " theta=" << theta << " contamination=" << contamination
              << " z=" << z << " min_trees=" << effective_min_trees
              << " explain=" << explain << " drop_time=" << (drop_time ? 1 : 0)
              << " retrain_every=" << retrain_every << " window=" << window
              << " n_retrains=" << n_retrains << " retrain_ms=" << retrain_ms
              << " build_ms=" << build_ms
              << " score_ms=" << score_ms << " throughput_tx_s=" << throughput
              << " avg_trees_used=" << avg_trees_used
              << " precision=" << cc.precision() << " recall=" << cc.recall()
              << " f1=" << cc.f1() << " tp=" << cc.tp << " fp=" << cc.fp
              << " fn=" << cc.fn << " tn=" << cc.tn
              << " agreement_full=" << agreement_full
              << " full_score_ms=" << full_score_ms
              << std::endl;

    if (!out_path.empty()) {
        std::ofstream out(out_path);
        out << "row_id,true_label,pred_label,score,trees_used\n";
        for (size_t i = 0; i < n_stream; i++) {
            out << stream_idx[i] << "," << stream_labels[i] << "," << pred_label[i]
                << "," << scores[i] << "," << trees_used[i] << "\n";
        }
    }

    return 0;
}
