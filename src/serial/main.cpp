// Serial Isolation Forest: correctness reference for the OpenMP and MPI
// implementations, and the baseline for every speedup number in this project.
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

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

    Dataset ds = load_creditcard_csv(data_path);
    int n_features = static_cast<int>(ds.feature_names.size());
    standardize_columns(ds.X, {0, n_features - 1}); // Time, Amount

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

    Timer build_timer;
    build_timer.start();
    forest.build_all(train_data, n_trees, seed);
    double build_ms = build_timer.elapsed_ms();

    if (!theta_explicit) {
        theta = calibrate_theta(forest, train_data, contamination);
    }

    size_t n_stream = stream_data.size();
    std::vector<double> scores(n_stream);
    std::vector<int> pred_label(n_stream);
    std::vector<int> trees_used(n_stream, n_trees);
    std::vector<std::vector<double>> contribs; // only populated if explain != none
    if (explain != "none") contribs.assign(n_stream, std::vector<double>(n_features, 0.0));

    double score_ms = 0.0;
    int effective_min_trees = adaptive ? min_trees : n_trees;

    for (int r = 0; r < repeat; r++) {
        Timer score_timer;
        score_timer.start();
        for (size_t i = 0; i < n_stream; i++) {
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
        score_ms += score_timer.elapsed_ms();
    }
    score_ms /= repeat;

    ConfusionCounts cc = compute_confusion(stream_labels, pred_label);
    long total_trees_used = 0;
    for (int t : trees_used) total_trees_used += t;
    double avg_trees_used = static_cast<double>(total_trees_used) / n_stream;

    double agreement_full = -1.0;
    double full_score_ms = -1.0;
    if (compare_full && adaptive) {
        std::vector<int> full_label(n_stream);
        Timer full_timer;
        full_timer.start();
        for (size_t i = 0; i < n_stream; i++) {
            double s = forest.score_full(stream_data[i]);
            full_label[i] = s > theta ? 1 : 0;
        }
        full_score_ms = full_timer.elapsed_ms();
        agreement_full = compute_agreement(pred_label, full_label);
    }

    double throughput = n_stream / (score_ms / 1000.0);

    std::cout << "RESULT binary=serial n_trees=" << n_trees << " psi=" << psi
              << " threads=1 ranks=1 n_train=" << train_data.size()
              << " n_stream=" << n_stream << " adaptive=" << (adaptive ? 1 : 0)
              << " theta=" << theta << " contamination=" << contamination
              << " z=" << z << " min_trees=" << effective_min_trees
              << " explain=" << explain << " build_ms=" << build_ms
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
        if (explain != "none") {
            std::ofstream eout(out_path + ".explain.csv");
            eout << "row_id,true_label,pred_label,top1_feat,top1_contrib,top2_feat,top2_contrib,top3_feat,top3_contrib\n";
            for (size_t i = 0; i < n_stream; i++) {
                if (pred_label[i] != 1) continue; // explain anomalies only
                std::vector<int> order(n_features);
                for (int f = 0; f < n_features; f++) order[f] = f;
                std::sort(order.begin(), order.end(), [&](int a, int b) {
                    return contribs[i][a] > contribs[i][b];
                });
                eout << stream_idx[i] << "," << stream_labels[i] << "," << pred_label[i];
                for (int k = 0; k < 3; k++) {
                    int f = order[k];
                    eout << "," << ds.feature_names[f] << "," << contribs[i][f];
                }
                eout << "\n";
            }
        }
    }

    return 0;
}
