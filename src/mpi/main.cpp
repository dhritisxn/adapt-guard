// MPI Isolation Forest: partitions the streaming/scoring workload across
// processes to simulate a multi-node deployment (each rank = one node).
//
// Design decision (see docs/03_mpi.md): tree construction is NOT
// communicated -- every rank independently builds the identical forest
// using the same deterministic per-tree seed used by serial/OpenMP. This
// is cheap (build_ms is a small, ~constant cost, see docs/02_baseline.md)
// and means the forest never needs to cross the network, consistent with a
// "replicated model, partitioned stream" deployment pattern. All the real
// MPI communication in this program is in the scoring phase: each rank
// scores its contiguous shard of the stream, then results are gathered
// back to rank 0 with MPI_Gatherv for aggregation/output.
//
// --dist-build switches to a distributed build instead: trees are split
// round-robin across ranks and exchanged with MPI_Allgatherv (see
// docs/03_mpi.md). The resulting forest is bit-identical either way.
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <mpi.h>

#include "../common/dataset.hpp"
#include "../common/isolation_forest.hpp"
#include "../common/args.hpp"
#include "../common/metrics.hpp"
#include "../common/split.hpp"
#include "../common/calibrate.hpp"

static void compute_shard(int n_stream, int nprocs, int rank, int& start, int& count) {
    int base = n_stream / nprocs;
    int rem = n_stream % nprocs;
    if (rank < rem) {
        count = base + 1;
        start = rank * (base + 1);
    } else {
        count = base;
        start = rem * (base + 1) + (rank - rem) * base;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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
    bool dist_build = args.has("dist-build");

    Dataset ds = load_creditcard_csv(data_path); // every rank loads independently (see design note above)
    int n_features = static_cast<int>(ds.feature_names.size());
    standardize_columns(ds.X, {0, n_features - 1});

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
    int n_stream = static_cast<int>(stream_data.size());

    IsolationForest forest;
    forest.init(n_features, psi);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_build0 = MPI_Wtime();
    double build_comm_ms = 0.0;
    if (dist_build) {
        // Distributed build: rank r builds trees r, r+P, r+2P, ... (round-
        // robin, so expensive and cheap trees spread evenly), then every rank
        // receives every tree via two MPI_Allgatherv calls (per-tree node
        // counts, then the POD IFNode buffers as raw bytes). Because tree t
        // is still seeded seed+t, the assembled forest is bit-identical to
        // the replicated build.
        forest.prepare_trees(n_trees);
        std::vector<int> my_counts;
        std::vector<IFNode> my_nodes;
        for (int t = rank; t < n_trees; t += nprocs) {
            forest.trees[t] = forest.build_one_tree(train_data, seed + static_cast<uint64_t>(t));
            my_counts.push_back(static_cast<int>(forest.trees[t].nodes.size()));
            my_nodes.insert(my_nodes.end(), forest.trees[t].nodes.begin(), forest.trees[t].nodes.end());
        }
        double t_comm_b0 = MPI_Wtime();
        std::vector<int> trees_per_rank(nprocs), tree_displs(nprocs);
        for (int r = 0; r < nprocs; r++) {
            trees_per_rank[r] = n_trees / nprocs + (r < n_trees % nprocs ? 1 : 0);
            tree_displs[r] = r == 0 ? 0 : tree_displs[r - 1] + trees_per_rank[r - 1];
        }
        std::vector<int> all_counts(n_trees);
        MPI_Allgatherv(my_counts.data(), static_cast<int>(my_counts.size()), MPI_INT,
                       all_counts.data(), trees_per_rank.data(), tree_displs.data(), MPI_INT, MPI_COMM_WORLD);
        std::vector<int> byte_counts(nprocs, 0), byte_displs(nprocs, 0);
        for (int r = 0; r < nprocs; r++) {
            for (int j = 0; j < trees_per_rank[r]; j++) {
                byte_counts[r] += all_counts[tree_displs[r] + j] * static_cast<int>(sizeof(IFNode));
            }
            byte_displs[r] = r == 0 ? 0 : byte_displs[r - 1] + byte_counts[r - 1];
        }
        int total_nodes = 0;
        for (int c : all_counts) total_nodes += c;
        std::vector<IFNode> all_nodes(total_nodes);
        MPI_Allgatherv(my_nodes.data(), static_cast<int>(my_nodes.size() * sizeof(IFNode)), MPI_BYTE,
                       all_nodes.data(), byte_counts.data(), byte_displs.data(), MPI_BYTE, MPI_COMM_WORLD);
        build_comm_ms = (MPI_Wtime() - t_comm_b0) * 1000.0;
        // Unpack: gathered order is rank-major (rank 0's trees, then rank 1's...).
        size_t pos = 0;
        for (int r = 0; r < nprocs; r++) {
            for (int j = 0; j < trees_per_rank[r]; j++) {
                int t = r + j * nprocs;
                int cnt = all_counts[tree_displs[r] + j];
                forest.trees[t].nodes.assign(all_nodes.begin() + pos, all_nodes.begin() + pos + cnt);
                pos += cnt;
            }
        }
    } else {
        forest.build_all(train_data, n_trees, seed); // identical on every rank, no communication needed
    }
    double local_build_ms = (MPI_Wtime() - t_build0) * 1000.0;

    if (!theta_explicit) {
        theta = calibrate_theta(forest, train_data, contamination); // deterministic, identical on every rank
    }

    int shard_start, shard_count;
    compute_shard(n_stream, nprocs, rank, shard_start, shard_count);

    std::vector<double> local_scores(shard_count);
    std::vector<int> local_pred(shard_count);
    std::vector<int> local_trees_used(shard_count, n_trees);
    int effective_min_trees = adaptive ? min_trees : n_trees;

    MPI_Barrier(MPI_COMM_WORLD);
    double local_score_ms = 0.0;
    for (int r = 0; r < repeat; r++) {
        double t_score0 = MPI_Wtime();
        for (int li = 0; li < shard_count; li++) {
            const auto& x = stream_data[shard_start + li];
            if (adaptive) {
                auto res = forest.score_adaptive(x, theta, z, min_trees);
                local_scores[li] = res.score; local_pred[li] = res.label; local_trees_used[li] = res.trees_used;
            } else {
                local_scores[li] = forest.score_full(x);
                local_pred[li] = local_scores[li] > theta ? 1 : 0;
                local_trees_used[li] = n_trees;
            }
        }
        local_score_ms += (MPI_Wtime() - t_score0) * 1000.0;
    }
    local_score_ms /= repeat;

    // Gather variable-sized shards back to rank 0, in original stream order.
    std::vector<int> recvcounts, displs;
    if (rank == 0) {
        recvcounts.resize(nprocs);
        displs.resize(nprocs);
        for (int r = 0; r < nprocs; r++) {
            int s, c;
            compute_shard(n_stream, nprocs, r, s, c);
            recvcounts[r] = c;
            displs[r] = s;
        }
    }
    std::vector<double> all_scores(rank == 0 ? n_stream : 0);
    std::vector<int> all_pred(rank == 0 ? n_stream : 0);
    std::vector<int> all_trees_used(rank == 0 ? n_stream : 0);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_comm0 = MPI_Wtime();
    MPI_Gatherv(local_scores.data(), shard_count, MPI_DOUBLE,
                all_scores.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_pred.data(), shard_count, MPI_INT,
                all_pred.data(), recvcounts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_trees_used.data(), shard_count, MPI_INT,
                all_trees_used.data(), recvcounts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);
    double comm_ms = (MPI_Wtime() - t_comm0) * 1000.0;

    double max_build_ms, max_score_ms, min_score_ms, sum_score_ms;
    MPI_Reduce(&local_build_ms, &max_build_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_score_ms, &max_score_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_score_ms, &min_score_ms, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_score_ms, &sum_score_ms, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Full-ensemble reference, for --compare-full: scored across the SAME
    // rank/shard partition as the adaptive run above, so the resulting
    // speedup is "adaptive at N ranks" vs "full ensemble at N ranks" --
    // an apples-to-apples parallel comparison, not adaptive-parallel vs
    // full-serial (which would conflate the adaptive speedup with the
    // MPI speedup and overstate the former).
    double full_score_ms_out = -1.0;
    std::vector<int> all_full_label(rank == 0 ? n_stream : 0);
    if (compare_full && adaptive) {
        std::vector<int> local_full_label(shard_count);
        MPI_Barrier(MPI_COMM_WORLD);
        double t_full0 = MPI_Wtime();
        for (int li = 0; li < shard_count; li++) {
            local_full_label[li] = forest.score_full(stream_data[shard_start + li]) > theta ? 1 : 0;
        }
        double local_full_ms = (MPI_Wtime() - t_full0) * 1000.0;
        double max_full_ms;
        MPI_Reduce(&local_full_ms, &max_full_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_full_label.data(), shard_count, MPI_INT,
                    all_full_label.data(), recvcounts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0) full_score_ms_out = max_full_ms;
    }

    std::cerr << "RANKINFO rank=" << rank << " nprocs=" << nprocs
              << " shard_start=" << shard_start << " shard_count=" << shard_count
              << " local_build_ms=" << local_build_ms << " local_score_ms=" << local_score_ms
              << std::endl;

    if (rank == 0) {
        ConfusionCounts cc = compute_confusion(stream_labels, all_pred);
        long total_trees_used = 0;
        for (int t : all_trees_used) total_trees_used += t;
        double avg_trees_used = static_cast<double>(total_trees_used) / n_stream;
        double throughput = n_stream / (max_score_ms / 1000.0);
        double load_imbalance_ms = max_score_ms - min_score_ms;
        double avg_score_ms = sum_score_ms / nprocs;

        double agreement_full = -1.0;
        double full_score_ms = full_score_ms_out;
        if (compare_full && adaptive) {
            agreement_full = compute_agreement(all_pred, all_full_label);
        }

        std::cout << "RESULT binary=mpi n_trees=" << n_trees << " psi=" << psi
                  << " threads=1 ranks=" << nprocs << " n_train=" << train_data.size()
                  << " n_stream=" << n_stream << " adaptive=" << (adaptive ? 1 : 0)
                  << " theta=" << theta << " contamination=" << contamination
                  << " z=" << z << " min_trees=" << effective_min_trees
                  << " explain=" << explain << " dist_build=" << (dist_build ? 1 : 0)
                  << " build_ms=" << max_build_ms << " build_comm_ms=" << build_comm_ms
                  << " score_ms=" << max_score_ms << " avg_score_ms=" << avg_score_ms
                  << " load_imbalance_ms=" << load_imbalance_ms << " comm_ms=" << comm_ms
                  << " throughput_tx_s=" << throughput
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
            for (int i = 0; i < n_stream; i++) {
                out << stream_idx[i] << "," << stream_labels[i] << "," << all_pred[i]
                    << "," << all_scores[i] << "," << all_trees_used[i] << "\n";
            }
        }
    }

    MPI_Finalize();
    return 0;
}
