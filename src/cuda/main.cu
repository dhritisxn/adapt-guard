// CUDA Isolation Forest: GPU-parallel transaction scoring (full ensemble and
// adaptive early-exit), one GPU thread per transaction.
//
// Design (see docs/09_cuda.md):
//  - The forest is built on the CPU with the exact same deterministic code
//    as serial/OpenMP/MPI (build is ~8 ms, not worth a GPU port), then
//    flattened into one contiguous node array (absolute child indices) plus
//    a per-tree root offset, and copied to the GPU once.
//  - Leaf corrections c(size) are precomputed on the host, so the device
//    path-length sum is the same sequence of double additions as the CPU
//    code. Compiled with --fmad=false so nvcc doesn't contract mul+add into
//    FMA (which rounds differently) -- this is what lets full-ensemble GPU
//    output be diffed bit-for-bit against the serial baseline.
//  - The stream is stored column-major (feature-major), so at the shallow
//    levels of every tree, where neighbouring threads test the same feature,
//    their loads coalesce.
//  - Adaptive early exit runs the identical Welford + z*SE stopping rule.
//    On SIMT hardware a warp (32 threads) only retires when its slowest
//    thread does, so early exit saves work only when a whole warp agrees.
//    warp_efficiency (useful tree-evaluations / issued tree-evaluations,
//    computed from trees_used) quantifies exactly that.
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#include "../common/dataset.hpp"
#include "../common/isolation_forest.hpp"
#include "../common/args.hpp"
#include "../common/metrics.hpp"
#include "../common/split.hpp"
#include "../common/calibrate.hpp"

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            std::cerr << "CUDA error " << cudaGetErrorString(err_) << " at "      \
                      << __FILE__ << ":" << __LINE__ << std::endl;                \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

struct GNode {
    int feature;      // -1 marks a leaf
    float split_value;
    int left;         // absolute index into the flattened node array
    int right;
};

__device__ __forceinline__ double gpu_path_length(const GNode* __restrict__ nodes,
                                                   const double* __restrict__ leaf_c,
                                                   int root, const float* __restrict__ X,
                                                   int n_stream, int i) {
    int node_idx = root;
    int depth = 0;
    while (true) {
        GNode node = nodes[node_idx];
        if (node.feature == -1) {
            return depth + leaf_c[node_idx];
        }
        float v = X[static_cast<size_t>(node.feature) * n_stream + i];
        node_idx = (v < node.split_value) ? node.left : node.right;
        depth++;
    }
}

// Full ensemble: writes the mean path length; the host converts it to a
// score with the same score_to_anomaly() the CPU versions use.
__global__ void score_full_kernel(const GNode* __restrict__ nodes, const double* __restrict__ leaf_c,
                                  const int* __restrict__ roots, int n_trees,
                                  const float* __restrict__ X, int n_stream,
                                  double* __restrict__ mean_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_stream) return;
    double sum = 0.0;
    for (int t = 0; t < n_trees; t++) {
        sum += gpu_path_length(nodes, leaf_c, roots[t], X, n_stream, i);
    }
    mean_out[i] = sum / static_cast<double>(n_trees);
}

// Adaptive early exit, same rule as IsolationForest::score_adaptive.
// label_out = 1/0 on early exit, -1 if all trees were used (host then
// decides the label from the point estimate, exactly like the CPU code).
__global__ void score_adaptive_kernel(const GNode* __restrict__ nodes, const double* __restrict__ leaf_c,
                                      const int* __restrict__ roots, int n_trees,
                                      const float* __restrict__ X, int n_stream,
                                      double h_theta, double z, int min_trees,
                                      double* __restrict__ mean_out, int* __restrict__ used_out,
                                      int* __restrict__ label_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_stream) return;
    double mean = 0.0, m2 = 0.0;
    int k = 0;
    for (int t = 0; t < n_trees; t++) {
        double h = gpu_path_length(nodes, leaf_c, roots[t], X, n_stream, i);
        k++;
        double delta = h - mean;
        mean += delta / k;
        double delta2 = h - mean;
        m2 += delta * delta2;

        if (k >= min_trees && k < n_trees) {
            // Same band test as the CPU code, but in path-length space:
            // 2^(-(mean + z*se)/c) > theta  <=>  mean + z*se < h_theta, with
            // h_theta = -c*log2(theta) precomputed on the host. This removes two
            // FP64 pow() calls per tree, which are very expensive on consumer
            // GPUs (FP64 throughput is 1/32 of FP32 on the GTX 1650).
            double variance = m2 / (k - 1);
            double se = sqrt(variance / k);
            if (mean + z * se < h_theta) { mean_out[i] = mean; used_out[i] = k; label_out[i] = 1; return; }
            if (mean - z * se > h_theta) { mean_out[i] = mean; used_out[i] = k; label_out[i] = 0; return; }
        }
    }
    mean_out[i] = mean;
    used_out[i] = k;
    label_out[i] = -1;
}

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
    std::string out_path = args.get("out", "");
    bool compare_full = args.has("compare-full");
    int repeat = std::max(1, args.get_int("repeat", 1));
    int block_size = args.get_int("block", 128);
    int device = args.get_int("device", 0);

    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    CUDA_CHECK(cudaFree(0)); // force context creation outside any timed region

    Dataset ds = load_creditcard_csv(data_path);
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
    int n_stream = static_cast<int>(stream_idx.size());
    std::vector<int> stream_labels(n_stream);
    std::vector<float> X_colmajor(static_cast<size_t>(n_features) * n_stream);
    for (int i = 0; i < n_stream; i++) {
        stream_labels[i] = ds.y[stream_idx[i]];
        for (int f = 0; f < n_features; f++) {
            X_colmajor[static_cast<size_t>(f) * n_stream + i] = ds.X[stream_idx[i]][f];
        }
    }

    IsolationForest forest;
    forest.init(n_features, psi);
    Timer build_timer;
    build_timer.start();
    forest.build_all(train_data, n_trees, seed); // CPU build, identical to serial
    double build_ms = build_timer.elapsed_ms();

    if (!theta_explicit) {
        theta = calibrate_theta(forest, train_data, contamination);
    }

    // Flatten the forest.
    std::vector<GNode> h_nodes;
    std::vector<double> h_leaf_c;
    std::vector<int> h_roots(n_trees);
    for (int t = 0; t < n_trees; t++) {
        int offset = static_cast<int>(h_nodes.size());
        h_roots[t] = offset;
        for (const IFNode& n : forest.trees[t].nodes) {
            GNode g;
            g.feature = n.feature;
            g.split_value = n.split_value;
            g.left = n.feature == -1 ? -1 : n.left + offset;
            g.right = n.feature == -1 ? -1 : n.right + offset;
            h_nodes.push_back(g);
            h_leaf_c.push_back(n.feature == -1 ? IsolationForest::c_factor(static_cast<double>(n.size)) : 0.0);
        }
    }

    GNode* d_nodes; double* d_leaf_c; int* d_roots; float* d_X;
    double* d_mean; int* d_used; int* d_label;
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    Timer h2d_timer;
    h2d_timer.start();
    CUDA_CHECK(cudaMalloc(&d_nodes, h_nodes.size() * sizeof(GNode)));
    CUDA_CHECK(cudaMalloc(&d_leaf_c, h_leaf_c.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_roots, n_trees * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_X, X_colmajor.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean, n_stream * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_used, n_stream * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_label, n_stream * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_nodes, h_nodes.data(), h_nodes.size() * sizeof(GNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_leaf_c, h_leaf_c.data(), h_leaf_c.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_roots, h_roots.data(), n_trees * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_X, X_colmajor.data(), X_colmajor.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    double h2d_ms = h2d_timer.elapsed_ms();

    int grid = (n_stream + block_size - 1) / block_size;
    double h_theta = -forest.c_psi * std::log2(theta); // decision threshold as a mean path length
    int effective_min_trees = adaptive ? min_trees : n_trees;

    // Warm-up launch (first launch pays module load / JIT costs).
    score_full_kernel<<<grid, block_size>>>(d_nodes, d_leaf_c, d_roots, n_trees, d_X, n_stream, d_mean);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    double score_ms = 0.0;
    for (int r = 0; r < repeat; r++) {
        CUDA_CHECK(cudaEventRecord(ev0));
        if (adaptive) {
            score_adaptive_kernel<<<grid, block_size>>>(d_nodes, d_leaf_c, d_roots, n_trees, d_X, n_stream,
                                                        h_theta, z, min_trees,
                                                        d_mean, d_used, d_label);
        } else {
            score_full_kernel<<<grid, block_size>>>(d_nodes, d_leaf_c, d_roots, n_trees, d_X, n_stream, d_mean);
        }
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        score_ms += ms;
    }
    score_ms /= repeat;

    std::vector<double> mean_path(n_stream);
    std::vector<int> trees_used(n_stream, n_trees);
    std::vector<int> dev_label(n_stream, -1);
    Timer d2h_timer;
    d2h_timer.start();
    CUDA_CHECK(cudaMemcpy(mean_path.data(), d_mean, n_stream * sizeof(double), cudaMemcpyDeviceToHost));
    if (adaptive) {
        CUDA_CHECK(cudaMemcpy(trees_used.data(), d_used, n_stream * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(dev_label.data(), d_label, n_stream * sizeof(int), cudaMemcpyDeviceToHost));
    }
    double d2h_ms = d2h_timer.elapsed_ms();

    std::vector<double> scores(n_stream);
    std::vector<int> pred_label(n_stream);
    for (int i = 0; i < n_stream; i++) {
        scores[i] = forest.score_to_anomaly(mean_path[i]);
        pred_label[i] = dev_label[i] >= 0 ? dev_label[i] : (scores[i] > theta ? 1 : 0);
    }

    ConfusionCounts cc = compute_confusion(stream_labels, pred_label);
    long total_trees_used = 0;
    for (int t : trees_used) total_trees_used += t;
    double avg_trees_used = static_cast<double>(total_trees_used) / n_stream;

    // SIMT cost of early exit: a warp runs as long as its slowest lane.
    long issued = 0;
    for (int w = 0; w < n_stream; w += 32) {
        int mx = 0;
        int lanes = std::min(32, n_stream - w);
        for (int l = 0; l < lanes; l++) mx = std::max(mx, trees_used[w + l]);
        issued += static_cast<long>(mx) * lanes;
    }
    double warp_efficiency = static_cast<double>(total_trees_used) / static_cast<double>(issued);

    double agreement_full = -1.0;
    double full_score_ms = -1.0;
    if (compare_full && adaptive) {
        CUDA_CHECK(cudaEventRecord(ev0));
        score_full_kernel<<<grid, block_size>>>(d_nodes, d_leaf_c, d_roots, n_trees, d_X, n_stream, d_mean);
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        full_score_ms = ms;
        std::vector<double> full_mean(n_stream);
        CUDA_CHECK(cudaMemcpy(full_mean.data(), d_mean, n_stream * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<int> full_label(n_stream);
        for (int i = 0; i < n_stream; i++) full_label[i] = forest.score_to_anomaly(full_mean[i]) > theta ? 1 : 0;
        agreement_full = compute_agreement(pred_label, full_label);
    }

    double throughput = n_stream / (score_ms / 1000.0);
    std::cout << "RESULT binary=cuda n_trees=" << n_trees << " psi=" << psi
              << " threads=" << block_size << " ranks=1 gpu=sm_" << prop.major << prop.minor
              << " n_train=" << train_data.size()
              << " n_stream=" << n_stream << " adaptive=" << (adaptive ? 1 : 0)
              << " theta=" << theta << " contamination=" << contamination
              << " z=" << z << " min_trees=" << effective_min_trees
              << " explain=none build_ms=" << build_ms
              << " score_ms=" << score_ms << " h2d_ms=" << h2d_ms << " d2h_ms=" << d2h_ms
              << " throughput_tx_s=" << throughput
              << " avg_trees_used=" << avg_trees_used
              << " warp_efficiency=" << warp_efficiency
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
            out << stream_idx[i] << "," << stream_labels[i] << "," << pred_label[i]
                << "," << scores[i] << "," << trees_used[i] << "\n";
        }
    }

    cudaFree(d_nodes); cudaFree(d_leaf_c); cudaFree(d_roots); cudaFree(d_X);
    cudaFree(d_mean); cudaFree(d_used); cudaFree(d_label);
    return 0;
}
