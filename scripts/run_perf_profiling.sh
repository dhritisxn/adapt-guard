#!/bin/bash
# Profiles OpenMP and MPI at low and high resource counts with `perf stat`,
# to see whether the scoring phase is compute-bound, memory-bound, or
# dominated by synchronization overhead / load imbalance. Raw output saved
# under docs/06_results/data/perf_*.txt -- see docs/06_results/results_summary.md
# for the interpretation.
set -e
cd /root/adaptguard
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
PERF=/usr/local/bin/perf
OUT=/root/adaptguard/results
mkdir -p $OUT

EVENTS="task-clock,context-switches,cpu-migrations,page-faults,cycles,instructions,cache-references,cache-misses,branch-instructions,branch-misses"

echo "--- openmp threads=1 ---"
$PERF stat -e $EVENTS -o $OUT/perf_openmp_t1.txt -- \
  ./build/openmp --data data/creditcard.csv --trees 100 --psi 256 --seed 42 --score-n 100000 --contamination 0.01 --threads 1 > /dev/null

echo "--- openmp threads=12 ---"
$PERF stat -e $EVENTS -o $OUT/perf_openmp_t12.txt -- \
  ./build/openmp --data data/creditcard.csv --trees 100 --psi 256 --seed 42 --score-n 100000 --contamination 0.01 --threads 12 > /dev/null

echo "--- mpi ranks=1 ---"
$PERF stat -e $EVENTS -o $OUT/perf_mpi_p1.txt -- \
  mpirun -np 1 --oversubscribe ./build/mpi --data data/creditcard.csv --trees 100 --psi 256 --seed 42 --score-n 100000 --contamination 0.01 > /dev/null

echo "--- mpi ranks=8 ---"
$PERF stat -e $EVENTS -o $OUT/perf_mpi_p8.txt -- \
  mpirun -np 8 --oversubscribe ./build/mpi --data data/creditcard.csv --trees 100 --psi 256 --seed 42 --score-n 100000 --contamination 0.01 > /dev/null

echo "=== results ==="
for f in $OUT/perf_openmp_t1.txt $OUT/perf_openmp_t12.txt $OUT/perf_mpi_p1.txt $OUT/perf_mpi_p8.txt; do
  echo "--- $f ---"
  cat $f
done
