#!/bin/bash
# perf record + perf report: which functions actually burn the cycles,
# complementing perf stat's aggregate counters (docs/06_results/data/perf_*.txt)
# with a hot-spot breakdown. Requires -g (call-graph) and -fno-omit-frame-pointer
# would give better unwinding, but frame-pointer-based unwinding still works
# reasonably with -O3 on these small binaries since they're not heavily inlined
# across translation units (single-TU builds).
set -e
cd /root/adaptguard
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
PERF=/usr/local/bin/perf
OUT=/root/adaptguard/results
mkdir -p $OUT

echo "--- openmp threads=12, perf record ---"
$PERF record -g -o $OUT/perf_openmp_t12.data -- \
  ./build/openmp --data data/creditcard.csv --trees 100 --psi 256 --seed 42 --score-n 200000 --contamination 0.01 --threads 12 > /dev/null
$PERF report -i $OUT/perf_openmp_t12.data --stdio --sort=overhead,symbol -g none --percent-limit 1 > $OUT/perf_record_openmp_t12_report.txt

echo "--- mpi ranks=8, perf record ---"
$PERF record -g -o $OUT/perf_mpi_p8.data -- \
  mpirun -np 8 --oversubscribe ./build/mpi --data data/creditcard.csv --trees 100 --psi 256 --seed 42 --score-n 200000 --contamination 0.01 > /dev/null
$PERF report -i $OUT/perf_mpi_p8.data --stdio --sort=overhead,symbol -g none --percent-limit 1 > $OUT/perf_record_mpi_p8_report.txt

echo "=== reports ==="
echo "--- openmp t12 top symbols ---"
head -40 $OUT/perf_record_openmp_t12_report.txt
echo "--- mpi p8 top symbols ---"
head -40 $OUT/perf_record_mpi_p8_report.txt
