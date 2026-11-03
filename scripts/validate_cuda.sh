#!/bin/bash
# Diffs CUDA output against the serial reference, full-ensemble and adaptive,
# at a small and at the full stream size, and at several block sizes.
set -e
cd /root/adaptguard
for N in 20000 -1; do
  ./build/serial --data data/creditcard.csv --trees 100 --psi 256 --seed 42 --score-n $N --contamination 0.01 --out /tmp/serial_full_$N.csv > /dev/null
  ./build/serial --data data/creditcard.csv --trees 100 --psi 256 --seed 42 --score-n $N --contamination 0.01 --adaptive --z 2.0 --min-trees 10 --out /tmp/serial_adapt_$N.csv > /dev/null
  for B in 64 128 256; do
    ./build/cuda --data data/creditcard.csv --trees 100 --psi 256 --seed 42 --score-n $N --contamination 0.01 --block $B --out /tmp/cuda_full_${N}_${B}.csv > /dev/null
    ./build/cuda --data data/creditcard.csv --trees 100 --psi 256 --seed 42 --score-n $N --contamination 0.01 --block $B --adaptive --z 2.0 --min-trees 10 --out /tmp/cuda_adapt_${N}_${B}.csv > /dev/null
    DF=$(diff /tmp/serial_full_$N.csv /tmp/cuda_full_${N}_${B}.csv | wc -l)
    DA=$(diff /tmp/serial_adapt_$N.csv /tmp/cuda_adapt_${N}_${B}.csv | wc -l)
    echo "score_n=$N block=$B full_diff_lines=$DF adaptive_diff_lines=$DA"
  done
done
