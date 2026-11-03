#!/bin/bash
set -e
cd /root/adaptguard
./build/serial --data data/creditcard.csv --trees 100 --psi 256 --seed 42 --score-n 20000 --contamination 0.01 --out /tmp/serial_out.csv > /tmp/serial_res.txt
for T in 1 2 4 8; do
  ./build/openmp --data data/creditcard.csv --trees 100 --psi 256 --seed 42 --score-n 20000 --contamination 0.01 --threads $T --out /tmp/openmp_out_$T.csv > /tmp/openmp_res_$T.txt
  DIFF=$(diff /tmp/serial_out.csv /tmp/openmp_out_$T.csv | wc -l)
  echo "threads=$T diff_lines=$DIFF"
done
echo "--- serial ---"
cat /tmp/serial_res.txt
echo "--- openmp threads=8 ---"
cat /tmp/openmp_res_8.txt
