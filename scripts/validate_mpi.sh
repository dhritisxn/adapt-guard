#!/bin/bash
set -e
cd /root/adaptguard
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
./build/serial --data data/creditcard.csv --trees 100 --psi 256 --seed 42 --score-n 20000 --contamination 0.01 --out /tmp/serial_out.csv > /tmp/serial_res.txt
for P in 1 2 4 8; do
  mpirun -np $P --oversubscribe ./build/mpi --data data/creditcard.csv --trees 100 --psi 256 --seed 42 --score-n 20000 --contamination 0.01 --out /tmp/mpi_out_$P.csv > /tmp/mpi_res_$P.txt 2>/tmp/mpi_rankinfo_$P.txt
  # sort both CSVs by row_id before diffing since gatherv preserves order already, but sort defensively
  sort -t, -k1,1n /tmp/serial_out.csv > /tmp/serial_out_sorted.csv
  sort -t, -k1,1n /tmp/mpi_out_$P.csv > /tmp/mpi_out_sorted_$P.csv
  DIFF=$(diff /tmp/serial_out_sorted.csv /tmp/mpi_out_sorted_$P.csv | wc -l)
  echo "procs=$P diff_lines=$DIFF"
done
echo "--- serial ---"; cat /tmp/serial_res.txt
echo "--- mpi procs=8 ---"; cat /tmp/mpi_res_8.txt
echo "--- rankinfo procs=8 ---"; cat /tmp/mpi_rankinfo_8.txt
