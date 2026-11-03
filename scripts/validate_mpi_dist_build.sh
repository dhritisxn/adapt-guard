#!/bin/bash
# Diffs MPI --dist-build output (trees built round-robin across ranks and
# exchanged with MPI_Allgatherv) against the serial reference.
set -e
cd /root/adaptguard
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
./build/serial --data data/creditcard.csv --trees 100 --psi 256 --seed 42 --score-n 20000 --contamination 0.01 --out /tmp/serial_out.csv > /dev/null
for P in 1 2 3 4 8; do
  mpirun -np $P --oversubscribe ./build/mpi --data data/creditcard.csv --trees 100 --psi 256 --seed 42 --score-n 20000 --contamination 0.01 --dist-build --out /tmp/mpi_dist_out_$P.csv > /tmp/mpi_dist_res_$P.txt 2>/dev/null
  DIFF=$(diff /tmp/serial_out.csv /tmp/mpi_dist_out_$P.csv | wc -l)
  echo "procs=$P dist_build diff_lines=$DIFF"
done
