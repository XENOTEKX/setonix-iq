#!/bin/bash
# build_avx512_r2_mpi.sh — Build IQ-TREE3 v3.1.2 + R2 + AVX-512 MPI binary
# on a Gadi SPR compute node using ICX + OpenMPI.
#
# Rebuilds /scratch/um09/as1708/iqtree3-3.1.2/build-profiling-mpi/iqtree3-mpi
# Source branch: gadi-spr-r2-avx512 (07966e40)
#
#PBS -N iq-build-avx512-r2
#PBS -P um09
#PBS -q normalsr
#PBS -l ncpus=32
#PBS -l mem=64GB
#PBS -l walltime=01:00:00
#PBS -l storage=scratch/um09
#PBS -j oe

set -euo pipefail

SRC=/scratch/um09/as1708/iqtree3-3.1.2/src/iqtree3
BUILD=/scratch/um09/as1708/iqtree3-3.1.2/build-profiling-mpi

echo "[build] Loading modules..."
module load openmpi/4.1.7
module load intel-compiler-llvm/2024.2.0

echo "[build] Compiler: $(icx --version 2>&1 | head -1)"
echo "[build] MPI:      $(mpirun --version 2>&1 | head -1)"
echo "[build] Source:   $(git -C $SRC log --oneline -1)"

mkdir -p "$BUILD"
cd "$BUILD"

cmake "$SRC" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_C_COMPILER=icx \
    -DIQTREE_FLAGS="avx512" \
    -DCMAKE_CXX_FLAGS="-O3 -xSAPPHIRERAPIDS -fno-omit-frame-pointer" \
    -DCMAKE_C_FLAGS="-O3 -xSAPPHIRERAPIDS -fno-omit-frame-pointer" \
    -DUSE_MPI=ON

make -j 32

echo "[build] Built: $BUILD/iqtree3-mpi"
$BUILD/iqtree3-mpi --version 2>&1 | head -3
echo "[build] DONE"
