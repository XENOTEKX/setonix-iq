# iqtree3-l2search — L2 batched-NNI search-axis dev tree

Isolated clone of iqtree3-gpu (branch tree-search-ts0 working tree) for the
GPU search-axis redesign. Does NOT share build dirs with iqtree3-gpu, so makes
here never contend with the gpu-on validation jobs.

- Branch: l2-batched-nni  (baseline commit = brute-force JOLT --ts-fused source)
- Frozen baseline binary: frozen-binaries/iqtree3-jolt-bruteforce.1a924889
- Build recipe: module load cuda/12.5.1 gcc/12.2.0; mkdir build-gpu-on; cd build-gpu-on;
  cmake -DIQTREE_GPU=ON -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/apps/cuda/12.5.1/bin/nvcc \
    -DCMAKE_CUDA_HOST_COMPILER=/apps/gcc/12.2.0/wrappers/g++ \
    -DCMAKE_CXX_COMPILER=/apps/gcc/12.2.0/wrappers/g++ .. ; make -j
- CUDA arch: sm_90 / compute_90 (H200/Hopper)
