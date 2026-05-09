# IQ-TREE 3.1.2 Patch Files

Patches for IQ-TREE 3.1.2 (`v3.1.2`, commit `4e91dd61`) optimised for
Gadi NCI Xeon 8470Q "Sapphire Rapids" (2 sockets × 52 cores, DDR5-4800).

Tested with: **icpx** (Intel oneAPI DPC++ 2025.3.2) + libiomp5.

---

## Patches

### `0001-r1r2-numa-first-touch.patch` — NUMA first-touch (R1 + R2)

**Apply to get NUMA-aware socket-local memory allocation.**

| File | Change |
|------|--------|
| `tree/phylotreesse.cpp` | R1: first-touch CLV arrays before OMP parallel region |
| `tree/phylokernelnew.h` | R2: 5 OpenMP loops → `schedule(static)` + `num_threads` |

Effect on xlarge_mf (200 taxa, 100K sites), Gadi SPR, MPI 2×52 (1 node):

| Config | Wall time | vs baseline |
|--------|-----------|-------------|
| Baseline 1×104T | 1111.6 s | — |
| R1+R2 2×52T | 523.7 s | −53% |

See [numa-first-touch.md](../numa-first-touch.md) for full analysis.

---

### `0002-p2p3-avx512-cmake-kernel.patch` — AVX-512 cmake + kernel (P2 + P3)

**Apply on top of R1+R2 to enable AVX-512 double-precision kernels.**

> **Requires build flag P1:** `-DIQTREE_FLAGS="mpi KNL"` at cmake time.
> See [avx512-cmake-icpx-patch.md](../avx512-cmake-icpx-patch.md).

| File | Change |
|------|--------|
| `CMakeLists.txt` | P2: `IntelLLVM` branch → `-mavx512f -mfma` (not `-xMIC-AVX512`) |
| `tree/phylokernelavx512.cpp` | P3: fix 3 template-arity bugs in nonrev dispatch |

Effect on xlarge_mf (200 taxa, 100K sites), Gadi SPR, MPI 2×104 (2 nodes):

| Config | Wall time | vs FMA baseline |
|--------|-----------|-----------------|
| FMA baseline 2×104T | 334.6 s | — |
| AVX-512 2×104T | 324.5 s | −3.0% |

AVX-512 confirmed active: `ZMM = 35 521` (perf stat). BW-bound at full
thread count — larger gains expected on smaller datasets or partial trees.

See [avx512-audit.md](../avx512-audit.md) for full analysis.

---

## How to Apply

```bash
# Clone IQ-TREE 3.1.2
git clone https://github.com/iqtree/iqtree3.git
cd iqtree3
git checkout v3.1.2

# Option A — NUMA patches only (R1+R2):
git am 0001-r1r2-numa-first-touch.patch

# Option B — NUMA + AVX-512 (R1+R2 + P2+P3):
git am 0001-r1r2-numa-first-touch.patch
git am 0002-p2p3-avx512-cmake-kernel.patch
```

Then build with:

```bash
# For NUMA-only build (Intel compiler):
cmake -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc \
      -DIQTREE_FLAGS="omp" -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

# For AVX-512 build (requires icpx + P1 flag):
cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
      -DIQTREE_FLAGS="mpi KNL" -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

> The `-DIQTREE_FLAGS="mpi KNL"` flag (P1) is what activates the AVX-512
> code path in CMakeLists.txt. Without it, the AVX-512 kernel is not compiled
> even if P2 and P3 are applied.

---

## Branch References (local, not yet pushed upstream)

These patches were generated from local branches on top of `v3.1.2`:

| Branch | Commits above v3.1.2 | Contents |
|--------|----------------------|----------|
| `gadi-spr-r2-numa` | 1 | R1 + R2 NUMA first-touch |
| `gadi-spr-avx512` | 2 | R1 + R2 + P2 + P3 AVX-512 |

The `iqtree3` upstream is `https://github.com/iqtree/iqtree3.git`.
To push branches there you would need a fork and PR.
