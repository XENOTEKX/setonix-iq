# IQ-TREE NUMA First-Touch Bug — Plain-English Explainer

**Status:** Verified in IQ-TREE 3 source at `/scratch/pawsey1351/asamuel/iqtree3/` (2026-05-07).
**Related:** `CHANGELOG.md` § 2026-05-05 (NUMA-aware OpenMP binding gap).
**Audience:** anyone touching IQ-TREE OpenMP performance who hasn't lived inside its source.

---

## TL;DR

Two functions in IQ-TREE — `Alignment::buildPattern()` and `PhyloTree::computePtnFreq()` — fill big per-pattern arrays from a **single thread** (the master). On a multi-socket box, this pins every page of those arrays to **one NUMA node** for the rest of the run. When the parallel likelihood loop later spawns threads on the *other* socket, every iteration of the inner loop pays a remote-memory penalty to read those arrays.

The fix is one line of OpenMP per init loop. The reason it works is a Linux memory-allocation rule called **first-touch placement**.

---

## 1. The cast of characters (plain-English)

### Pattern (a column of the alignment, deduplicated)

A phylogenetic alignment is a matrix: rows are species, columns are sites in the genome. Many columns are identical to each other — e.g. every species has an `A` at site 12 and also at site 47. Recomputing likelihoods for identical columns is wasteful, so IQ-TREE compresses the alignment into **patterns**: the set of *unique* columns. Each pattern stores how many original sites collapsed onto it. That count is its **frequency**.

So if a 5,000-site DNA alignment compresses to 1,200 unique patterns, IQ-TREE will:

- Store the 1,200 patterns once (in `aln->getPattern(i)`).
- Store a frequency for each: `ptn_freq[i]` = how many original sites looked like pattern `i`.
- Run the likelihood kernel **once per pattern**, then multiply by `ptn_freq[i]` to get the alignment-wide likelihood contribution.

This is a 4–5× speedup over computing per-site likelihoods, but it makes `ptn_freq[]` the single most-frequently-read array in the whole program.

### `buildPattern()` — the deduplicator

`alignment/alignment.cpp:2219`. Takes raw sequences and produces the deduplicated pattern list. Walks every site, hashes it, sticks it in a `std::vector<Pattern>` if it's new, bumps a counter if it's a repeat. Output:

- `aln`'s internal `vector<Pattern>` (the unique columns).
- `site_pattern[s]` = which pattern site `s` ended up assigned to.
- Each `Pattern.frequency` = how many sites collapsed onto it.

Runs once per alignment load. **Single-threaded.**

### `computePtnFreq()` — the frequency-vector copier

`tree/phylotreesse.cpp:537`. Copies `Pattern.frequency` (one per pattern) into a **flat double array** called `ptn_freq[]`. The flat array exists because the SIMD likelihood kernels need contiguous, aligned doubles, not `std::vector<Pattern>` accessors.

```cpp
void PhyloTree::computePtnFreq() {
    if (ptn_freq_computed) return;
    ptn_freq_computed = true;
    size_t nptn = aln->getNPattern();
    size_t maxptn = get_safe_upper_limit(nptn) + ...;
    int ptn;
    for (ptn = 0; ptn < nptn; ptn++)
        ptn_freq[ptn] = (*aln)[ptn].frequency;       // ← serial fill
    for (ptn = nptn; ptn < maxptn; ptn++)
        ptn_freq[ptn] = 0.0;                          // ← serial fill
}
```

That's the whole function. No `#pragma omp` anywhere. **Single-threaded.**

A sister function, `computePtnInvar()` at `tree/phylotreesse.cpp:549`, does the same thing for the invariant-site probability vector `ptn_invar[]`, starting with a serial `memset`.

### Where `ptn_freq[]` is used

`grep` finds ~25 read sites across the kernel files (`phylokernelnew.h`, `phylokernelmixrate.h`, `phylokernelsafe.h`, `phylokernelsitemodel.h`). Representative example, `tree/phylokernelnew.h:2386`:

```cpp
#pragma omp parallel for schedule(dynamic,1) num_threads(num_threads) \
        reduction(+:all_lh,all_df,all_ddf,...)
for (...) {
    ...
    freq.load_a(&ptn_freq[ptn]);                     // every thread, every pattern
    ...
}
```

Every thread, on every pattern, on every Newton-Raphson branch-length step, on every NNI evaluation, reads `ptn_freq[ptn]`. This is the inner loop of the inner loop.

---

## 2. The Linux memory rule that bites us: first-touch

### Virtual vs. physical memory (one-paragraph refresher)

When a program calls `malloc` (or `posix_memalign`, or `new`), the OS doesn't immediately give you physical RAM. It gives you a **virtual address range** — a promise. Physical RAM is only allocated when the program actually **writes** to that address. The first write triggers a page fault, and the OS responds by mapping a real 4 KB page of RAM behind the address.

### NUMA in one paragraph

A modern multi-socket server has memory controllers on each socket. RAM physically attached to socket 0's controller is **local** to socket 0 and **remote** to socket 1. Local reads take ~80 ns; remote reads cross the inter-socket link (UPI on Intel, Infinity Fabric on AMD EPYC) and cost ~140–200 ns — roughly 2× slower. A NUMA "node" = one socket's local memory.

### First-touch placement

The Linux default policy says: **when a page faults in, allocate it on the NUMA node of whichever CPU triggered the fault.** Whoever writes first wins; the page lives there until freed.

Practical consequence: where you put your **first write** decides where the data lives for the rest of the program. The `malloc` call is irrelevant. The first read is irrelevant. The first **write** is everything.

---

## 3. The bug — all roads lead to socket 0

### Step-by-step what actually happens

1. Master thread starts. It is bound to a core on socket 0 (this is what `OMP_PROC_BIND=close` does — packs threads starting from socket 0).

2. Master thread calls `aligned_alloc<double>(maxptn)` (`tree/phylotree.cpp:942`). This is just a `posix_memalign` wrapper (`tree/phylotree.h:95`). **No physical pages yet** — only virtual addresses reserved.

3. Master thread eventually calls `computePtnFreq()`. The serial `for` loop writes `ptn_freq[0]`, then `ptn_freq[1]`, etc. Each write to a *new* 4 KB page triggers a page fault. Linux first-touch policy maps every one of those pages to **socket 0's memory**, because the master thread runs on socket 0.

4. Same story for `ptn_invar[]` — `computePtnInvar()` starts with `memset(ptn_invar, 0, maxptn*sizeof(double))` on the master, pinning all its pages to socket 0.

5. Same story for the alignment's pattern data: `buildPattern()` walks every site sequentially on the master thread and pushes patterns onto a `std::vector<Pattern>`. All those allocations get first-touched on socket 0.

6. Tree-search begins. IQ-TREE forks an OpenMP team. With `OMP_NUM_THREADS=128` on Setonix, that's 64 threads on socket 0 + 64 threads on socket 1.

7. Inner LH loop runs. The 64 threads on socket 0 read `ptn_freq[]` from local memory — fast. The 64 threads on socket 1 read the *same* `ptn_freq[]` from socket 0's memory across the inter-socket link — **slow**, on every iteration, of every NNI, of every model fit.

### Code evidence (file : line)

| Buffer | Allocated | First serial write (the bug) | Hot read site (parallel) |
|---|---|---|---|
| `ptn_freq` | `tree/phylotree.cpp:942` (`posix_memalign`) | `tree/phylotreesse.cpp:543-546` (master `for`) | `tree/phylokernelnew.h:2386,3633,...` |
| `ptn_invar` | `tree/phylotree.cpp:948` | `tree/phylotreesse.cpp:571` (master `memset`) | LH kernels (e.g. `phylokernelnew.h:3182`) |
| Pattern vector | inside `addPatternLazy` | `alignment/alignment.cpp:2376` (master `for`) | indirectly via `aln->getPattern()` |

**Confirmed by grep:** zero occurrences of `numa_alloc`, `first_touch`, `numactl`, or any NUMA-aware code anywhere in `tree/` or `utils/`. Nothing in IQ-TREE attempts to control page placement.

---

## 4. Why your existing flags don't save you

The current Setonix and Gadi job scripts use:

```bash
export OMP_PROC_BIND=close
export OMP_PLACES=cores
numactl --localalloc ./iqtree3 ...
```

This works fine **at ≤32 threads on Setonix** (≤26 on Gadi) because all threads sit on one socket and every read is local anyway. It breaks at higher thread counts because:

- **`--localalloc`** says: "allocate each new page on the NUMA node of whoever first-touches it." That is *exactly* the default first-touch policy, and the master thread is doing all the first-touching. Every page lands on socket 0. `--localalloc` doesn't help.

- **`OMP_PROC_BIND=close`** says: "pack worker threads onto cores starting near the master." Threads 0–63 land on socket 0 (good). Threads 64–127 overflow onto socket 1 (forced — there's no more room on socket 0). Their reads of `ptn_freq[]` go remote. Binding policy doesn't help when the data is already mis-placed.

- **Master thread placement is the root cause.** No combination of `OMP_PROC_BIND` + `--localalloc` can fix data that was first-touched serially.

### The observed symptom

From `CHANGELOG.md`: at 128T on Setonix, GCC/libgomp wall time is **1.9× slower than at 8T**. That's not load imbalance or cache pressure — those would scale gradually. The 64→128 cliff is a classic NUMA signature: half your threads suddenly start crossing the socket boundary, and one of the hottest reads (`ptn_freq`) is on the wrong side.

---

## 5. The fix

### What to change

Replace the master-thread serial init loops with parallel-static loops that match the read pattern of the LH kernel.

**`tree/phylotreesse.cpp:537` — `computePtnFreq()`:**

```cpp
void PhyloTree::computePtnFreq() {
    if (ptn_freq_computed) return;
    ptn_freq_computed = true;
    size_t nptn   = aln->getNPattern();
    size_t maxptn = get_safe_upper_limit(nptn) +
                    get_safe_upper_limit(model_factory->unobserved_ptns.size());

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t ptn = 0; ptn < maxptn; ptn++)
        ptn_freq[ptn] = (ptn < nptn) ? (double)(*aln)[ptn].frequency : 0.0;
}
```

**`tree/phylotreesse.cpp:571` — replace `memset(ptn_invar, 0, ...)`:**

```cpp
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < maxptn; i++)
        ptn_invar[i] = 0.0;
```

(The state-dependent fill below the memset can stay serial — what matters is that the **first write** to each page happens in parallel.)

### Why those two changes are enough

`ptn_freq[]` and `ptn_invar[]` are the only per-pattern arrays in IQ-TREE that have **explicit serial init code** running on the master thread before the parallel region. The other big buffers (`_pattern_lh`, `_pattern_lh_cat`, `_pattern_scaling`, `theta_all`, `buffer_scale_all`, `central_partial_lh`) are first-written *inside* the parallel LH kernels themselves, so first-touch already happens in parallel for those. The two listed above are the lone serial first-touchers.

The alignment's `Pattern` vector (built in `buildPattern()`) is a separate, harder problem — it's a `std::vector<Pattern>` populated via `push_back`, not a flat array. Fixing it requires restructuring the data layout, which is a bigger change. The two-loop fix above gets the LH inner-loop hot read (`ptn_freq`); the pattern data itself is read less frequently and through a vector accessor that already costs more than the NUMA penalty.

### Why `schedule(static)`

- **Same idiom as the file already uses.** `tree/phylotreesse.cpp:267` (`computeTipPartialLikelihood`) already wraps a similar init loop in `#pragma omp parallel for schedule(static)` inside `#ifdef _OPENMP`. The patch matches existing code style — no new dependencies, no new build flags.

- **Static gives perfect locality if the LH loop is also static.** Some hot loops in `phylokernelnew.h` use `schedule(static)`; others use `schedule(dynamic,1)` for load balancing. With static init:
  - For static LH loops → thread `t` first-touches the same range it later reads → 100% local hits.
  - For dynamic LH loops → thread `t` first-touches a contiguous range, but reads jump around → ~50% local hits on average. Still vastly better than 0% for socket-1 threads with the current code.

- **Works with both `OMP_PROC_BIND=close` and `=spread`.** Under `spread`, threads are interleaved across sockets at thread-ID intervals, so the static partition automatically distributes pages evenly across NUMA nodes.

### What the fix doesn't cover

- The alignment pattern vector itself (read indirectly, less hot).
- The bootstrap-replicate code paths in `iqtree.cpp:2804-2810, 2888-2894` that reset `ptn_freq_computed = false` and recall `computePtnFreq()` — those benefit automatically because the patched function is parallel.
- Cache-line false sharing, NUMA-aware tree-traversal scheduling, etc. — separate concerns.

---

## 6. Verification plan (before / after)

### A. Confirm the diagnosis without code changes

Run the existing 128T case with `--interleave=all`. This forces the OS to round-robin newly faulted pages across both NUMA nodes regardless of which thread touches them first.

```bash
OMP_NUM_THREADS=128 OMP_PROC_BIND=spread OMP_PLACES=cores \
numactl --interleave=all ./iqtree3 -s mega_dna.phy -nt 128 -seed 1
```

Compare against the current `--localalloc` 128T baseline. If `--interleave=all` recovers a meaningful fraction of the 1.9× regression, that's empirical confirmation that NUMA placement (not OpenMP scheduling, not cache, not memory bandwidth saturation) is the dominant cross-socket bottleneck.

Expected outcome: `--interleave=all` should land somewhere between current 128T and the linear-scaling target. Not as good as the patched code (because pages are still randomly placed), but provably better than the master-touched baseline.

### B. After the patch

Rebuild from `build-profiling/` (already configured for gcc-14 on Setonix):

```bash
cd /scratch/pawsey1351/asamuel/iqtree3/build-profiling
make -j
```

Then run the same 128T case with the existing flags:

```bash
OMP_NUM_THREADS=128 OMP_PROC_BIND=spread OMP_PLACES=cores \
numactl --cpunodebind=0,1 --membind=0,1 ./iqtree3 -s mega_dna.phy -nt 128 -seed 1
```

Should beat both the current baseline and the `--interleave=all` baseline at 128T. Should be neutral to slightly faster at 8T (the parallel init has a tiny fork-join cost, paid once).

### C. Correctness

The patch only changes **execution order** of independent writes (each `ptn_freq[ptn] = ...` writes a different array slot). No data races. Bit-identical output.

---

## 7. Glossary

| Term | What it means here |
|---|---|
| Pattern | A unique column of the alignment after deduplication. |
| Pattern frequency | How many original sites collapsed onto a given pattern. |
| `ptn_freq[i]` | Flat `double` array holding pattern `i`'s frequency. Hot inner-loop read. |
| First-touch | Linux policy: a freshly malloc'd page is physically placed on the NUMA node of whichever CPU first writes to it. |
| NUMA node | A socket's local memory + the cores attached to it. Two-socket box = two NUMA nodes. |
| Local read | CPU on socket X reads memory on socket X. Fast (~80 ns). |
| Remote read | CPU on socket X reads memory on socket Y. Slow (~140–200 ns). |
| `OMP_PROC_BIND=close` | OpenMP places threads near the master, packing onto sockets in order. |
| `OMP_PROC_BIND=spread` | OpenMP interleaves threads evenly across sockets. |
| `numactl --localalloc` | Use first-touch policy (the default). Doesn't help if the master is the toucher. |
| `numactl --interleave=all` | Round-robin pages across NUMA nodes at fault time. Workaround, not a fix. |

---

## 8. References in the source

- `alignment/alignment.cpp:2219` — `Alignment::buildPattern` (serial)
- `tree/phylotreesse.cpp:537` — `PhyloTree::computePtnFreq` (serial — patch target)
- `tree/phylotreesse.cpp:549` — `PhyloTree::computePtnInvar` (serial memset — patch target)
- `tree/phylotreesse.cpp:267` — existing `#pragma omp parallel for schedule(static)` idiom to mirror
- `tree/phylotree.cpp:925-948` — `aligned_alloc` calls for the per-pattern buffers
- `tree/phylotree.h:95` — `aligned_alloc` template (= `posix_memalign` on Linux, no zeroing)
- `tree/phylokernelnew.h:2386, 3633, 3595, ...` — hot LH loops that read `ptn_freq[]`
- `CHANGELOG.md` § 2026-05-05 — prior analysis of the binding gap
