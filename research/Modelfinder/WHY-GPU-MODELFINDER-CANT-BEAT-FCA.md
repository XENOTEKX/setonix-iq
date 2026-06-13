# Why a single GPU can't beat a 16-node CPU cluster at ModelFinder — in plain terms

**Author:** as1708 / Claude Opus 4.8, 2026-06-10. Plain-language companion to PART V (the technical version).
**Audience:** the core concepts, simply, with the evidence behind each claim.

---

## The one-sentence answer

**ModelFinder is a "many independent small jobs" problem, which rewards having many workers; a 16-node CPU
cluster has 16 workers, and a single GPU — at the data sizes that matter — can only act as *one* worker, because
one model already fills it. So 1 GPU ≈ 1 node, and 1 node loses to 16. It is a mismatch between the *shape* of
the problem and the *shape* of the hardware, not a bug we can fix.**

---

## 1. What ModelFinder actually is

ModelFinder tries ~224 candidate substitution models on a **fixed tree** and picks the best by BIC. Each model
is scored independently of the others. So it is **embarrassingly parallel across models** — 224 separate jobs
that don't talk to each other.

**Analogy:** 224 exam papers to grade. The papers are independent. The fastest way to grade them is to hand
them out to as many graders as you have.

---

## 2. The two shapes of parallel hardware — and why it matters here

There are two completely different ways to "go parallel," and ModelFinder only benefits from one of them:

- **BREADTH** = many independent workers, each doing a *whole* job. → A CPU cluster. 16 nodes = 16 workers.
  Each node grabs ~14 of the 224 models and grinds through them. Total time ≈ "14 models" worth, not "224."
- **DEPTH** = one job split into thousands of tiny identical pieces done at once. → A GPU. A GPU is *not* "many
  independent workers." It is one machine with thousands of threads that must all do the *same* step together.
  It is brilliant at taking **one** model's likelihood and splitting its ~100,000 alignment sites across
  thousands of threads — making that **one** model fast.

**The mismatch:** ModelFinder's parallelism is BREADTH (across the 224 models). A GPU's parallelism is DEPTH
(within one model). They don't line up.

**Analogy:** the CPU cluster is 16 graders working in parallel. The GPU is one grader with 16 hands — but all
16 hands must work on the *same paper*. Great for finishing one paper fast; useless for grading 224 at once.

---

## 3. The single measurement that explains everything

To use a GPU on ModelFinder you must run the likelihood calculation, and that calculation turns out to be
**memory-bound**: the GPU spends almost all its time *waiting for numbers to arrive from memory*, not computing.
We measured this directly (Nsight Compute, the real kernel):

| | memory bandwidth actually achieved |
|---|---|
| **One V100 GPU** | **308 GB/s** (only ~⅓ of its 900 GB/s peak — it can't go faster, it's waiting on memory) |
| **One Sapphire-Rapids CPU node** | **~350 GB/s** |

**→ For this calculation, one GPU ≈ one CPU node. They are the same speed.** The GPU's huge raw compute power
(its FLOPS) is irrelevant, because the work is *fetching* numbers, not crunching them — and at fetching, one
GPU and one node are equals.

FCA runs on **16 nodes**. So in raw terms: **1 GPU ≈ 1 node, and you're up against 16 of them.**

---

## 4. "Then just put many models on the GPU at once!" — why that fails

The obvious fix: if one model only uses part of the GPU, pack several models on at once to recover breadth.
This is the idea I chased hardest (and it's the right instinct). **It fails because, at real data sizes, one
model already fills the entire GPU.**

A GPU has ~80 "slots" (streaming multiprocessors). When you run one model's likelihood at 100,000 sites, the
work is split into **~380 blocks** — already **~5× more than the 80 slots**. The GPU is full. Adding a second
model doesn't run it in parallel; it just makes it **wait in line.** No speed-up.

(There *is* a regime where packing works — when each model is tiny, e.g. a 1,000-site sample, the GPU is mostly
empty and you can pack ~all 224 in. But that only ranks models roughly; you still have to fully score the
finalists at full size, and that's the part that dominates — and that part can't be packed.)

---

## 5. We proved this is fundamental — three different ways, same wall

This wasn't one failed idea. We attacked the problem from three independent directions and every one hit the
*same* physics:

1. **Pack many models on the GPU (batching):** blocked — one model already fills the GPU at full size.
2. **"It'll pay off at huge 1M-site data, where the GPU's memory bandwidth dominates":** tested, **falsified** —
   the kernel never becomes bandwidth-limited; it stays memory-*latency*-limited at every size (it's waiting,
   not streaming). 308 GB/s at 100K, 300K — flat.
3. **Speed up the tree-search half of a full run instead:** same likelihood kernel, same full GPU, same wall.

Three probes, one answer: **at full data on one GPU, this calculation runs at about one CPU node's speed.** That
is a *robust, triangulated finding*, not three separate failures.

---

## 6. What JOLT genuinely won — and why it doesn't change the verdict

JOLT is real and worth keeping. IQ-TREE optimizes a model's branch lengths **one branch at a time, in sequence**
(~197 steps in a chain) — which a GPU cannot parallelize. JOLT rewrites this to optimize **all branches at once**
and reaches the *exact same answer* (same maximum-likelihood, machine-precision identical), in ~27 parallel
steps instead of a 197-deep chain. Measured: **4.8× faster per model**, correct, bit-identical.

But 4.8×-per-model is a **DEPTH** win, and ModelFinder is a **BREADTH** problem. Do the arithmetic:
- 16 nodes: 224 models ÷ 16 ≈ **14 models per node, in sequence** → wall ≈ 14 model-times.
- 1 GPU: **224 models in sequence** (one at a time) → wall ≈ 224 model-times.
- Even if each GPU model were 4.8× faster: 224 ÷ 4.8 ≈ **47 model-times — still ~3.3× slower than the cluster's
  14.** And since per-model is really a *wash* (§3), it's ~16× slower.

**Breadth (16×) beats depth (4.8×).** This matches what we measured: at 100K, GPU ModelFinder = 3493 s vs the
CPU's ~259 s ≈ **13.5× slower** — right in line with the "~16×" the theory predicts. **Theory matches
measurement** — that's the scientific-standard confirmation.

---

## 7. What went wrong (honest introspection)

- **The original bet was that the GPU's raw power would beat the cluster.** That bet quietly assumed the work was
  *compute*-bound. It isn't — it's *memory*-bound, so the only thing that counts is memory bandwidth, and one GPU
  has about one node's worth. The premise was wrong from the start; we just didn't have it measured.
- **ModelFinder is the worst-case problem for a single GPU.** It's pure breadth (many independent models), which
  is exactly what a CPU cluster is built for and exactly what a single GPU is worst at. We picked the one
  workload where the hardware mismatch is maximal.
- **The clever ideas (batching, tiling, tree-search) weren't wasted — they were the proof.** Each one independently
  confirmed the same wall. That's *why* we can now say "fundamental" with evidence instead of guessing.
- **JOLT was the right thing to build and is genuinely good** — it just answers a different question ("make one
  model's optimization fast") than the one ModelFinder asks ("score 224 models").

---

## 8. The honest bottom line

**A single GPU cannot beat a 16-node CPU cluster at ModelFinder. This is a hardware-shape vs problem-shape
mismatch (a breadth problem on a depth machine, with a memory-bound kernel that makes 1 GPU ≈ 1 node), not an
implementation gap a better kernel would close.** Multi-GPU would only be the GPU version of FCA's breadth
(M GPUs = M workers) — and you've said you won't go there unless a single GPU first beats FCA, which it can't
for this problem.

**Where a GPU *would* genuinely win** (different problems, not ModelFinder breadth):
- **One enormous tree** that can't be split across CPU ranks (there's only one tree) — here JOLT's all-branches-
  at-once optimization beats the CPU's one-branch-at-a-time chain, and breadth doesn't help the CPU. *Untested —
  the one regime not falsified.*
- **JOLT as a faster single-model optimizer** inside the existing tool, for users on one GPU. Real, banked, modest.
