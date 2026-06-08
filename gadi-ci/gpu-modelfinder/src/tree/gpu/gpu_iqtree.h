#ifndef IQTREE_GPU_IQTREE_H
#define IQTREE_GPU_IQTREE_H

// C-linkage entry points for the in-tree CUDA GPU module (Phase G.1+).
//
// These declarations are compiled into the `iqtree_gpu` static library, which
// is built ONLY when the CMake option IQTREE_GPU is ON (see CMakeLists.txt:
// `if(IQTREE_GPU) enable_language(CUDA) ... add_library(iqtree_gpu ...)`).
//
// We use C linkage so the nvcc-compiled .cu translation units and the host C++
// objects (main/main.cpp etc.) agree on symbol names without C++ mangling.
//
// Host callers must guard the include and the call with `#ifdef IQTREE_GPU`
// (the macro comes from the generated <iqtree_config.h>), so a CPU-only build
// neither references these symbols nor links the GPU library.

#ifdef __cplusplus
extern "C" {
#endif

// Phase G.1.0 build-scaffold diagnostic. Enumerates the CUDA device(s), prints
// device name / compute capability / VRAM, launches a trivial kernel, verifies
// it executed (marker readback) and that cudaGetLastError() == cudaSuccess,
// then returns. Pure side effect (stdout/stderr); never throws. A no-op-ish
// failure path just prints a message if no device is present.
void iqtree_gpu_diag();

// Phase G.2.0a — clean-room GPU log-likelihood cross-check launcher.
//
// Runs the validated K1 eigen-space postorder partial-likelihood sweep on the GPU from host-prepared,
// already-extracted arrays (eigen factors, per-child echild, compact tip states, per-internal-node child
// descriptors, pattern frequencies) and returns the total tree log-likelihood
//   tree_lh = sum_ptn ptn_freq[ptn] * log|lh_ptn|.
// It is the in-tree analog of the standalone gpu_k1_lnl.cu harness; the host caller
// (PhyloTree::gpuLnLCrossCheckOnce) builds the inputs from the LIVE model/tree/alignment and compares the
// result against IQ-TREE's own curScore. NORM_LH / unscaled (AA-100K, leafNum<2000); FP64; native-20 path
// works for nstates in {4,20} (no 32-pad). Pure compute + a few cudaMalloc/Memcpy; returns NaN on any
// internal CUDA error (caller treats NaN as "cross-check unavailable", never aborts the CPU run).
//
// Descriptor arrays are length nInternal (postorder) except the *_child* arrays which are nInternal*3
// (up to 3 children per node; unused slots = -1). echild is indexed [child_node_id][cat][x][i] with stride
// ncat*nstates*nstates; the root node's echild slot is unused. outSlot/childSlot index the partials arena
// in units of (ncat*nstates*nptn). childIsLeaf!=0 => use childLeaf (taxon id) + the tip array; else childSlot.
//
// G.2.0b: if out_patlh != NULL it is filled with the per-pattern log-likelihoods log|lh_ptn| (nptn entries,
// pattern order) — exactly what IQ-TREE's _pattern_lh[] holds under NORM_LH (no scaling). This lets the
// Branch-pointer override (PhyloTree::computeLikelihoodBranchGPU) feed the host _pattern_lh[] so the
// downstream (unconditional) computeLogLVariance()/computePatternLikelihood() produce the correct s.e.
double gpu_lnl_crosscheck(
    int nstates, int nptn, int ncat, int ntax, int nnodes, int nInternal,
    const double* Uinv,          // nstates*nstates (row-major; IQ-TREE inverse eigenvectors)
    const double* UinvRowSum,    // nstates (row sums of Uinv, for fully-ambiguous tip states)
    const double* freq,          // nstates (state frequencies, for the root reduction)
    const double* catProp,       // ncat (category proportions / weights)
    const double* echild,        // nnodes * ncat*nstates*nstates
    const unsigned char* tip,    // ntax * nptn (compact states; >=nstates means ambiguous)
    const double* ptn_freq,      // nptn (pattern multiplicities)
    const int* desc_isRoot,      // nInternal
    const int* desc_nchild,      // nInternal
    const int* desc_outSlot,     // nInternal (partials-arena slot; root = -1)
    const int* desc_childNode,   // nInternal*3 (child node id -> echild base; -1 = unused)
    const int* desc_childIsLeaf, // nInternal*3
    const int* desc_childLeaf,   // nInternal*3 (taxon id if leaf child)
    const int* desc_childSlot,   // nInternal*3 (partials slot if internal child)
    double* out_patlh);          // nptn (optional; per-pattern log|lh_ptn| if non-NULL) — G.2.0b

// Phase G.2.1a — clean-room single-edge branch-length derivative launcher (K2 in-tree). The descriptor list
// covers BOTH subtrees split by the central edge (two sub-roots = the edge endpoints), all entries isRoot=0 so
// every internal node (incl. the two endpoints) writes its eigen-space partial to its slot. nodeSlot/dadSlot
// index the two endpoint partials. Returns df = d(lnL)/dt (= Σ ptn_freq·d1/lh); *out_ddf = the second
// derivative (Σ ptn_freq·(d2/lh−(d1/lh)²)); *out_lnL = the tree lnL at the central length t (free cross-check).
// val0/val1/val2 are built on-device from eval/catRate/catProp/t. Returns NaN on CUDA error.
double gpu_derv_crosscheck(
    int nstates, int nptn, int ncat, int ntax, int nnodes, int nInternal,
    const double* Uinv, const double* UinvRowSum, const double* freq, const double* catProp,
    const double* echild, const unsigned char* tip, const double* ptn_freq,
    const int* desc_isRoot, const int* desc_nchild, const int* desc_outSlot,
    const int* desc_childNode, const int* desc_childIsLeaf, const int* desc_childLeaf, const int* desc_childSlot,
    int nodeSlot, int nodeLeafTax,   // node endpoint: slot>=0 if internal, else leaf taxon id (G.2.1b)
    int dadSlot,  int dadLeafTax,    // dad  endpoint: slot>=0 if internal, else leaf taxon id (G.2.1b)
    const double* eval,          // nstates (eigenvalues)
    const double* catRate,       // ncat (per-category rates)
    double t,                    // central branch length
    double* out_ddf,             // out: second derivative
    double* out_lnL);            // out: tree lnL at t

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IQTREE_GPU_IQTREE_H
