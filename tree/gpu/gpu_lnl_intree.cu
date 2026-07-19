// gpu_lnl_intree.cu — Phase G.2.0a: in-tree clean-room GPU log-likelihood cross-check launcher.
//
// The device side is the VALIDATED K1 eigen-space postorder kernel from the standalone harness
// gadi-ci/gpu-modelfinder/gpu_k1_lnl.cu (which matches the G.0 BEAGLE oracle to rel ~1e-12 for g4/r8/r10/g1).
// Only the host plumbing differs: instead of parsing files, the caller (PhyloTree::gpuLnLCrossCheckOnce)
// extracts eigen factors / rates / topology / tip states / pattern frequencies from the LIVE IQ-TREE objects
// and passes them in; the reduction here is ptn_freq-weighted (IQ-TREE stores compressed patterns, vs the
// harness's uncompressed weight-1 sites): tree_lh = sum_ptn ptn_freq[ptn] * log|lh_ptn|.
//
// MATH (eigen space; reproduces computePartialLikelihoodSIMD):
//   leaf state s:   L[c][i] = Uinv[i][s]            (column s of U^-1; ambiguous -> row-sum over s)
//   internal node:  pk[x]   = sum_i echild_k[c][x][i] * L_child_k[c][i]
//                   prod[x] = prod_k pk[x]
//                   L[c][r] = sum_x Uinv[r][x] * prod[x]
//   root node:      lh_ptn  = sum_c prop_c * sum_x freq[x] * prod_root_c[x]
//   echild_k[c][x][i] = U[x][i] * exp(eval[i] * rate_c * len_k)   (built host-side, passed in)
// FP64, UNSCALED, native-20 (nstates in {4,20}). Returns NaN on any CUDA error (caller never aborts on that).
//
// Built into the iqtree_gpu static lib by CMake only when IQTREE_GPU=ON.
#include <cuda_runtime.h>
#include "gpu_iqtree.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <functional>   // G.4.2: recursive DFS lambdas in the JOLT launcher
#include <mutex>        // G.4.2: serialize GPU access — ModelFinder is across-model OpenMP-parallel
#include <cstring>      // G.7.1: memcpy for the pattern-tiling tip-slice gather
#include <omp.h>        // CONSTCACHE: omp_in_parallel() -> bypass the shadow in parallel regions (mirrors screen-cache)
#include <cstdlib>      // G.7.1: getenv/atoi for JOLT_NTILE
#include <chrono>       // --jolt-diag (A3): host-side timer for the per-eval echild rebuild tax

#define NS_MAX 20

// ---- device model constants (set per cross-check call; one tree, one model) ----
__constant__ double g_Uinv[NS_MAX*NS_MAX];
__constant__ double g_U[NS_MAX*NS_MAX];   // G.4.2: eigenvectors (evec), needed by the JOLT preorder kernel kj_pre
__constant__ double g_UinvRowSum[NS_MAX];
__constant__ double g_freq[NS_MAX];
__constant__ double g_catw[64];
// G.2.1a single-edge derivative coefficients (per cat,state at the central branch length t):
__constant__ double g_val0[64*NS_MAX];   // exp(eval[x]*rate_c*t) * prop_c
__constant__ double g_val1[64*NS_MAX];   // (rate_c*eval[x]) * val0
__constant__ double g_val2[64*NS_MAX];   // (rate_c*eval[x]) * val1
__constant__ double g_rscale[64];        // G.4.2: per-cat edge scale b_e/(r_k*w_k) for the +R/alpha rate-grad numerator

// [CONSTCACHE 2026-07-15] gpu_jolt_optimize + its inner eval fns re-upload the MODEL constants g_Uinv/g_U/
// g_UinvRowSum/g_freq/g_catw on EVERY call, though they change only per-eigensystem (nsys DNA-1M -m MF: 9.06M
// cudaMemcpyToSymbol = 209s host-API + launch/idle churn). This guard shadows the last-uploaded bytes per symbol id
// and SKIPS the toSymbol when unchanged -- correct because these __constant__ symbols are written ONLY through this
// guard (5 canonical sites + the free-Q qApply lambda), so the device already holds that value. getenv JOLT_CONSTCACHE
// gates it; UNSET => always upload => byte-identical to prod. Bypasses omp_in_parallel() (mirrors the screen-cache
// guard, phylotreegpu.cpp:1442) so concurrent candidate evals cannot race the process-global shadow. g_val0/1/2 are
// branch-dependent (contain edge length t) => NOT cached; they always upload.
static long g_cc_red[8] = {0}, g_cc_tot[8] = {0};
static inline bool cc_skip_toSymbol(int id, const void* src, size_t sz){
    static const bool on = (getenv("JOLT_CONSTCACHE") != nullptr);
    if(!on) return false;
    if(omp_in_parallel()) return false;
    if(id < 0 || id >= 8) return false;
    static std::vector<char> shadow[8];
    std::vector<char>& s = shadow[id];
    g_cc_tot[id]++;
    bool skip = (s.size()==sz && memcmp(s.data(), src, sz)==0);
    if(skip) g_cc_red[id]++;
    else s.assign((const char*)src, (const char*)src + sz);
    if(id==0 && (g_cc_tot[0] & ((1L<<16)-1))==0)
        fprintf(stderr,"[CONSTCACHE] red/tot Uinv:%ld/%ld U:%ld/%ld URS:%ld/%ld freq:%ld/%ld catw:%ld/%ld\n",
                g_cc_red[0],g_cc_tot[0],g_cc_red[1],g_cc_tot[1],g_cc_red[2],g_cc_tot[2],g_cc_red[3],g_cc_tot[3],g_cc_red[4],g_cc_tot[4]);
    return skip;
}
#define CC_TOSYM(id, sym, src, sz) do{ if(!cc_skip_toSymbol((id),(src),(sz))) GCK(cudaMemcpyToSymbol((sym),(src),(sz))); }while(0)

// per-child probability-space contribution: prod[x] *= sum_i echild[c][x][i] * L_child[c][i]
__device__ __forceinline__ void accum_child(double* prod, int ns, int c, int ptn, int nptn,
        const double* __restrict__ ec, const double* __restrict__ p, const unsigned char* __restrict__ t) {
    const double* ecc = ec + (size_t)c*ns*ns;
    if (p) {                              // internal child: read its eigen-space partial (coalesced over ptn)
        const double* pc = p + (size_t)(c*ns)*nptn + ptn;
        for (int x=0;x<ns;x++){ double v=0.0;
            for (int i=0;i<ns;i++) v += ecc[x*ns+i]*pc[(size_t)i*nptn];
            prod[x]*=v; }
    } else {                              // leaf child: L[i] = column s of U^-1 (row-sum if ambiguous)
        int s = t[ptn];
        for (int x=0;x<ns;x++){ double v=0.0;
            for (int i=0;i<ns;i++){ double Li = (s<ns)? g_Uinv[i*ns+s] : g_UinvRowSum[i]; v += ecc[x*ns+i]*Li; }
            prod[x]*=v; }
    }
}

// one internal node (or the root) for all patterns; one thread per pattern.
__global__ void k1_node(int ns, int nptn, int ncat, int isRoot, double* __restrict__ out, double* __restrict__ patlh,
        int nchild,
        const double* ec0, const double* p0, const unsigned char* t0,
        const double* ec1, const double* p1, const unsigned char* t1,
        const double* ec2, const double* p2, const unsigned char* t2,
        double pinv = 0.0, const double* base_invar = nullptr) {   // A3 (+I): add pinv*base_invar at root when base_invar!=null (else byte-identical)
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    double lh = 0.0;
    for (int c=0;c<ncat;c++){
        double prod[NS_MAX];
        for (int x=0;x<ns;x++) prod[x]=1.0;
        accum_child(prod,ns,c,ptn,nptn,ec0,p0,t0);
        if (nchild>1) accum_child(prod,ns,c,ptn,nptn,ec1,p1,t1);
        if (nchild>2) accum_child(prod,ns,c,ptn,nptn,ec2,p2,t2);
        if (isRoot){
            double s=0.0;
            for (int x=0;x<ns;x++) s += g_freq[x]*prod[x];
            lh += g_catw[c]*s;
        } else {
            double* o = out + (size_t)(c*ns)*nptn + ptn;
            for (int r=0;r<ns;r++){ double v=0.0;
                for (int x=0;x<ns;x++) v += g_Uinv[r*ns+x]*prod[x];
                o[(size_t)r*nptn]=v; }
        }
    }
    if (isRoot) patlh[ptn] = log(base_invar ? fabs(lh)+pinv*base_invar[ptn] : fabs(lh));   // A3 (+I): +pinv*base_invar
}

// ============================================================================================================
// G.8.0 — N-class PROFILE-MIXTURE lnL (the "K1-for-mixtures" gate). Each regime r = m*ncat + c is an INDEPENDENT
// Felsenstein sweep with class m's eigen (echild_m,c, Uinv_m) and gamma cat c; regimes never mix until the ROOT
// fold  L_p = Σ_r w_r·(π_m · prod_r),  w_r = weight_m · catProp_c.  Per-class Uinv/freq/rowsum live in GLOBAL
// memory (N·ns·ns overflows the 64KB __constant__ budget at MEOW80's 320 regimes). Separate kernel from k1_node
// so the validated single-model path is byte-unchanged. One thread/pattern; straightforward r-loop (correctness
// gate — a low-register class mapping is a later perf optimisation; this is a one-shot diagnostic).
// ============================================================================================================
__device__ __forceinline__ void accum_child_mix(double* prod, int ns, int r, int ptn, int nptn,
        const double* __restrict__ ec, const double* __restrict__ p, const unsigned char* __restrict__ t,
        const double* __restrict__ Uinv_m, const double* __restrict__ UinvRowSum_m) {
    const double* ecc = ec + (size_t)r*ns*ns;
    if (p) {                                   // internal child: its eigen-space partial for regime r
        const double* pc = p + (size_t)r*ns*nptn + ptn;
        for (int x=0;x<ns;x++){ double v=0.0;
            for (int i=0;i<ns;i++) v += ecc[x*ns+i]*pc[(size_t)i*nptn];
            prod[x]*=v; }
    } else {                                   // leaf child: L[i] = column s of class-m U^-1 (row-sum if ambiguous)
        int s = t[ptn];
        for (int x=0;x<ns;x++){ double v=0.0;
            for (int i=0;i<ns;i++){ double Li = (s<ns)? Uinv_m[i*ns+s] : UinvRowSum_m[i]; v += ecc[x*ns+i]*Li; }
            prod[x]*=v; }
    }
}

__global__ void k1_node_mix(int ns, int nptn, int ncat, int nmix, int isRoot,
        double* __restrict__ out, double* __restrict__ patlh,
        const double* __restrict__ d_Uinv,        // [nmix][ns*ns]
        const double* __restrict__ d_UinvRowSum,  // [nmix][ns]
        const double* __restrict__ d_freq,        // [nmix][ns]
        const double* __restrict__ d_wreg,        // [nmix*ncat]  w_m * catProp_c
        double* __restrict__ out_lhcat,           // G.8.1: optional per-class L_{p,m}=w_m*Σ_c catProp_c*L_{p,m,c} [nmix][nptn]
        double pinv, const double* __restrict__ clsinv,   // A1 (+I): per-class invariant clsinv[m][ptn]=w_m*pinv*base_invar_m (chunk-local [nmix][nptn]); ADDED only at the root path when pinv>0, so pinv<=0 (clsinv may be nullptr) is BYTE-IDENTICAL
        int nchild,
        const double* ec0, const double* p0, const unsigned char* t0,
        const double* ec1, const double* p1, const unsigned char* t1,
        const double* ec2, const double* p2, const unsigned char* t2) {
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    int R = nmix*ncat;
    // G.8.2.3 — 2D-grid REGIME parallelisation. A 2D launch (gridDim.y==R) assigns each blockIdx.y ONE regime r,
    // so the GPU saturates on small nptn (the old nptn-only grid was ~2 blocks => ~0.3% V100 occupancy, the JOLTMix
    // bottleneck per nsys). A 1D launch (gridDim.y==1 — the root path, which sums lh over ALL regimes into patlh, and
    // any legacy caller) keeps the full in-thread r-loop. Non-root writes out[r*ns*nptn+..] are regime-disjoint => no race.
    int r0 = (gridDim.y>1) ? blockIdx.y : 0;
    int rN = (gridDim.y>1) ? (blockIdx.y+1) : R;
    double lh = 0.0, clh = 0.0;   // clh = running per-class accumulator (G.8.1) — root path only (always 1D)
    for (int r=r0;r<rN;r++){
        int m = r/ncat;
        const double* Uinv_m = d_Uinv + (size_t)m*ns*ns;
        const double* Urs_m  = d_UinvRowSum + (size_t)m*ns;
        double prod[NS_MAX];
        for (int x=0;x<ns;x++) prod[x]=1.0;
        accum_child_mix(prod,ns,r,ptn,nptn,ec0,p0,t0,Uinv_m,Urs_m);
        if (nchild>1) accum_child_mix(prod,ns,r,ptn,nptn,ec1,p1,t1,Uinv_m,Urs_m);
        if (nchild>2) accum_child_mix(prod,ns,r,ptn,nptn,ec2,p2,t2,Uinv_m,Urs_m);
        if (isRoot){
            const double* freq_m = d_freq + (size_t)m*ns;
            double s=0.0;
            for (int x=0;x<ns;x++) s += freq_m[x]*prod[x];
            double contrib = d_wreg[r]*s;   // w_m*catProp_c*(π_m·prod) = this regime's likelihood contribution
            // A1 (+I): add class m's invariant term ONCE per class (at its FIRST gamma cat, c==r-m*ncat==0). Folding it
            // into contrib makes BOTH the root sum (lh) and the per-class EM accumulator (clh) carry it: lh gets
            // Σ_m clsinv_m (the full root invariant pinv·Σ_m w_m·base_invar_m), out_lhcat[m] gets the per-class invariant.
            if (pinv > 0.0 && (r - m*ncat == 0)) contrib += clsinv[(size_t)m*nptn + ptn];
            lh += contrib;
            if (out_lhcat){                 // accumulate per class m over its ncat gamma cats, flush at c==ncat-1
                clh += contrib;
                if (r - m*ncat == ncat-1){ out_lhcat[(size_t)m*nptn + ptn] = clh; clh = 0.0; }
            }
        } else {
            double* o = out + (size_t)r*ns*nptn + ptn;
            for (int rr=0; rr<ns; rr++){ double v=0.0;
                for (int x=0;x<ns;x++) v += Uinv_m[rr*ns+x]*prod[x];
                o[(size_t)rr*nptn]=v; }
        }
    }
    if (isRoot) patlh[ptn] = log(fabs(lh));
}

// G.2.1a single-edge derivative (K2): theta = node_eig elementwise* dad_eig (t-independent eigen-space
// partials at the two edge endpoints); per pattern lh=Σ val0·theta, d1=Σ val1·theta, d2=Σ val2·theta;
// pdf = d1/lh = d log|lh| / dt, pddf = d2/lh − (d1/lh)^2 = d²log|lh|/dt². One thread/pattern.
__global__ void k2_derv(int ns, int nptn, int ncat,
        const double* __restrict__ node_eig, const double* __restrict__ dad_eig,
        double* __restrict__ pdf, double* __restrict__ pddf, double* __restrict__ patlh) {
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    double lh=0.0, d1=0.0, d2=0.0;
    for (int c=0;c<ncat;c++) for (int x=0;x<ns;x++){
        size_t o=(size_t)(c*ns+x)*nptn+ptn;
        double th=node_eig[o]*dad_eig[o]; int k=c*ns+x;
        lh+=g_val0[k]*th; d1+=g_val1[k]*th; d2+=g_val2[k]*th;
    }
    double inv=1.0/lh, r=d1*inv;
    pdf[ptn]=r; pddf[ptn]=d2*inv-r*r; patlh[ptn]=log(fabs(lh));
}

// G.2.1b — synthesize a LEAF endpoint's eigen-space directed partial (the tip vector, rate-independent):
// L[c][i] = Uinv[i][s] (column s of Uinv; UinvRowSum[i] if the state is ambiguous), replicated over all cats.
// Same slot layout as an internal eigen partial, so k2_derv reads node_eig/dad_eig uniformly.
__global__ void k_leaf_eig(int ns, int nptn, int ncat, const unsigned char* __restrict__ tipt, double* __restrict__ out){
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    int s = tipt[ptn];
    for (int c=0;c<ncat;c++) for (int i=0;i<ns;i++){
        double Li = (s<ns) ? g_Uinv[i*ns+s] : g_UinvRowSum[i];
        out[(size_t)(c*ns+i)*nptn+ptn] = Li;
    }
}

// G.8.1b — single-edge branch derivative for PROFILE MIXTURES: k2_derv generalised to the regime axis
// r = m*ncat + c (R = nmix*ncat). The central-edge derivative coefficients dval0/1/2 live in GLOBAL memory
// (per-class eigenvalues -> R*ns entries exceed the __constant__ 64-cat budget at C60/MEOW80, the same reason
// k1_node_mix moved Uinv/freq to global). node_eig/dad_eig are the per-regime eigen-space endpoint partials from
// k1_node_mix (isRoot=0). Per pattern: lh=Σ dval0·θ, d1=Σ dval1·θ, d2=Σ dval2·θ (θ=node_eig·dad_eig); the
// per-class weight w_m·catProp_c is folded into dval0 (= wreg[r]); π_m is already in the eigen-space partials.
__global__ void k2_derv_mix(int ns, int nptn, int R,
        const double* __restrict__ node_eig, const double* __restrict__ dad_eig,
        const double* __restrict__ dval0, const double* __restrict__ dval1, const double* __restrict__ dval2,
        double* __restrict__ pdf, double* __restrict__ pddf, double* __restrict__ patlh) {
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    double lh=0.0, d1=0.0, d2=0.0;
    for (int r=0;r<R;r++) for (int x=0;x<ns;x++){
        size_t o=(size_t)(r*ns+x)*nptn+ptn;
        double th=node_eig[o]*dad_eig[o]; int k=r*ns+x;
        lh+=dval0[k]*th; d1+=dval1[k]*th; d2+=dval2[k]*th;
    }
    double inv=1.0/lh, rr=d1*inv;
    pdf[ptn]=rr; pddf[ptn]=d2*inv-rr*rr; patlh[ptn]=log(fabs(lh));
}

// A3 (screener +I): the +I variant of k2_derv_mix — adds the branch/topology-INDEPENDENT invariant term
// pinv*baseinvar[ptn] to the per-pattern likelihood before log (mirrors kj_derv_fused vs kj_derv). The screener's
// catRate/catProp already carry the 1/(1-pinv) rescale (getRate/getProp for RateGammaInvar), so this additive term
// is the ONLY missing +I piece. Used by the screener move loop ONLY when pinv>0; pinv==0 stays on k2_derv_mix
// (bit-identical), so non-+I screening is UNCHANGED. baseinvar is the chunk slice (indexed 0..nptn-1 of the chunk).
__global__ void k2_derv_mix_inv(int ns, int nptn, int R,
        const double* __restrict__ node_eig, const double* __restrict__ dad_eig,
        const double* __restrict__ dval0, const double* __restrict__ dval1, const double* __restrict__ dval2,
        double pinv, const double* __restrict__ baseinvar,
        double* __restrict__ pdf, double* __restrict__ pddf, double* __restrict__ patlh) {
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    double lh=0.0, d1=0.0, d2=0.0;
    for (int r=0;r<R;r++) for (int x=0;x<ns;x++){
        size_t o=(size_t)(r*ns+x)*nptn+ptn;
        double th=node_eig[o]*dad_eig[o]; int k=r*ns+x;
        lh+=dval0[k]*th; d1+=dval1[k]*th; d2+=dval2[k]*th;
    }
    double Lp=fabs(lh)+pinv*baseinvar[ptn]; double inv=1.0/Lp, rr=d1*inv;
    pdf[ptn]=rr; pddf[ptn]=d2*inv-rr*rr; patlh[ptn]=log(Lp);
}

// ============================================================================================================
// TS RAKE BATCH (2026-06-29) — batched per-move screener folds. Each kernel is the CHARACTER-EXACT twin of its
// serial counterpart (k1_node / kj_pre_node(eigOut=1) / k2_derv_mix) with only (a) a blockIdx.y move selector that
// resolves per-move device pointers from the int descriptor arrays EXACTLY as the host ecP/plP/tpP lambdas do, and
// (b) a per-local-move output slot j*slotSz. The per-thread arithmetic + accumulation order are unchanged => each
// output double is bit-identical to the serial fold for that move; only the LAUNCH is collapsed (nB moves in one
// grid (GB,nB) that fills the SMs vs nB tiny under-occupied launches). m = batchStart + blockIdx.y. slotSz = the
// chunk eigen slot = ncat*ns*Pn (same stride d_partial/d_upper/n1eig/n2eig all use). Gated JOLT_TS_BATCHFOLD.
// ============================================================================================================
__global__ void screen_node2_batch(int ns, int nptn, int ncat, int batchStart,
        double* __restrict__ out, size_t slotSz,
        const double* __restrict__ d_echild, size_t ecStride,
        const double* __restrict__ d_partial, const unsigned char* __restrict__ d_tip,
        const int* __restrict__ n2a_ec, const int* __restrict__ n2a_slot, const int* __restrict__ n2a_leaf,
        const int* __restrict__ n2b_ec, const int* __restrict__ n2b_slot, const int* __restrict__ n2b_leaf) {
    int j = blockIdx.y; int m = batchStart + j;
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    const double* ec0=(n2a_ec[m]>=0)? d_echild+(size_t)n2a_ec[m]*ecStride:nullptr;
    const double* p0 =(n2a_slot[m]>=0)? d_partial+(size_t)n2a_slot[m]*slotSz:nullptr;
    const unsigned char* t0=(n2a_leaf[m]>=0)? d_tip+(size_t)n2a_leaf[m]*nptn:nullptr;
    const double* ec1=(n2b_ec[m]>=0)? d_echild+(size_t)n2b_ec[m]*ecStride:nullptr;
    const double* p1 =(n2b_slot[m]>=0)? d_partial+(size_t)n2b_slot[m]*slotSz:nullptr;
    const unsigned char* t1=(n2b_leaf[m]>=0)? d_tip+(size_t)n2b_leaf[m]*nptn:nullptr;
    double* o_base = out + (size_t)j*slotSz;
    for (int c=0;c<ncat;c++){                                    // == k1_node(isRoot=0,nchild=2) body
        double prod[NS_MAX]; for (int x=0;x<ns;x++) prod[x]=1.0;
        accum_child(prod,ns,c,ptn,nptn,ec0,p0,t0);
        accum_child(prod,ns,c,ptn,nptn,ec1,p1,t1);
        double* o = o_base + (size_t)(c*ns)*nptn + ptn;
        for (int r=0;r<ns;r++){ double v=0.0;
            for (int x=0;x<ns;x++) v += g_Uinv[r*ns+x]*prod[x];
            o[(size_t)r*nptn]=v; }
    }
}
__global__ void screen_node1_batch(int ns, int nptn, int ncat, int batchStart,
        double* __restrict__ out, size_t slotSz,
        const double* __restrict__ d_echild, size_t ecStride,
        const double* __restrict__ d_partial, const unsigned char* __restrict__ d_tip,
        const double* __restrict__ d_upper, const double* __restrict__ d_pmat,
        const int* __restrict__ mv_u, const int* __restrict__ mv_uIsRoot,
        const int* __restrict__ n1a_ec, const int* __restrict__ n1a_slot, const int* __restrict__ n1a_leaf,
        const int* __restrict__ n1b_ec, const int* __restrict__ n1b_slot, const int* __restrict__ n1b_leaf) {
    int j = blockIdx.y; int m = batchStart + j;
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    double* o_base = out + (size_t)j*slotSz;
    const double* ec0=(n1a_ec[m]>=0)? d_echild+(size_t)n1a_ec[m]*ecStride:nullptr;
    const double* p0 =(n1a_slot[m]>=0)? d_partial+(size_t)n1a_slot[m]*slotSz:nullptr;
    const unsigned char* t0=(n1a_leaf[m]>=0)? d_tip+(size_t)n1a_leaf[m]*nptn:nullptr;
    if (mv_uIsRoot[m]) {                                         // u==root: == k1_node(isRoot=0,nchild=2) body
        const double* ec1=(n1b_ec[m]>=0)? d_echild+(size_t)n1b_ec[m]*ecStride:nullptr;
        const double* p1 =(n1b_slot[m]>=0)? d_partial+(size_t)n1b_slot[m]*slotSz:nullptr;
        const unsigned char* t1=(n1b_leaf[m]>=0)? d_tip+(size_t)n1b_leaf[m]*nptn:nullptr;
        for (int c=0;c<ncat;c++){
            double prod[NS_MAX]; for (int x=0;x<ns;x++) prod[x]=1.0;
            accum_child(prod,ns,c,ptn,nptn,ec0,p0,t0);
            accum_child(prod,ns,c,ptn,nptn,ec1,p1,t1);
            double* o = o_base + (size_t)(c*ns)*nptn + ptn;
            for (int r=0;r<ns;r++){ double v=0.0;
                for (int x=0;x<ns;x++) v += g_Uinv[r*ns+x]*prod[x];
                o[(size_t)r*nptn]=v; }
        }
    } else {                                                     // interior u: == kj_pre_node(eigOut=1,nsib=1) body
        int u = mv_u[m];
        const double* up_uu = d_upper + (size_t)u*slotSz;
        const double* Pmat_u = d_pmat + (size_t)u*ecStride;
        for (int c=0;c<ncat;c++){
            double fsib[NS_MAX]; for (int x=0;x<ns;x++) fsib[x]=1.0;
            accum_child(fsib,ns,c,ptn,nptn,ec0,p0,t0);
            const double* uuc = up_uu + (size_t)(c*ns)*nptn + ptn;
            const double* Pc  = Pmat_u + (size_t)c*ns*ns;
            double rr[NS_MAX];
            for (int x=0;x<ns;x++){ double v=0.0;
                for (int t=0;t<ns;t++) v += Pc[x*ns+t]*uuc[(size_t)t*nptn];
                rr[x] = v*fsib[x]; }
            double* o = o_base + (size_t)(c*ns)*nptn + ptn;
            for (int jj=0;jj<ns;jj++){ double v=0.0;
                for (int x=0;x<ns;x++) v += g_Uinv[jj*ns+x]*rr[x];
                o[(size_t)jj*nptn]=v; }
        }
    }
}
__global__ void screen_k2_batch(int ns, int nptn, int ncat, int batchStart,
        const double* __restrict__ n1_all, const double* __restrict__ n2_all, size_t slotSz,
        const double* __restrict__ d_valall, double* __restrict__ patlh_all,
        int useInv, double pinv, const double* __restrict__ baseinvar) {
    int j = blockIdx.y; int m = batchStart + j;
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    const double* node_eig = n1_all + (size_t)j*slotSz;          // == k2_derv_mix(R=ncat) body (lh channel only)
    const double* dad_eig  = n2_all + (size_t)j*slotSz;
    const double* dval0 = d_valall + (size_t)m*3*ncat*ns;
    const double* dval1 = dval0 + ncat*ns; const double* dval2 = dval0 + 2*ncat*ns;
    double lh=0.0, d1=0.0, d2=0.0;
    for (int r=0;r<ncat;r++) for (int x=0;x<ns;x++){
        size_t o=(size_t)(r*ns+x)*nptn+ptn;
        double th=node_eig[o]*dad_eig[o]; int k=r*ns+x;
        lh+=dval0[k]*th; d1+=dval1[k]*th; d2+=dval2[k]*th;       // d1/d2 mirror the serial loop (dead here) so lh's order is identical
    }
    (void)d1; (void)d2;
    double* patlh = patlh_all + (size_t)j*nptn;
    if (useInv) { double Lp=fabs(lh)+pinv*baseinvar[ptn]; patlh[ptn]=log(Lp); }      // == k2_derv_mix_inv
    else        { patlh[ptn]=log(fabs(lh)); }                                        // == k2_derv_mix
}

// G.8.1b — LEAF endpoint eigen partial for mixtures (per-class). Regime r=m*ncat+c: L[r][i] = Uinv_m[i][s]
// (column s of class m's Uinv; UinvRowSum_m[i] if the state is ambiguous), replicated over the ncat cats within
// class m. Reads the GLOBAL per-class d_Uinv/d_UinvRowSum (NOT the single-model __constant__ g_Uinv).
__global__ void k_leaf_eig_mix(int ns, int nptn, int ncat, int nmix,
        const unsigned char* __restrict__ tipt,
        const double* __restrict__ d_Uinv, const double* __restrict__ d_UinvRowSum,
        double* __restrict__ out){
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    int s = tipt[ptn];
    int R = nmix*ncat;
    for (int r=0;r<R;r++){ int m=r/ncat;
        const double* Uinv_m = d_Uinv + (size_t)m*ns*ns;
        const double* Urs_m  = d_UinvRowSum + (size_t)m*ns;
        for (int i=0;i<ns;i++){
            double Li = (s<ns) ? Uinv_m[i*ns+s] : Urs_m[i];
            out[(size_t)(r*ns+i)*nptn+ptn] = Li;
        }
    }
}

// G.8.2.1a — Ji-2020 top-down PREORDER eigen-space partial pre_v for PROFILE MIXTURES (kj_pre generalised to the
// regime axis r = m*ncat + c). pre_v = "rest of tree above edge (parent u)->v", WITHOUT v's own branch b_v (the
// per-edge derivative reapplies b_v once, via k2_derv_mix's dval). The PARENT branch b_u is applied here through
// expfac_u[r][i] = exp(eval_m[i]·catRate_c·b_u). Within one regime r the class m is constant root-to-tip, so every
// per-class array (U_m up-map / Uinv_m down-map / Urs_m for ambiguous leaf siblings) is indexed by m=r/ncat, and
// the per-category rate is c=r%ncat — exactly as k1_node_mix. π_m is ABSORBED in the eigen-space partials (theta
// trick), so this kernel must NOT re-apply d_freq[m]. Output layout matches k1_node_mix's non-root write
// (out_pre[r*ns*nptn + j*nptn + ptn]) so k2_derv_mix reads pl_v (lower) and pre_v (upper) endpoints uniformly.
//   pus[t]      = Σ_i U_m[t][i]·expfac_u[r][i]·pre_u[r][i]      (map UP with eigenvectors U_m)
//   fsib[r][t]  = Π_{siblings of v} ( Σ_i echild_sib[r][t][i]·pl_sib[r][i] )    (accum_child_mix, per class m)
//   pre_v[r][j] = Σ_t Uinv_m[j][t]·pus[t]·fsib[t]               (map back DOWN with Uinv_m)
__global__ void k7_pre_mix(int ns, int nptn, int ncat, int nmix,
        double* __restrict__ out_pre,
        const double* __restrict__ pre_u,          // [R*ns*nptn]  parent's upper partial
        const double* __restrict__ expfac_u,       // [R*ns]       exp(eval_m[i]*catRate_c*b_u)
        const double* __restrict__ d_U,            // [nmix][ns*ns] eigenvectors (up-map)
        const double* __restrict__ d_Uinv,         // [nmix][ns*ns] inverse eigenvectors (down-map)
        const double* __restrict__ d_UinvRowSum,   // [nmix][ns]    leaf-sibling ambiguous
        int nsib,
        const double* ec0, const double* sp0, const unsigned char* st0,
        const double* ec1, const double* sp1, const unsigned char* st1) {
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    int R = nmix*ncat;
    // G.8.2.3 — 2D-grid regime parallelisation (gridDim.y==R => one regime per blockIdx.y; saturates small nptn).
    // pre_v[r] is computed independently per regime (no cross-regime accumulation) so the 2D split is race-free.
    int r0 = (gridDim.y>1) ? blockIdx.y : 0;
    int rN = (gridDim.y>1) ? (blockIdx.y+1) : R;
    for (int r=r0;r<rN;r++){ int m=r/ncat;
        const double* U_m    = d_U          + (size_t)m*ns*ns;
        const double* Uinv_m = d_Uinv       + (size_t)m*ns*ns;
        const double* Urs_m  = d_UinvRowSum + (size_t)m*ns;
        double fsib[NS_MAX]; for (int t=0;t<ns;t++) fsib[t]=1.0;
        accum_child_mix(fsib,ns,r,ptn,nptn,ec0,sp0,st0,Uinv_m,Urs_m);
        if (nsib>1) accum_child_mix(fsib,ns,r,ptn,nptn,ec1,sp1,st1,Uinv_m,Urs_m);
        const double* puc = pre_u + (size_t)r*ns*nptn + ptn;
        const double* ef  = expfac_u + (size_t)r*ns;
        double pus[NS_MAX];
        for (int t=0;t<ns;t++){ double v=0.0; for (int i=0;i<ns;i++) v += U_m[t*ns+i]*ef[i]*puc[(size_t)i*nptn]; pus[t]=v; }
        double* o = out_pre + (size_t)r*ns*nptn + ptn;
        for (int j=0;j<ns;j++){ double v=0.0; for (int t=0;t<ns;t++) v += Uinv_m[j*ns+t]*pus[t]*fsib[t]; o[(size_t)j*nptn]=v; }
    }
}

// =============================== G.4.2 JOLT kernels (ported from gpu_k8b_jolt_alpha.cu, runtime-ns) ===============================
// kj_theta — t-independent edge product theta[c*ns+x] = node_eig (.) dad_eig, materialised so kj_derv AND
// kj_ratenum can both read it (k2_derv above fuses the product; JOLT needs theta cached for the rate gradient).
__global__ void kj_theta(int ns, int nptn, int blockc,
        const double* __restrict__ node, const double* __restrict__ dad, double* __restrict__ theta){
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    for (int k=0;k<blockc;k++){ size_t o=(size_t)k*nptn+ptn; theta[o]=node[o]*dad[o]; }
}
// kj_derv — per-pattern lnL/df/ddf from cached theta and the t-dependent g_val0/1/2 (= k2_derv from theta).
// G.4.3b +I: the total pattern likelihood is L_p = (1-pinv)*Vbar_p + pinv*base_invar[p]. Because g_val0 folds in
// catProp = (1-pinv)/K (RateGammaInvar::getProp), the theta-reduction lh already equals (1-pinv)*Vbar_p, so the
// total is simply Lp = lh + pinv*baseinvar[ptn]. The invariant term is branch/alpha-INDEPENDENT, so d1=dLp/dt and
// d2=d2Lp/dt2 are unchanged — only the DENOMINATOR becomes Lp (=> df/ddf are the +I-correct log-derivatives).
// pinv=0 => Lp==lh, byte-identical to the pre-+I behaviour (baseinvar may be any valid buffer).
__global__ void kj_derv(int ns, int nptn, int ncat, const double* __restrict__ theta,
        double pinv, const double* __restrict__ baseinvar,
        double* __restrict__ patlh, double* __restrict__ pdf, double* __restrict__ pddf){
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    double lh=0.0,d1=0.0,d2=0.0;
    for (int c=0;c<ncat;c++) for (int x=0;x<ns;x++){
        double th=theta[(size_t)(c*ns+x)*nptn+ptn]; int k=c*ns+x;
        lh+=g_val0[k]*th; d1+=g_val1[k]*th; d2+=g_val2[k]*th; }
    double Lp=fabs(lh)+pinv*baseinvar[ptn]; double inv=1.0/Lp, r=d1*inv;
    patlh[ptn]=log(Lp); pdf[ptn]=r; pddf[ptn]=d2*inv-r*r;
}
// part8 #3 — kj_derv_fused: FUSE kj_theta + kj_derv + kj_ratenum. Compute theta = node*dad in REGISTERS (never
// materialised to the 601 MB d_theta), then emit patlh/pdf/pddf AND (if rnum!=null) accumulate the per-category
// rate-gradient numerator (rnum[c] += g_rscale[c]*Σ_x g_val1[c,x]*theta) — one pass, node+dad read once each, the
// theta VRAM round-trip (1 write + 2 reads = 3x slotSz/edge) ELIMINATED. On the bandwidth-bound kernel this is the
// win. BIT-IDENTICAL to the unfused path: FP64 store/load is lossless, and the per-(c,x) products are evaluated in
// the same order. rnum==null => derv-only (the evalLnL path, no rate gradient). d1 = Σ_c rc reuses the per-cat sum.
__global__ void kj_derv_fused(int ns, int nptn, int ncat,
        const double* __restrict__ node, const double* __restrict__ dad,
        double pinv, const double* __restrict__ baseinvar,
        double* __restrict__ patlh, double* __restrict__ pdf, double* __restrict__ pddf,
        double* __restrict__ rnum, double* __restrict__ wnum){   // G.5.1: wnum[c]=Lc(p) per-category likelihood (weight-grad numerator)
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    double lh=0.0,d1=0.0,d2=0.0;
    for (int c=0;c<ncat;c++){
        double rc=0.0, lcc=0.0;
        for (int x=0;x<ns;x++){ int k=c*ns+x; size_t o=(size_t)k*nptn+ptn;
            double th=node[o]*dad[o];
            lcc+=g_val0[k]*th; rc+=g_val1[k]*th; d2+=g_val2[k]*th; }
        lh+=lcc; d1+=rc;
        if (rnum) rnum[(size_t)c*nptn+ptn]+=g_rscale[c]*rc;
        if (wnum) wnum[(size_t)c*nptn+ptn]=lcc;   // G.5.1: Lc(p)=Σ_x g_val0[c,x]·θ (category weight w_c already folded into g_val0)
    }
    double Lp=fabs(lh)+pinv*baseinvar[ptn]; double inv=1.0/Lp, r=d1*inv;
    patlh[ptn]=log(Lp); pdf[ptn]=r; pddf[ptn]=d2*inv-r*r;
}
// Inc 2 (async/streams ladder): kj_derv_fused_args — a CHARACTER-FOR-CHARACTER twin of kj_derv_fused whose four
// per-edge coefficient tables (val0/val1/val2/rscale) arrive as KERNEL ARGS (dval0/dval1/dval2/drscale) instead of
// the __constant__ g_val0/g_val1/g_val2/g_rscale. SAME loop nesting, SAME (c,x) product order, SAME lh/d1/d2 +=
// accumulation, SAME rnum+= / wnum= stores => BIT-IDENTICAL output to kj_derv_fused given the SAME table bytes. The 4
// pointer args are LAST so existing 12-arg kj_derv_fused<<<>>>(...) call sites are untouched. Used ONLY on the gated
// (JOLT_TS_ASYNC=1) reopt path; the per-edge coefficient block is staged into a private valpool slot and copied
// async, eliminating the per-edge cudaMemcpyToSymbol storm (constant memory must be set on the default stream).
__global__ void kj_derv_fused_args(int ns,int nptn,int ncat,
        const double* __restrict__ node,const double* __restrict__ dad,
        double pinv,const double* __restrict__ baseinvar,
        double* __restrict__ patlh,double* __restrict__ pdf,double* __restrict__ pddf,
        double* __restrict__ rnum,double* __restrict__ wnum,
        const double* __restrict__ dval0,const double* __restrict__ dval1,
        const double* __restrict__ dval2,const double* __restrict__ drscale){
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    double lh=0.0,d1=0.0,d2=0.0;
    for (int c=0;c<ncat;c++){
        double rc=0.0, lcc=0.0;
        for (int x=0;x<ns;x++){ int k=c*ns+x; size_t o=(size_t)k*nptn+ptn;
            double th=node[o]*dad[o];
            lcc+=dval0[k]*th; rc+=dval1[k]*th; d2+=dval2[k]*th; }
        lh+=lcc; d1+=rc;
        if (rnum) rnum[(size_t)c*nptn+ptn]+=drscale[c]*rc;
        if (wnum) wnum[(size_t)c*nptn+ptn]=lcc;   // G.5.1: Lc(p)=Σ_x dval0[c,x]·θ (category weight w_c already folded in)
    }
    double Lp=fabs(lh)+pinv*baseinvar[ptn]; double inv=1.0/Lp, r=d1*inv;
    patlh[ptn]=log(Lp); pdf[ptn]=r; pddf[ptn]=d2*inv-r*r;
}
// kj_pre — Ji-2020 top-down PREORDER eigen-space partial pre_v ("rest of tree" above edge u->v), WITHOUT v's
// own branch (the gradient's g_val0/1(b_v) reapply it once). The PARENT branch b_u (expfac_u) is applied here.
//   pus[t]  = Sum_i U[t][i]*expfac_u[i]*pre_u[c][i]
//   fsib[t] = Prod_siblings ( Sum_i echild_sib[t][i]*pl_sib[i] )         (via accum_child)
//   pre_v[c][j] = Sum_t Uinv[j][t]*pus[t]*fsib[t]
__global__ void kj_pre(int ns, int nptn, int ncat, double* __restrict__ out_pre,
        const double* __restrict__ pre_u, const double* __restrict__ expfac_u,
        int nsib, const double* ec0, const double* sp0, const unsigned char* st0,
                 const double* ec1, const double* sp1, const unsigned char* st1){
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    for (int c=0;c<ncat;c++){
        double fsib[NS_MAX]; for (int t=0;t<ns;t++) fsib[t]=1.0;
        accum_child(fsib,ns,c,ptn,nptn,ec0,sp0,st0);
        if (nsib>1) accum_child(fsib,ns,c,ptn,nptn,ec1,sp1,st1);
        const double* puc=pre_u+(size_t)(c*ns)*nptn+ptn; const double* ef=expfac_u+(size_t)c*ns;
        double pus[NS_MAX];
        for (int t=0;t<ns;t++){ double v=0.0; for (int i=0;i<ns;i++) v+=g_U[t*ns+i]*ef[i]*puc[(size_t)i*nptn]; pus[t]=v; }
        double* o=out_pre+(size_t)(c*ns)*nptn+ptn;
        for (int j=0;j<ns;j++){ double v=0.0; for (int t=0;t<ns;t++) v+=g_Uinv[j*ns+t]*pus[t]*fsib[t]; o[(size_t)j*nptn]=v; }
    }
}

// ============================================================================================================
// TS.2 persistent-upper PRECISION FIX (F39/F40) — NODE-SPACE upper partials. The eigen-stored preorder upper
// (kj_pre above) round-trips through U·diag(expfac)·Uinv, whose Uinv·prod coefficient extraction is catastrophically
// ill-conditioned (sign-indefinite Uinv) on a near-equilibrium partial (long ancestor branch) — the source of the
// up-to-1.9% intermediate-tree error. The cure (matching the LOWER/FOLD path + BEAGLE/Ji-2020/PhyloBayes): keep the
// upper as a NODE-space POSITIVE-PRODUCT vector and apply P(b) as a single well-conditioned node-space matvec — no
// eigen round-trip, no cancellation, exact at any branch length. Two kernels:
//   k1_node_prod : the ROOT-CHILD seed = Π_{root's other children}(P·L) in node space (k1_node WITHOUT the final Uinv).
//   kj_pre_node  : the interior recurrence  up_v[x] = (P(b_u)·up_u)[x] · fsib[x]  (push through b_u FIRST, then ⊙ the
//                  node-space sibling product). eigOut=1 applies a SINGLE Uinv (node→eigen) for a move endpoint fed to
//                  k2_derv; eigOut=0 stores node-space for the persistent buffer. P(b_u) is the node-space transition
//                  Pmat_u = U·diag(exp(eval·rate·b_u))·Uinv = echild_u·Uinv, supplied by the host (pattern-independent).
// ============================================================================================================
__global__ void k1_node_prod(int ns, int nptn, int ncat, double* __restrict__ out, int nchild,
        const double* ec0, const double* p0, const unsigned char* t0,
        const double* ec1, const double* p1, const unsigned char* t1,
        const double* ec2, const double* p2, const unsigned char* t2) {
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    for (int c=0;c<ncat;c++){
        double prod[NS_MAX];
        for (int x=0;x<ns;x++) prod[x]=1.0;
        accum_child(prod,ns,c,ptn,nptn,ec0,p0,t0);
        if (nchild>1) accum_child(prod,ns,c,ptn,nptn,ec1,p1,t1);
        if (nchild>2) accum_child(prod,ns,c,ptn,nptn,ec2,p2,t2);
        double* o = out + (size_t)(c*ns)*nptn + ptn;
        for (int x=0;x<ns;x++) o[(size_t)x*nptn]=prod[x];   // NODE-space write (no final Uinv)
    }
}

__global__ void kj_pre_node(int ns, int nptn, int ncat, int eigOut, double* __restrict__ out,
        const double* __restrict__ up_u,        // node-space upper at u: [c*ns + t]*nptn
        const double* __restrict__ Pmat_u,      // node-space P(b_u): [c*ns*ns + x*ns + t]
        int nsib,
        const double* ec0, const double* sp0, const unsigned char* st0,
        const double* ec1, const double* sp1, const unsigned char* st1){
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    for (int c=0;c<ncat;c++){
        double fsib[NS_MAX]; for (int x=0;x<ns;x++) fsib[x]=1.0;     // node-space sibling product
        accum_child(fsib,ns,c,ptn,nptn,ec0,sp0,st0);
        if (nsib>1) accum_child(fsib,ns,c,ptn,nptn,ec1,sp1,st1);
        const double* uuc = up_u + (size_t)(c*ns)*nptn + ptn;        // u's node-space upper, this cat
        const double* Pc  = Pmat_u + (size_t)c*ns*ns;
        double r[NS_MAX];
        for (int x=0;x<ns;x++){ double v=0.0;                        // r[x] = (P(b_u)·up_u)[x] · fsib[x]
            for (int t=0;t<ns;t++) v += Pc[x*ns+t]*uuc[(size_t)t*nptn];
            r[x] = v*fsib[x]; }
        double* o = out + (size_t)(c*ns)*nptn + ptn;
        if (eigOut){                                                 // move endpoint: SINGLE Uinv -> eigen for k2_derv
            for (int j=0;j<ns;j++){ double v=0.0;
                for (int x=0;x<ns;x++) v += g_Uinv[j*ns+x]*r[x];
                o[(size_t)j*nptn]=v; }
        } else {                                                     // persistent buffer: store node-space
            for (int x=0;x<ns;x++) o[(size_t)x*nptn]=r[x];
        }
    }
}

// ============================================================================================================
// TS.OCC.1 (2026-07-04; DEFAULT-ON 2026-07-05) — compile-time state-count (NS) specialization of the 3 hot partial-lh
// kernels. VALIDATED bit-identical (RF=0, lnL identical) at every scale/state; decisive AA-1M no-boot = 2.314x (job
// 173024944). Now DEFAULT-ON; set JOLT_NS_TEMPLATE=0 to force the runtime-ns kernels (byte-identical escape hatch).
// WHY: with runtime `ns`, prod[NS_MAX]/fsib[NS_MAX]/pus[NS_MAX] cannot be register-promoted -> they spill to LOCAL
// memory (k1_node 160B, kj_pre 320B stack frame; confirmed `nvcc -Xptxas -v`). Templating NS in {4,20} + #pragma
// unroll de-spills them (Phase-0 probe: NS=4 -> 48reg/0spill, occupancy unchanged; NS=20 -> 128reg/0spill, occupancy
// 34%->25%). The state-loop ORDER is preserved (unroll only, no reassociation) so pattern_lh stays FP64 BIT-IDENTICAL
// (rel<=5e-16, same-device). `ns` is gated to {4,20} on the GPU (phylotreegpu.cpp); this dispatch falls back to the
// runtime kernel for any other ns, preserving the generic contract. Plan: GPU-TREESEARCH-OCCUPANCY-ATTACK-PLAN.md.
// ============================================================================================================
static inline bool jolt_ns_template_enabled(){
    static int c=-1;
    if (c<0){ const char* e=getenv("JOLT_NS_TEMPLATE"); c=(e && atoi(e)==0)?0:1; }  // DEFAULT-ON; =0 forces OFF
    return c!=0;
}

template<int NS>
__device__ __forceinline__ void accum_child_t(double* prod, int c, int ptn, int nptn,
        const double* __restrict__ ec, const double* __restrict__ p, const unsigned char* __restrict__ t) {
    const double* ecc = ec + (size_t)c*NS*NS;
    if (p) {
        const double* pc = p + (size_t)(c*NS)*nptn + ptn;
        #pragma unroll
        for (int x=0;x<NS;x++){ double v=0.0;
            #pragma unroll
            for (int i=0;i<NS;i++) v += ecc[x*NS+i]*pc[(size_t)i*nptn];
            prod[x]*=v; }
    } else {
        int s = t[ptn];
        #pragma unroll
        for (int x=0;x<NS;x++){ double v=0.0;
            #pragma unroll
            for (int i=0;i<NS;i++){ double Li = (s<NS)? g_Uinv[i*NS+s] : g_UinvRowSum[i]; v += ecc[x*NS+i]*Li; }
            prod[x]*=v; }
    }
}

template<int NS>
__global__ void k1_node_t(int nptn, int ncat, int isRoot, double* __restrict__ out, double* __restrict__ patlh,
        int nchild,
        const double* ec0, const double* p0, const unsigned char* t0,
        const double* ec1, const double* p1, const unsigned char* t1,
        const double* ec2, const double* p2, const unsigned char* t2,
        double pinv = 0.0, const double* base_invar = nullptr) {   // A3 (+I): add pinv*base_invar at root when base_invar!=null (else byte-identical)
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    double lh = 0.0;
    for (int c=0;c<ncat;c++){
        double prod[NS];
        #pragma unroll
        for (int x=0;x<NS;x++) prod[x]=1.0;
        accum_child_t<NS>(prod,c,ptn,nptn,ec0,p0,t0);
        if (nchild>1) accum_child_t<NS>(prod,c,ptn,nptn,ec1,p1,t1);
        if (nchild>2) accum_child_t<NS>(prod,c,ptn,nptn,ec2,p2,t2);
        if (isRoot){
            double s=0.0;
            #pragma unroll
            for (int x=0;x<NS;x++) s += g_freq[x]*prod[x];
            lh += g_catw[c]*s;
        } else {
            double* o = out + (size_t)(c*NS)*nptn + ptn;
            #pragma unroll
            for (int r=0;r<NS;r++){ double v=0.0;
                #pragma unroll
                for (int x=0;x<NS;x++) v += g_Uinv[r*NS+x]*prod[x];
                o[(size_t)r*nptn]=v; }
        }
    }
    if (isRoot) patlh[ptn] = log(base_invar ? fabs(lh)+pinv*base_invar[ptn] : fabs(lh));   // A3 (+I): +pinv*base_invar
}

template<int NS>
__global__ void kj_pre_t(int nptn, int ncat, double* __restrict__ out_pre,
        const double* __restrict__ pre_u, const double* __restrict__ expfac_u,
        int nsib, const double* ec0, const double* sp0, const unsigned char* st0,
                 const double* ec1, const double* sp1, const unsigned char* st1){
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    for (int c=0;c<ncat;c++){
        double fsib[NS];
        #pragma unroll
        for (int t=0;t<NS;t++) fsib[t]=1.0;
        accum_child_t<NS>(fsib,c,ptn,nptn,ec0,sp0,st0);
        if (nsib>1) accum_child_t<NS>(fsib,c,ptn,nptn,ec1,sp1,st1);
        const double* puc=pre_u+(size_t)(c*NS)*nptn+ptn; const double* ef=expfac_u+(size_t)c*NS;
        double pus[NS];
        #pragma unroll
        for (int t=0;t<NS;t++){ double v=0.0;
            #pragma unroll
            for (int i=0;i<NS;i++) v+=g_U[t*NS+i]*ef[i]*puc[(size_t)i*nptn]; pus[t]=v; }
        double* o=out_pre+(size_t)(c*NS)*nptn+ptn;
        #pragma unroll
        for (int j=0;j<NS;j++){ double v=0.0;
            #pragma unroll
            for (int t=0;t<NS;t++) v+=g_Uinv[j*NS+t]*pus[t]*fsib[t]; o[(size_t)j*nptn]=v; }
    }
}

template<int NS>
__global__ void kj_pre_node_t(int nptn, int ncat, int eigOut, double* __restrict__ out,
        const double* __restrict__ up_u,
        const double* __restrict__ Pmat_u,
        int nsib,
        const double* ec0, const double* sp0, const unsigned char* st0,
        const double* ec1, const double* sp1, const unsigned char* st1){
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    for (int c=0;c<ncat;c++){
        double fsib[NS];
        #pragma unroll
        for (int x=0;x<NS;x++) fsib[x]=1.0;
        accum_child_t<NS>(fsib,c,ptn,nptn,ec0,sp0,st0);
        if (nsib>1) accum_child_t<NS>(fsib,c,ptn,nptn,ec1,sp1,st1);
        const double* uuc = up_u + (size_t)(c*NS)*nptn + ptn;
        const double* Pc  = Pmat_u + (size_t)c*NS*NS;
        double r[NS];
        #pragma unroll
        for (int x=0;x<NS;x++){ double v=0.0;
            #pragma unroll
            for (int t=0;t<NS;t++) v += Pc[x*NS+t]*uuc[(size_t)t*nptn];
            r[x] = v*fsib[x]; }
        double* o = out + (size_t)(c*NS)*nptn + ptn;
        if (eigOut){
            #pragma unroll
            for (int j=0;j<NS;j++){ double v=0.0;
                #pragma unroll
                for (int x=0;x<NS;x++) v += g_Uinv[j*NS+x]*r[x];
                o[(size_t)j*nptn]=v; }
        } else {
            #pragma unroll
            for (int x=0;x<NS;x++) o[(size_t)x*nptn]=r[x];
        }
    }
}

// Launch dispatchers — JOLT_NS_TEMPLATE on + ns in {4,20} -> de-spilled template<NS>; else runtime kernel (byte-id).
// Signature order mirrors the raw <<<GB,TB,0,stream>>> launch: (GB,TB,stream, then the kernel's own arg list incl. ns).
static inline void launch_k1_node(int GB, int TB, cudaStream_t stream,
        int ns, int nptn, int ncat, int isRoot, double* out, double* patlh, int nchild,
        const double* ec0, const double* p0, const unsigned char* t0,
        const double* ec1, const double* p1, const unsigned char* t1,
        const double* ec2, const double* p2, const unsigned char* t2,
        double pinv = 0.0, const double* base_invar = nullptr){   // A3 (+I): threaded to the kernel's root term; defaults keep every existing caller byte-identical
    const bool T = jolt_ns_template_enabled();
    if (T && ns==4)  k1_node_t<4> <<<GB,TB,0,stream>>>(nptn,ncat,isRoot,out,patlh,nchild,ec0,p0,t0,ec1,p1,t1,ec2,p2,t2,pinv,base_invar);
    else if (T && ns==20) k1_node_t<20><<<GB,TB,0,stream>>>(nptn,ncat,isRoot,out,patlh,nchild,ec0,p0,t0,ec1,p1,t1,ec2,p2,t2,pinv,base_invar);
    else k1_node<<<GB,TB,0,stream>>>(ns,nptn,ncat,isRoot,out,patlh,nchild,ec0,p0,t0,ec1,p1,t1,ec2,p2,t2,pinv,base_invar);
}
static inline void launch_kj_pre(int GB, int TB, cudaStream_t stream,
        int ns, int nptn, int ncat, double* out_pre,
        const double* pre_u, const double* expfac_u,
        int nsib, const double* ec0, const double* sp0, const unsigned char* st0,
        const double* ec1, const double* sp1, const unsigned char* st1){
    const bool T = jolt_ns_template_enabled();
    if (T && ns==4)  kj_pre_t<4> <<<GB,TB,0,stream>>>(nptn,ncat,out_pre,pre_u,expfac_u,nsib,ec0,sp0,st0,ec1,sp1,st1);
    else if (T && ns==20) kj_pre_t<20><<<GB,TB,0,stream>>>(nptn,ncat,out_pre,pre_u,expfac_u,nsib,ec0,sp0,st0,ec1,sp1,st1);
    else kj_pre<<<GB,TB,0,stream>>>(ns,nptn,ncat,out_pre,pre_u,expfac_u,nsib,ec0,sp0,st0,ec1,sp1,st1);
}
static inline void launch_kj_pre_node(int GB, int TB, cudaStream_t stream,
        int ns, int nptn, int ncat, int eigOut, double* out,
        const double* up_u, const double* Pmat_u,
        int nsib, const double* ec0, const double* sp0, const unsigned char* st0,
        const double* ec1, const double* sp1, const unsigned char* st1){
    const bool T = jolt_ns_template_enabled();
    if (T && ns==4)  kj_pre_node_t<4> <<<GB,TB,0,stream>>>(nptn,ncat,eigOut,out,up_u,Pmat_u,nsib,ec0,sp0,st0,ec1,sp1,st1);
    else if (T && ns==20) kj_pre_node_t<20><<<GB,TB,0,stream>>>(nptn,ncat,eigOut,out,up_u,Pmat_u,nsib,ec0,sp0,st0,ec1,sp1,st1);
    else kj_pre_node<<<GB,TB,0,stream>>>(ns,nptn,ncat,eigOut,out,up_u,Pmat_u,nsib,ec0,sp0,st0,ec1,sp1,st1);
}

// make_pmat — derive the node-space transition Pmat[v][c][x][t] = Σ_i echild[v][c][x][i]·Uinv[i][t] = P(b_v)[x][t]
// from the already-uploaded echild (= U·diag(exp(eval·rate·b_v))) and g_Uinv. Pattern-INDEPENDENT (no ptn axis),
// run ONCE per launcher; same per-node layout/stride as echild. One thread per (v,c,x) row, writing ns entries (t).
__global__ void make_pmat(int ns, int ncat, int nnodes, const double* __restrict__ echild, double* __restrict__ Pmat){
    int row = blockIdx.x*blockDim.x + threadIdx.x;   // row = (v*ncat+c)*ns + x  over nnodes*ncat*ns rows
    if (row >= nnodes*ncat*ns) return;
    const double* ec = echild + (size_t)row*ns;      // echild[v][c][x][:]  (over i)
    double* pm = Pmat + (size_t)row*ns;              // Pmat[v][c][x][:]    (over t)
    for (int t=0;t<ns;t++){ double v=0.0;
        for (int i=0;i<ns;i++) v += ec[i]*g_Uinv[i*ns+t];   // Σ_i echild[x][i]·Uinv[i][t] = P(b)[x][t]
        pm[t]=v; }
}
// kj_ratenum — accumulate b_e*qp_e[k] into rnum[k][ptn] per category (the +R/alpha rate-grad numerator).
// g_rscale[k]=b_e/(r_k*w_k) folds the chain rule; Sum_x g_val1[k,x]*theta = r_k*w_k*qp_e[k].
__global__ void kj_ratenum(int ns, int nptn, int ncat, const double* __restrict__ theta, double* __restrict__ rnum){
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    for (int k=0;k<ncat;k++){ double s=0.0;
        for (int x=0;x<ns;x++) s+=g_val1[k*ns+x]*theta[(size_t)(k*ns+x)*nptn+ptn];
        rnum[(size_t)k*nptn+ptn]+=g_rscale[k]*s; }
}
// G.5.0 — kj_reduce3: deterministic block-level FP64 reduction of the 3 per-pattern derivative channels
// {patlh,pdf,pddf}, each weighted by ptn_freq, replacing reduceDerv's 3x nptn D2H + single-threaded host Kahan
// loop (called ~197x/gradient-sweep + 1x/evalLnL — the measured #1 host-reduction stall, part8 VIII.2 #1).
// Each block tree-reduces its blockDim patterns into 3 partial sums out3[ch*nblk + blockIdx]; the host then
// Kahan-combines the (small) nblk per-block partials in the SAME channel order as the old loop. The within-block
// reduce is a fixed-blockDim shared-memory tree (NOT atomicAdd) => bit-reproducible across launches, so the LM
// accept/reject trajectory is stable. Requires blockDim a power of two (TB=256). FP64 throughout (parity).
__global__ void kj_reduce3(int nptn, const double* __restrict__ patlh, const double* __restrict__ pdf,
        const double* __restrict__ pddf, const double* __restrict__ ptnfreq, int nblk, double* __restrict__ out3){
    extern __shared__ double sm[];                  // 3*blockDim doubles
    double* sL=sm; double* sD=sm+blockDim.x; double* sDD=sm+2*blockDim.x;
    int tid=threadIdx.x; int p=blockIdx.x*blockDim.x+tid;
    double f = (p<nptn) ? ptnfreq[p] : 0.0;
    sL[tid]  = (p<nptn) ? f*patlh[p] : 0.0;
    sD[tid]  = (p<nptn) ? f*pdf[p]   : 0.0;
    sDD[tid] = (p<nptn) ? f*pddf[p]  : 0.0;
    __syncthreads();
    for (int s=blockDim.x>>1; s>0; s>>=1){
        if (tid<s){ sL[tid]+=sL[tid+s]; sD[tid]+=sD[tid+s]; sDD[tid]+=sDD[tid+s]; }
        __syncthreads();
    }
    if (tid==0){ out3[blockIdx.x]=sL[0]; out3[(size_t)nblk+blockIdx.x]=sD[0]; out3[(size_t)2*nblk+blockIdx.x]=sDD[0]; }
}
// G.5.0 Part B — kj_invl: 1/L_p = exp(-patlh[p]) on-device (was a host exp() loop over nptn at the base edge).
__global__ void kj_invl(int nptn, const double* __restrict__ patlh, double* __restrict__ invl){
    int p=blockIdx.x*blockDim.x+threadIdx.x; if(p>=nptn) return; invl[p]=exp(-patlh[p]);
}
// G.5.0 Part B — kj_reduce_gradnum: per-category deterministic block reduction of ptn_freq[p]*rnum[c][p]*invl[p]
// (the +R/alpha rate-gradient numerator), replacing the ncat*nptn d_rnum D2H + host long-double loop. The +R ladder
// hammers this ncat-fold (ncat up to 10), so it must be on-device. out[c*nblk + blockIdx] = per-block partial;
// the host sums the (small) nblk partials per category, then scales by catProp_v[c]. Deterministic shared-mem tree
// reduce (no atomicAdd). The common factor ptn_freq*invl is loaded once/thread; ncat passes reuse one shared array.
__global__ void kj_reduce_gradnum(int nptn, int ncat, const double* __restrict__ rnum,
        const double* __restrict__ invl, const double* __restrict__ ptnfreq, int nblk, double* __restrict__ out){
    extern __shared__ double sm[];                  // blockDim doubles
    int tid=threadIdx.x; int p=blockIdx.x*blockDim.x+tid;
    double fw = (p<nptn) ? ptnfreq[p]*invl[p] : 0.0;
    for (int c=0;c<ncat;c++){
        sm[tid] = (p<nptn) ? fw*rnum[(size_t)c*nptn+p] : 0.0;
        __syncthreads();
        for (int s=blockDim.x>>1; s>0; s>>=1){ if (tid<s) sm[tid]+=sm[tid+s]; __syncthreads(); }
        if (tid==0) out[(size_t)c*nblk+blockIdx.x]=sm[0];
        __syncthreads();
    }
}

// ================= OPG (empirical-Fisher) Gram — Phase 1 (JOLT_OPG, default-OFF) =================
// H = Sum_p ptn_freq[p]*s_p*s_p^T over the 2K +R coords theta=(y_0..y_{K-1}, z_0..z_{K-1}), y_c=log(meanR_c),
// z_c = softmax weight-logit. Per-SITE scores (OPG-SOLVER-IMPLEMENTATION.md section 1; both reductions FD-validated
// by GATE-0.5 and the algebra re-derived by two independent audits):
//     s^y_{p,c} = catRate[c]*catProp_v[c]*invl[p]*rnum[c][p]
//     s^z_{p,c} = invl[p]*wnum[c][p] - bprop[c]*R_p,   R_p = Sum_k invl[p]*wnum[k][p]
// R_p is NOT 1 under +I (=1-pinv*I_p/L_p); using the pure-+R rN shortcut here would be wrong -- the #1 bug-hiding spot.
// TEMPLATED on K so the channel-loop (a,b) indices are COMPILE-TIME => s[2K] stays in REGISTERS. With a runtime ncat
// the loop is dynamically indexed => nvcc spills s[] to LOCAL memory => ~2*NCH*Pn*8 B of local traffic per chunk
// (~5-10% of the sweep, scaling as k^2) -- the real cost risk, not the barrier count. Warp-shuffle reduce gives ONE
// __syncthreads per block instead of NCH shared-tree passes. Deterministic: no atomics, fixed shuffle/warp order.
template<int K>
__launch_bounds__(256,2)
__global__ void kj_opg_gram(int Pn,                      // chunk width (buffers are Pn-strided; Pn shrinks on the last chunk)
        const double* __restrict__ rnum, const double* __restrict__ wnum,
        const double* __restrict__ invl, const double* __restrict__ ptnfreq,
        const double* __restrict__ cRcP,                 // [K] catRate[c]*catProp_v[c]
        const double* __restrict__ bpr,                  // [K] bprop[c]
        int nblk, double* __restrict__ out){             // out[NCH*nblk], nblk == GB of THIS chunk
    constexpr int N2  = 2*K;
    constexpr int NCH = K*(2*K+1);                       // == N2*(N2+1)/2 upper-triangle channels
    extern __shared__ double sm[];                       // [NCH * nWarp] doubles
    const int tid=threadIdx.x, lane=tid&31, warp=tid>>5, nWarp=blockDim.x>>5;
    const int p = blockIdx.x*blockDim.x + tid;
    double s[N2]; double f=0.0;
    if (p<Pn){
        f = ptnfreq[p]; const double il = invl[p];
        double Rp=0.0;
        #pragma unroll
        for(int c=0;c<K;c++) Rp += il*wnum[(size_t)c*Pn+p];
        #pragma unroll
        for(int c=0;c<K;c++){
            s[c]   = cRcP[c]*il*rnum[(size_t)c*Pn+p];
            s[K+c] = il*wnum[(size_t)c*Pn+p] - bpr[c]*Rp;
        }
    } else {
        #pragma unroll
        for(int c=0;c<N2;c++) s[c]=0.0;
    }
    int ch=0;
    #pragma unroll
    for(int a=0;a<N2;a++){
        #pragma unroll
        for(int b=a;b<N2;b++){
            double v = f*s[a]*s[b];
            #pragma unroll
            for(int off=16; off>0; off>>=1) v += __shfl_down_sync(0xffffffffu, v, off);
            if(lane==0) sm[ch*nWarp + warp] = v;
            ch++;
        }
    }
    __syncthreads();                                     // ONE barrier for the whole kernel
    for(int c2=tid; c2<NCH; c2+=blockDim.x){
        double acc=0.0;
        for(int w=0; w<nWarp; w++) acc += sm[(size_t)c2*nWarp + w];   // fixed warp order => deterministic
        out[(size_t)c2*nblk + blockIdx.x] = acc;
    }
}
// Stage 2: one block per channel; fixed grid-stride + shared tree over the nblk per-block partials. Drops the D2H
// from NCH*GB doubles (43.7 MB at avian nptn~20M) to NCH doubles (~1.7 KB) and removes a host loop over millions of
// partials. Deterministic (fixed stride order + fixed tree, no atomics).
__global__ void kj_opg_reduce2(int nblk, const double* __restrict__ in, double* __restrict__ out){
    extern __shared__ double sm[];                       // blockDim doubles
    const int ch=blockIdx.x, tid=threadIdx.x;
    double acc=0.0;
    for(int b=tid; b<nblk; b+=blockDim.x) acc += in[(size_t)ch*nblk + b];
    sm[tid]=acc; __syncthreads();
    for(int s=blockDim.x>>1; s>0; s>>=1){ if(tid<s) sm[tid]+=sm[tid+s]; __syncthreads(); }
    if(tid==0) out[ch]=sm[0];
}
// ncat -> template dispatch. ncat>10 is a HARD diagonal fallback (no OPG): at ncat=64 s[128] would be 1 KB/thread of
// local memory and NCH=8256 > TB=256 would silently break the epilogue. JOLT_FREERATE_HIGHK can push MAXCAT past 10.
static inline bool jolt_opg_gram_launch(int ncat,int GB,int TB,size_t shbytes,int Pn,
        const double* rnum,const double* wnum,const double* invl,const double* ptnfreq,
        const double* cRcP,const double* bpr,int nblk,double* out){
    switch(ncat){
#define JOPG_CASE(N) case N: kj_opg_gram<N><<<GB,TB,shbytes>>>(Pn,rnum,wnum,invl,ptnfreq,cRcP,bpr,nblk,out); return true;
        JOPG_CASE(2) JOPG_CASE(3) JOPG_CASE(4) JOPG_CASE(5) JOPG_CASE(6)
        JOPG_CASE(7) JOPG_CASE(8) JOPG_CASE(9) JOPG_CASE(10)
#undef JOPG_CASE
        default: return false;
    }
}
// Self-contained symmetric cyclic Jacobi (n<=32). Fixed sweep count + fixed (p,q) order => bit-reproducible; no malloc,
// no external symbols. Deliberately NOT whtest/eigen_sym.h (C-linkage NR tred2/tqli in a GPL module whose header
// #defines NUM_STATE 4 / ZERO / DVec20 would pollute this CUDA TU) and NOT utils/eigendecomposition.cpp (non-symmetric
// rate-matrix path). Overwrites a[]; returns eigenvalues (unsorted) in ev[].
static void jolt_jacobi_eig(double* a, int n, double* ev){
    for(int sweep=0; sweep<12; sweep++){
        double off=0.0;
        for(int p=0;p<n;p++) for(int q=p+1;q<n;q++) off += a[p*n+q]*a[p*n+q];
        if(off <= 1e-300) break;
        for(int p=0;p<n;p++) for(int q=p+1;q<n;q++){
            double apq=a[p*n+q]; if(fabs(apq)<1e-300) continue;
            double app=a[p*n+p], aqq=a[q*n+q];
            double th=(aqq-app)/(2.0*apq);
            double t=(th>=0.0? 1.0:-1.0)/(fabs(th)+sqrt(th*th+1.0));
            double c=1.0/sqrt(t*t+1.0), s=t*c;
            for(int k=0;k<n;k++){ double akp=a[k*n+p], akq=a[k*n+q];
                a[k*n+p]=c*akp-s*akq; a[k*n+q]=s*akp+c*akq; }
            for(int k=0;k<n;k++){ double apk=a[p*n+k], aqk=a[q*n+k];
                a[p*n+k]=c*apk-s*aqk; a[q*n+k]=s*apk+c*aqk; }
        }
    }
    for(int i=0;i<n;i++) ev[i]=a[i*n+i];
}

#define GCK(x) do{ cudaError_t _e=(x); if(_e!=cudaSuccess){ \
    fprintf(stderr,"[GPU-XCHECK] %s failed at %s:%d: %s\n",#x,__FILE__,__LINE__,cudaGetErrorString(_e)); \
    return (double)NAN; } }while(0)

// G.2.1b — persistent device-buffer pool. The seam calls these launchers thousands of times during branch
// optimisation; cudaMalloc/cudaFree of the multi-GB partial arena PER CALL would dominate wall time. The pool
// allocates each buffer once (grown on demand) and reuses it across calls. Contents are fully overwritten every
// call (fresh H2D + recomputed sweep) -> STATELESS correctness preserved (no device-resident state carries
// across calls, per the verified G.2.1 coherence contract); only the ALLOCATION persists. Single GPU, single
// thread (the GPU gate forces serial), so no concurrency. Never freed (released at process exit).
struct DevBuf { void* p = nullptr; size_t cap = 0; };
static bool devbuf_ensure(DevBuf& b, size_t need) {
    if (need <= b.cap && b.p) return true;
    if (b.p) { cudaFree(b.p); b.p = nullptr; b.cap = 0; }
    if (cudaMalloc(&b.p, need) != cudaSuccess) { b.p = nullptr; b.cap = 0; return false; }
    b.cap = need; return true;
}
static DevBuf gb_echild, gb_tip, gb_partial, gb_patlh, gb_pdf, gb_pddf, gb_nodeleaf, gb_dadleaf;
static DevBuf gb_n1eig, gb_n2eig;   // TS.2-I3a: re-pairing-fold scratch (node1/node2 swapped directed eigen partials)
static DevBuf gb_valall;            // TS.2.1 K1: per-move central-edge {v0,v1,v2} coeff tables (nMoves*3*ncat*ns), uploaded ONCE
static DevBuf gb_baseinvar;         // A3 (+I): screener per-pattern invariant base, chunk-sized, uploaded per chunk
static DevBuf gb_sptnfreq, gb_sredpart;   // TS.2.1 K1: screener on-device reduction — per-chunk device ptn_freq + 3*nblk block-partials (env TS_SCREEN_GPUREDUCE)
static DevBuf gb_uexpfac, gb_upper;  // TS.2-I3b: per-node expfac (parent branch) + PERSISTENT per-node upper-partial buffer
static DevBuf gb_pmat;               // TS.2 fix: per-node node-space transition P(b)=echild·Uinv (derived on-device, screeners)
static DevBuf gb_n1batch, gb_n2batch, gb_patlhbatch, gb_mvdesc;   // TS RAKE BATCH: B-wide per-move fold scratch (n1/n2 eigen + patlh) + packed [14][nMoves] move-descriptors
#define DEVB(b, bytes) do{ if(!devbuf_ensure((b),(size_t)(bytes))){ \
    fprintf(stderr,"[GPU] devbuf_ensure failed (%zu bytes) at %s:%d\n",(size_t)(bytes),__FILE__,__LINE__); \
    return (double)NAN; } }while(0)

// ============================================================================================================
// ASYNC/STREAMS substrate (Inc 0) — process-global stream pool + pinned host staging for screener move overlap.
// Master gate JOLT_TS_ASYNC (default OFF). When OFF, the screener uses S=1, the default stream (arg 0), and the
// exact same scratch allocation as before => BYTE-IDENTICAL behavior (verify by diffing the OFF code path).
// When ON (JOLT_TS_ASYNC set), the screener move loop round-robins moves across g_ts_streams[0..S-1] into S
// private scratch slots, async-D2H's each move's per-pattern lnL into a pinned per-move row, ONE sync after the
// chunk, then the host Kahan gather over m runs in the EXACT existing order on identical floats. The stream pool
// and pinned buffer are lazily created and NEVER freed (released at process exit), mirroring the DevBuf pool.
// JOLT_TS_NSTREAMS sets the pool size K (default 8). The active stream count S used by the screener is
// min(K, nMoves) when async, else 1.
// ---- RETIRED EXPERIMENTS (2026-07-14): the env surface is GONE -- these can no longer be set and the strings
// vanish from the binary, so they leave the flag audit entirely (GPU-BINARIES.md §5).
//   JOLT_TS_ASYNC      -- async stream pool for the screener. Measured NULL; retired (project-gpu-tree-search).
//   JOLT_TS_BATCHFOLD  -- 2D-grid batched screener folds (blockIdx.y=move). Never beat the shipped path.
// Held as `constexpr false` rather than hand-deleted ON PURPOSE: these gates are interwoven with the PRODUCTION
// screener's if/else-if chain (`if (g_ts_async && ... && !TS_GPURED) ... else if (g_ts_batchfold && ...)`) and they
// size device buffers. With constexpr, the COMPILER dead-strips every branch provably -- none of the regression risk
// of hand-editing the hottest CUDA path for a cosmetic gain. Gate: production lnL must be BIT-IDENTICAL.
// (The experiments remain recoverable from git history if ever revisited.)
static constexpr bool g_ts_async           = false;
static constexpr bool g_ts_async_check     = false;
static constexpr bool g_ts_batchfold       = false;
static constexpr bool g_ts_batchfold_check = false;
static constexpr int  g_ts_batchfold_B     = 64;   // referenced in dead `g_ts_batchfold ? ... : 1` sizing exprs
// Lazily-initialized stream pool. g_ts_nstreams is the realized pool size; 0 until first ts_streams() call.
static cudaStream_t* g_ts_streams = nullptr;
static int           g_ts_nstreams = 0;
// Return the process-global stream pool (lazily created, never freed). Returns nullptr / sets *K=0 on any error
// (caller then falls back to the default-stream serial path => still correct, just unoptimized).
static cudaStream_t* ts_streams(int* K) {
    if (g_ts_streams == nullptr) {
        int want = 8;
        if (const char* e = getenv("JOLT_TS_NSTREAMS")) { int v = atoi(e); if (v >= 1) want = v; }
        cudaStream_t* arr = (cudaStream_t*)malloc((size_t)want * sizeof(cudaStream_t));
        if (!arr) { *K = 0; return nullptr; }
        int made = 0;
        for (int i = 0; i < want; i++) { if (cudaStreamCreate(&arr[i]) != cudaSuccess) break; made++; }
        if (made == 0) { free(arr); *K = 0; return nullptr; }
        g_ts_streams = arr; g_ts_nstreams = made;
        if (getenv("JOLT_DEBUG")) fprintf(stderr,"[TS-ASYNC] stream pool created: %d streams\n", made);
    }
    *K = g_ts_nstreams; return g_ts_streams;
}
// Pinned host staging for async per-move D2H of the per-pattern lnL (one contiguous row of chunk0 doubles per
// move). Lazily grown (cudaFreeHost+cudaMallocHost on growth); never freed at the end. Used ONLY when async.
static double* g_ts_pin_patlh = nullptr;
static size_t  g_ts_pin_cap   = 0;   // capacity in doubles
static bool ts_pin_ensure(size_t needDoubles) {
    if (needDoubles <= g_ts_pin_cap && g_ts_pin_patlh) return true;
    if (g_ts_pin_patlh) { cudaFreeHost(g_ts_pin_patlh); g_ts_pin_patlh = nullptr; g_ts_pin_cap = 0; }
    if (cudaMallocHost((void**)&g_ts_pin_patlh, needDoubles*sizeof(double)) != cudaSuccess) {
        g_ts_pin_patlh = nullptr; g_ts_pin_cap = 0; return false; }
    g_ts_pin_cap = needDoubles; return true;
}
// Inc 2 reopt valpool — device-side staging for the per-edge coefficient block [v0|v1|v2|rs] (the per-stream block is
// valBlk = 3*ncat*ns + ncat doubles: v0,v1,v2 each ncat*ns then rscale ncat). Replaces the per-edge memcpyToSymbol of
// g_val0/g_val1/g_val2/g_rscale (constant memory, default-stream only) with ONE async H2D into a private slot the new
// kj_derv_fused_args reads as kernel args. Lazily grown (cudaFree+cudaMalloc on growth); NEVER freed (released at
// process exit), mirroring g_ts_pin_patlh and the DevBuf pool. Inc 2 keeps reopt SERIAL, so one slot suffices, but the
// pool is sized S*valBlk so Inc 3 can round-robin slots without re-plumbing.
static double* g_ts_valpool = nullptr; static size_t g_ts_valpool_cap = 0;
static bool ts_valpool_ensure(size_t needDoubles){
  if(needDoubles<=g_ts_valpool_cap && g_ts_valpool) return true;
  if(g_ts_valpool){cudaFree(g_ts_valpool);g_ts_valpool=nullptr;g_ts_valpool_cap=0;}
  if(cudaMalloc((void**)&g_ts_valpool,needDoubles*sizeof(double))!=cudaSuccess){g_ts_valpool=nullptr;g_ts_valpool_cap=0;return false;}
  g_ts_valpool_cap=needDoubles; return true; }

// ============================================================================================================
// TS.8 GPU PARSIMONY (Phase A) — batched per-taxon-insertion branch scoring.
// The stepwise-addition parsimony builder (computeParsimonyTree) is O(nseq^2 * patterns) and CPU-only ->
// 214s @ AA-100K, ~2780s @ 1M (stalls), ~28000s @ 10M; it scales linearly with #patterns. This kernel scores
// ALL candidate insertion branches for ONE taxon in a SINGLE launch (one block per branch), reproducing the
// CPU Fitch (computeParsimonyBranchFast + the internal combine in computePartialParsimonyFast) EXACTLY.
// partial_pars is bit-packed: 32 sites / UINT, nstates UINTs per site-block. Per branch b (endpoints L,R):
//   z[i] = L[i]&R[i];  w1 = ~OR_i(z[i]);  local += popc(w1);  z[i] |= w1 & (L[i]|R[i]);     (internal combine)
//   w2 = ~OR_i(tip[i]&z[i]);  branch += popc(w2);                                            (branch score)
//   out[b] = tipScore + scoreL[b] + scoreR[b] + sum_blk(local+branch)
// Pure integer => BIT-IDENTICAL to CPU (integer add is associative; padding bits are pre-set in the leaf
// partials exactly as the CPU expects). Host argmin uses first-index tie-break to match `score < best_pars`.
// Additive + gated: nothing here runs unless the host parsimony hook calls the launcher.
// ============================================================================================================
static DevBuf gb_parsTip, gb_parsEndL, gb_parsEndR, gb_parsScoreL, gb_parsScoreR, gb_parsOut;
#define PARS_MAXST 32
#define PARS_TPB 256
__global__ void k_pars_score_branches(
        const unsigned int* __restrict__ tip,     // nstates*nsblk (broadcast across branches)
        const unsigned int* __restrict__ endL,    // B * nstates*nsblk
        const unsigned int* __restrict__ endR,    // B * nstates*nsblk
        const unsigned int* __restrict__ scoreL,  // B  (accumulated subtree score, endpoint L)
        const unsigned int* __restrict__ scoreR,  // B
        unsigned int tipScore,
        int nstates, int nsblk, int B,
        unsigned int* __restrict__ out)           // B
{
    int b = blockIdx.x;
    if (b >= B) return;
    const unsigned int* eL = endL + (size_t)b * nstates * nsblk;
    const unsigned int* eR = endR + (size_t)b * nstates * nsblk;
    unsigned int acc = 0u;
    for (int s = threadIdx.x; s < nsblk; s += blockDim.x) {
        const unsigned int* x = eL  + (size_t)s * nstates;
        const unsigned int* y = eR  + (size_t)s * nstates;
        const unsigned int* t = tip + (size_t)s * nstates;
        unsigned int z[PARS_MAXST];
        unsigned int w1 = 0u;
        for (int i = 0; i < nstates; i++) { z[i] = x[i] & y[i]; w1 |= z[i]; }
        w1 = ~w1;
        acc += __popc(w1);
        unsigned int w2 = 0u;
        for (int i = 0; i < nstates; i++) { z[i] |= w1 & (x[i] | y[i]); w2 |= t[i] & z[i]; }
        w2 = ~w2;
        acc += __popc(w2);
    }
    __shared__ unsigned int sh[PARS_TPB];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        out[b] = sh[0] + tipScore + scoreL[b] + scoreR[b];
}

// Returns 0 on success (h_out[0..B-1] filled), -1 on any failure (caller falls back to CPU).
extern "C" int gpu_parsimony_score_branches(
        const unsigned int* h_tip, unsigned int tipScore,
        const unsigned int* h_endL, const unsigned int* h_endR,
        const unsigned int* h_scoreL, const unsigned int* h_scoreR,
        int nstates, int nsblk, int B, unsigned int* h_out)
{
    if (nstates < 1 || nstates > PARS_MAXST || nsblk < 1 || B < 1) return -1;
    static std::mutex pars_mtx; std::lock_guard<std::mutex> lk(pars_mtx);
    size_t setU   = (size_t)nstates * nsblk;
    size_t setB   = setU * sizeof(unsigned int);
    size_t packB  = (size_t)B * setB;
    size_t scB    = (size_t)B * sizeof(unsigned int);
    if (!devbuf_ensure(gb_parsTip,    setB))  return -1;
    if (!devbuf_ensure(gb_parsEndL,   packB)) return -1;
    if (!devbuf_ensure(gb_parsEndR,   packB)) return -1;
    if (!devbuf_ensure(gb_parsScoreL, scB))   return -1;
    if (!devbuf_ensure(gb_parsScoreR, scB))   return -1;
    if (!devbuf_ensure(gb_parsOut,    scB))   return -1;
    if (cudaMemcpy(gb_parsTip.p,    h_tip,    setB,  cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    if (cudaMemcpy(gb_parsEndL.p,   h_endL,   packB, cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    if (cudaMemcpy(gb_parsEndR.p,   h_endR,   packB, cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    if (cudaMemcpy(gb_parsScoreL.p, h_scoreL, scB,   cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    if (cudaMemcpy(gb_parsScoreR.p, h_scoreR, scB,   cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    k_pars_score_branches<<<B, PARS_TPB>>>(
        (const unsigned int*)gb_parsTip.p, (const unsigned int*)gb_parsEndL.p, (const unsigned int*)gb_parsEndR.p,
        (const unsigned int*)gb_parsScoreL.p, (const unsigned int*)gb_parsScoreR.p,
        tipScore, nstates, nsblk, B, (unsigned int*)gb_parsOut.p);
    if (cudaGetLastError() != cudaSuccess) return -1;
    if (cudaMemcpy(h_out, gb_parsOut.p, scB, cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
    if (cudaDeviceSynchronize() != cudaSuccess) return -1;
    return 0;
}

// ===========================================================================================================
// TS.8 Phase-B v1 keystone: device postorder COMBINE-to-arena. Reuses the EXACT Fitch combine validated by
// k_pars_score_branches (parsval 172472092: AA+DNA bit-identical), but WRITES the combined state-set z[] to an
// arena slot and accumulates the subtree score, instead of scoring against a tip. One block per combine-task;
// all tasks in a launch are one tree level (children already materialised). Inputs A/B are arena slots (idx>=0)
// or resident leaves (idx<0, encoded -(leafid+1)); leaf subtree-score = 0. This is the per-node primitive of the
// recompute-from-resident-leaves design (no per-insertion partial H2D, host driver untouched). NOT yet wired.
__global__ void k_pars_combine_to_arena(
        const unsigned int* __restrict__ leaves,   // nseq * (nstates*nsblk)
        unsigned int* __restrict__ arena,          // maxSlots * (nstates*nsblk)
        unsigned int* __restrict__ arenaScore,     // maxSlots
        const int* __restrict__ taskOut,           // T : destination arena slot
        const int* __restrict__ taskA,             // T : child A (>=0 arena slot, <0 leaf=-(id+1))
        const int* __restrict__ taskB,             // T : child B
        int nstates, int nsblk, int T)
{
    int tk = blockIdx.x;
    if (tk >= T) return;
    size_t setU = (size_t)nstates * nsblk;
    int a = taskA[tk], b = taskB[tk];
    const unsigned int* X = (a >= 0) ? arena  + (size_t)a * setU : leaves + (size_t)(-(a + 1)) * setU;
    const unsigned int* Y = (b >= 0) ? arena  + (size_t)b * setU : leaves + (size_t)(-(b + 1)) * setU;
    unsigned int* O = arena + (size_t)taskOut[tk] * setU;
    unsigned int acc = 0u;
    for (int s = threadIdx.x; s < nsblk; s += blockDim.x) {
        const unsigned int* x = X + (size_t)s * nstates;
        const unsigned int* y = Y + (size_t)s * nstates;
        unsigned int* o       = O + (size_t)s * nstates;
        unsigned int z[PARS_MAXST];
        unsigned int w1 = 0u;
        for (int i = 0; i < nstates; i++) { z[i] = x[i] & y[i]; w1 |= z[i]; }
        w1 = ~w1;
        acc += __popc(w1);
        for (int i = 0; i < nstates; i++) { z[i] |= w1 & (x[i] | y[i]); o[i] = z[i]; }
    }
    __shared__ unsigned int sh[PARS_TPB];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        unsigned int sA = (a >= 0) ? arenaScore[a] : 0u;
        unsigned int sB = (b >= 0) ? arenaScore[b] : 0u;
        arenaScore[taskOut[tk]] = sh[0] + sA + sB;
    }
}

// TS.8 Phase-B v1: score candidate insertion branches by INDEX (no per-branch materialised copy / no H2D of
// partials). One block per candidate k: gather candL[k]/candR[k] from arena(>=0) or resident leaves(<0), combine
// (Fitch), then combine with the new-taxon tip (resident leaf tipLeaf, score 0). Identical math to
// k_pars_score_branches (validated bit-identical), just index-gathered from device-resident buffers.
__global__ void k_pars_score_indexed(
        const unsigned int* __restrict__ leaves,    // nLeaf * setU
        const unsigned int* __restrict__ arena,     // maxSlots * setU
        const unsigned int* __restrict__ arenaScore,// maxSlots
        const int* __restrict__ candL,              // nCand : ref (>=0 arena slot, <0 leaf -(id+1))
        const int* __restrict__ candR,              // nCand
        const unsigned int* __restrict__ tipPart,   // setU : the new-taxon tip partial (resident leaf row)
        int nstates, int nsblk, int nCand,
        unsigned int* __restrict__ out)             // nCand
{
    int k = blockIdx.x;
    if (k >= nCand) return;
    size_t setU = (size_t)nstates * nsblk;
    int a = candL[k], b = candR[k];
    const unsigned int* X = (a >= 0) ? arena + (size_t)a * setU : leaves + (size_t)(-(a + 1)) * setU;
    const unsigned int* Y = (b >= 0) ? arena + (size_t)b * setU : leaves + (size_t)(-(b + 1)) * setU;
    unsigned int acc = 0u;
    for (int s = threadIdx.x; s < nsblk; s += blockDim.x) {
        const unsigned int* x = X + (size_t)s * nstates;
        const unsigned int* y = Y + (size_t)s * nstates;
        const unsigned int* t = tipPart + (size_t)s * nstates;
        unsigned int z[PARS_MAXST];
        unsigned int w1 = 0u;
        for (int i = 0; i < nstates; i++) { z[i] = x[i] & y[i]; w1 |= z[i]; }
        w1 = ~w1;
        acc += __popc(w1);
        unsigned int w2 = 0u;
        for (int i = 0; i < nstates; i++) { z[i] |= w1 & (x[i] | y[i]); w2 |= t[i] & z[i]; }
        w2 = ~w2;
        acc += __popc(w2);
    }
    __shared__ unsigned int sh[PARS_TPB];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        unsigned int sA = (a >= 0) ? arenaScore[a] : 0u;
        unsigned int sB = (b >= 0) ? arenaScore[b] : 0u;
        out[k] = sh[0] + sA + sB;   // tipScore = 0 (leaf)
    }
}

// ===== TS.8 v2 Stage-1: 2D-grid variants (blockIdx.y = pattern-chunk) — saturate the H200 on a SINGLE launch.
// v1 kernels put nsblk in the per-thread stride => grid = task/cand count = 16..191 blocks on a 132-SM GPU (starved).
// Here blockIdx.y indexes a chunk of CH sblk so grid = cnt*nChunk blocks. The popcount is split across chunks and
// accumulated with INTEGER atomicAdd (associative => bit-identical to the v1 sequential reduction). The child
// sub-scores (sA+sB) are added EXACTLY ONCE by the blockIdx.y==0 block. arenaScore/out MUST be pre-zeroed by the host.
__global__ void k_pars_combine_to_arena_2d(
        const unsigned int* __restrict__ leaves, unsigned int* __restrict__ arena,
        unsigned int* __restrict__ arenaScore, const int* __restrict__ taskOut,
        const int* __restrict__ taskA, const int* __restrict__ taskB,
        int nstates, int nsblk, int T, int CH)
{
    int tk = blockIdx.x; if (tk >= T) return;
    int s0 = blockIdx.y * CH; if (s0 >= nsblk) return; int s1 = min(s0 + CH, nsblk);
    size_t setU = (size_t)nstates * nsblk;
    int a = taskA[tk], b = taskB[tk];
    const unsigned int* X = (a >= 0) ? arena + (size_t)a*setU : leaves + (size_t)(-(a+1))*setU;
    const unsigned int* Y = (b >= 0) ? arena + (size_t)b*setU : leaves + (size_t)(-(b+1))*setU;
    unsigned int* O = arena + (size_t)taskOut[tk] * setU;
    unsigned int acc = 0u;
    for (int s = s0 + threadIdx.x; s < s1; s += blockDim.x) {
        const unsigned int* x = X + (size_t)s*nstates; const unsigned int* y = Y + (size_t)s*nstates;
        unsigned int* o = O + (size_t)s*nstates;
        unsigned int z[PARS_MAXST]; unsigned int w1 = 0u;
        for (int i = 0; i < nstates; i++) { z[i] = x[i] & y[i]; w1 |= z[i]; } w1 = ~w1; acc += __popc(w1);
        for (int i = 0; i < nstates; i++) { z[i] |= w1 & (x[i] | y[i]); o[i] = z[i]; }
    }
    __shared__ unsigned int sh[PARS_TPB]; sh[threadIdx.x] = acc; __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) { if (threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x+st]; __syncthreads(); }
    if (threadIdx.x == 0) {
        unsigned int add = sh[0];
        if (blockIdx.y == 0) {                                   // add the child sub-scores exactly once
            unsigned int sA = (a >= 0) ? arenaScore[a] : 0u;
            unsigned int sB = (b >= 0) ? arenaScore[b] : 0u;
            add += sA + sB;
        }
        atomicAdd(&arenaScore[taskOut[tk]], add);
    }
}
__global__ void k_pars_score_indexed_2d(
        const unsigned int* __restrict__ leaves, const unsigned int* __restrict__ arena,
        const unsigned int* __restrict__ arenaScore, const int* __restrict__ candL,
        const int* __restrict__ candR, const unsigned int* __restrict__ tipPart,
        int nstates, int nsblk, int nCand, int CH, unsigned int* __restrict__ out)
{
    int k = blockIdx.x; if (k >= nCand) return;
    int s0 = blockIdx.y * CH; if (s0 >= nsblk) return; int s1 = min(s0 + CH, nsblk);
    size_t setU = (size_t)nstates * nsblk;
    int a = candL[k], b = candR[k];
    const unsigned int* X = (a >= 0) ? arena + (size_t)a*setU : leaves + (size_t)(-(a+1))*setU;
    const unsigned int* Y = (b >= 0) ? arena + (size_t)b*setU : leaves + (size_t)(-(b+1))*setU;
    unsigned int acc = 0u;
    for (int s = s0 + threadIdx.x; s < s1; s += blockDim.x) {
        const unsigned int* x = X + (size_t)s*nstates; const unsigned int* y = Y + (size_t)s*nstates;
        const unsigned int* t = tipPart + (size_t)s*nstates;
        unsigned int z[PARS_MAXST]; unsigned int w1 = 0u;
        for (int i = 0; i < nstates; i++) { z[i] = x[i] & y[i]; w1 |= z[i]; } w1 = ~w1; acc += __popc(w1);
        unsigned int w2 = 0u; for (int i = 0; i < nstates; i++) { z[i] |= w1 & (x[i] | y[i]); w2 |= t[i] & z[i]; } w2 = ~w2; acc += __popc(w2);
    }
    __shared__ unsigned int sh[PARS_TPB]; sh[threadIdx.x] = acc; __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) { if (threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x+st]; __syncthreads(); }
    if (threadIdx.x == 0) {
        unsigned int add = sh[0];
        if (blockIdx.y == 0) {                                   // tipScore = 0 (leaf); add child sub-scores once
            unsigned int sA = (a >= 0) ? arenaScore[a] : 0u;
            unsigned int sB = (b >= 0) ? arenaScore[b] : 0u;
            add += sA + sB;
        }
        atomicAdd(&out[k], add);
    }
}

// ---- Phase-B device-resident state ----
static DevBuf gb_pbLeaves, gb_pbArena, gb_pbArenaScore;
static DevBuf gb_pbTaskOut, gb_pbTaskA, gb_pbTaskB, gb_pbCandL, gb_pbCandR, gb_pbScores;
static int g_pb_nLeaf = 0, g_pb_setU = 0, g_pb_nstates = 0, g_pb_nsblk = 0;
static const void* g_pb_alnId = nullptr;   // [red-team] identity of the alignment whose leaves are resident
static std::mutex g_pb_mtx;   // shared across set_leaves / set_leaf_row / build_and_score (OpenMP 98-tree gen)

// Allocate the resident leaf buffer: nLeaf rows of setU=nstates*nsblk UINTs (rows filled via set_leaf_row).
// Idempotent — safe to call from every OpenMP thread's computeParsimonyTree (dims must match the alignment).
// alnId = an opaque per-alignment token (the Alignment*): build_and_score rejects if a DIFFERENT alignment has
// since clobbered the shared resident leaves (the auto-on default-gate newly exposes this; mismatch => CPU fallback).
extern "C" int gpu_parsimony_set_leaves(int nLeaf, int nstates, int nsblk, const void* alnId) {
    if (nLeaf < 1 || nstates < 1 || nstates > PARS_MAXST || nsblk < 1) return -1;
    std::lock_guard<std::mutex> lk(g_pb_mtx);
    size_t setU = (size_t)nstates * nsblk;
    if (!devbuf_ensure(gb_pbLeaves, (size_t)nLeaf * setU * sizeof(unsigned int))) return -1;
    g_pb_nLeaf = nLeaf; g_pb_setU = (int)setU; g_pb_nstates = nstates; g_pb_nsblk = nsblk; g_pb_alnId = alnId;
    return 0;
}

// Upload ONE bit-packed leaf row (setU UINTs) into resident slot lid (= taxon node->id). Idempotent across
// threads (same lid => byte-identical data, since leaf packs are tree-independent).
extern "C" int gpu_parsimony_set_leaf_row(const unsigned int* h_row, int lid) {
    std::lock_guard<std::mutex> lk(g_pb_mtx);
    if (g_pb_nLeaf < 1 || lid < 0 || lid >= g_pb_nLeaf) return -1;
    unsigned int* base = (unsigned int*)gb_pbLeaves.p;
    if (cudaMemcpy(base + (size_t)lid * g_pb_setU, h_row, (size_t)g_pb_setU * sizeof(unsigned int),
                   cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    return 0;
}

// Recompute all directed partials from resident leaves via the level schedule, then score candidate branches.
extern "C" int gpu_parsimony_build_and_score(
        int maxSlots,
        const int* h_taskOut, const int* h_taskA, const int* h_taskB,
        const int* h_levelStart, int nLevel, int nTask,
        const int* h_candL, const int* h_candR, int tipLeaf,
        int nstates, int nsblk, int nCand, unsigned int* h_scores, const void* alnId) {
    std::lock_guard<std::mutex> lk(g_pb_mtx);   // [red-team] take the lock BEFORE reading shared g_pb_* globals
    if (g_pb_nLeaf < 1) return -1;                                   // leaves not uploaded
    if (alnId != g_pb_alnId) return -1;          // [red-team] a different alignment clobbered residency => CPU fallback
    if (nstates != g_pb_nstates || nsblk != g_pb_nsblk) return -1;   // must match resident leaves
    if (maxSlots < 1 || nCand < 1 || nTask < 0 || nLevel < 0) return -1;
    if (tipLeaf < 0 || tipLeaf >= g_pb_nLeaf) return -1;
    size_t setU = (size_t)nstates * nsblk;
    if (!devbuf_ensure(gb_pbArena,      (size_t)maxSlots * setU * sizeof(unsigned int))) return -1;
    if (!devbuf_ensure(gb_pbArenaScore, (size_t)maxSlots * sizeof(unsigned int)))        return -1;
    if (nTask > 0) {
        if (!devbuf_ensure(gb_pbTaskOut, (size_t)nTask * sizeof(int))) return -1;
        if (!devbuf_ensure(gb_pbTaskA,   (size_t)nTask * sizeof(int))) return -1;
        if (!devbuf_ensure(gb_pbTaskB,   (size_t)nTask * sizeof(int))) return -1;
        if (cudaMemcpy(gb_pbTaskOut.p, h_taskOut, (size_t)nTask*sizeof(int), cudaMemcpyHostToDevice)!=cudaSuccess) return -1;
        if (cudaMemcpy(gb_pbTaskA.p,   h_taskA,   (size_t)nTask*sizeof(int), cudaMemcpyHostToDevice)!=cudaSuccess) return -1;
        if (cudaMemcpy(gb_pbTaskB.p,   h_taskB,   (size_t)nTask*sizeof(int), cudaMemcpyHostToDevice)!=cudaSuccess) return -1;
    }
    if (!devbuf_ensure(gb_pbCandL,  (size_t)nCand*sizeof(int)))          return -1;
    if (!devbuf_ensure(gb_pbCandR,  (size_t)nCand*sizeof(int)))          return -1;
    if (!devbuf_ensure(gb_pbScores, (size_t)nCand*sizeof(unsigned int))) return -1;
    if (cudaMemcpy(gb_pbCandL.p, h_candL, (size_t)nCand*sizeof(int), cudaMemcpyHostToDevice)!=cudaSuccess) return -1;
    if (cudaMemcpy(gb_pbCandR.p, h_candR, (size_t)nCand*sizeof(int), cudaMemcpyHostToDevice)!=cudaSuccess) return -1;
    const unsigned int* dLeaves = (const unsigned int*)gb_pbLeaves.p;
    unsigned int* dArena  = (unsigned int*)gb_pbArena.p;
    unsigned int* dScore  = (unsigned int*)gb_pbArenaScore.p;
    const int* dTaskOut = (const int*)gb_pbTaskOut.p;
    const int* dTaskA   = (const int*)gb_pbTaskA.p;
    const int* dTaskB   = (const int*)gb_pbTaskB.p;
    const unsigned int* dTip = dLeaves + (size_t)tipLeaf * setU;
    // Stage-1 parsimony: 2D-grid Fitch (saturating, atomicAdd) — the shipped path.
    {
        const int CH = 128;                                      // spike-tuned; nChunk=ceil(nsblk/CH) saturates SMs
        const int nChunk = (nsblk + CH - 1) / CH;
        // atomicAdd accumulators MUST start at 0 (combine writes touched slots; score writes nCand)
        if (cudaMemset(dScore, 0, (size_t)maxSlots * sizeof(unsigned int)) != cudaSuccess) return -1;
        if (cudaMemset(gb_pbScores.p, 0, (size_t)nCand * sizeof(unsigned int)) != cudaSuccess) return -1;
        for (int l = 0; l < nLevel; l++) {                       // sequential launches => level L+1 reads level L's scores
            int base = h_levelStart[l], cnt = h_levelStart[l+1] - base;
            if (cnt <= 0) continue;
            k_pars_combine_to_arena_2d<<<dim3(cnt, nChunk), PARS_TPB>>>(dLeaves, dArena, dScore,
                dTaskOut + base, dTaskA + base, dTaskB + base, nstates, nsblk, cnt, CH);
            if (cudaGetLastError() != cudaSuccess) return -1;
        }
        k_pars_score_indexed_2d<<<dim3(nCand, nChunk), PARS_TPB>>>(dLeaves, dArena, dScore,
            (const int*)gb_pbCandL.p, (const int*)gb_pbCandR.p, dTip, nstates, nsblk, nCand, CH,
            (unsigned int*)gb_pbScores.p);
        if (cudaGetLastError() != cudaSuccess) return -1;
    }
    if (cudaMemcpy(h_scores, gb_pbScores.p, (size_t)nCand*sizeof(unsigned int), cudaMemcpyDeviceToHost)!=cudaSuccess) return -1;
    if (cudaDeviceSynchronize() != cudaSuccess) return -1;
    return 0;
}

extern "C" double gpu_lnl_crosscheck(
    int nstates, int nptn, int ncat, int ntax, int nnodes, int nInternal,
    const double* Uinv, const double* UinvRowSum, const double* freq, const double* catProp,
    const double* echild, const unsigned char* tip, const double* ptn_freq,
    const int* desc_isRoot, const int* desc_nchild, const int* desc_outSlot,
    const int* desc_childNode, const int* desc_childIsLeaf, const int* desc_childLeaf, const int* desc_childSlot,
    double pinv, const double* base_invar,   // A3 (+I): pinv + host per-pattern invariant base (nullptr/pinv<=0 => byte-identical, no +I term)
    double* out_patlh)
{
    int ns = nstates;
    if (ns > NS_MAX || ncat > 64) { fprintf(stderr,"[GPU-XCHECK] unsupported ns=%d ncat=%d\n",ns,ncat); return (double)NAN; }

    CC_TOSYM(0, g_Uinv, Uinv, sizeof(double)*ns*ns);
    CC_TOSYM(2, g_UinvRowSum, UinvRowSum, sizeof(double)*ns);
    CC_TOSYM(3, g_freq, freq, sizeof(double)*ns);
    CC_TOSYM(4, g_catw, catProp, sizeof(double)*ncat);

    size_t ecStride = (size_t)ncat*ns*ns;
    size_t slotSz   = (size_t)ncat*ns*nptn;
    DEVB(gb_echild, (size_t)nnodes*ecStride*sizeof(double));
    DEVB(gb_tip,    (size_t)ntax*nptn);
    DEVB(gb_partial,(size_t)(nInternal>0?nInternal:1)*slotSz*sizeof(double));
    DEVB(gb_patlh,  (size_t)nptn*sizeof(double));
    double *d_echild=(double*)gb_echild.p, *d_partial=(double*)gb_partial.p, *d_patlh=(double*)gb_patlh.p;
    unsigned char *d_tip=(unsigned char*)gb_tip.p;
    GCK(cudaMemcpy(d_echild, echild, (size_t)nnodes*ecStride*sizeof(double), cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(d_tip, tip, (size_t)ntax*nptn, cudaMemcpyHostToDevice));

    // A3 (+I): upload the per-pattern invariant base (once) if pinv>0; else d_baseinvar stays null => root term unchanged.
    double* d_baseinvar = nullptr;
    if (pinv > 0.0 && base_invar) {
        DEVB(gb_baseinvar, (size_t)nptn*sizeof(double));
        d_baseinvar = (double*)gb_baseinvar.p;
        GCK(cudaMemcpy(d_baseinvar, base_invar, (size_t)nptn*sizeof(double), cudaMemcpyHostToDevice));
    }

    int TB=256, GB=(nptn+TB-1)/TB;
    for (int idx=0; idx<nInternal; idx++){
        int isRoot = desc_isRoot[idx], nchild = desc_nchild[idx];
        double* out = (desc_outSlot[idx] < 0) ? nullptr : (d_partial + (size_t)desc_outSlot[idx]*slotSz);
        const double* ec[3]={nullptr,nullptr,nullptr};
        const double* p[3]={nullptr,nullptr,nullptr};
        const unsigned char* t[3]={nullptr,nullptr,nullptr};
        for (int k=0;k<nchild && k<3;k++){
            int cn = desc_childNode[idx*3+k];
            if (cn>=0) ec[k] = d_echild + (size_t)cn*ecStride;
            if (desc_childIsLeaf[idx*3+k]) t[k] = d_tip + (size_t)desc_childLeaf[idx*3+k]*nptn;
            else                          p[k] = d_partial + (size_t)desc_childSlot[idx*3+k]*slotSz;
        }
        launch_k1_node(GB,TB,0,ns,nptn,ncat,isRoot,out,d_patlh,nchild,
            ec[0],p[0],t[0], ec[1],p[1],t[1], ec[2],p[2],t[2],
            pinv, d_baseinvar);   // A3 (+I): kernel adds pinv*base_invar at the root fold (d_baseinvar null => unchanged)
    }
    GCK(cudaDeviceSynchronize());
    GCK(cudaGetLastError());

    std::vector<double> patlh(nptn);
    GCK(cudaMemcpy(patlh.data(), d_patlh, (size_t)nptn*sizeof(double), cudaMemcpyDeviceToHost));

    // G.2.0b: hand the per-pattern log|lh_ptn| back to the caller (for the host _pattern_lh[] mirror).
    if (out_patlh) for (int p2=0; p2<nptn; p2++) out_patlh[p2] = patlh[p2];

    // ptn_freq-weighted Kahan log-sum: tree_lh = sum_ptn ptn_freq[ptn]*log|lh_ptn|
    double lnL=0.0, kc=0.0;
    for (int p2=0; p2<nptn; p2++){ double term = ptn_freq[p2]*patlh[p2];
        double y=term-kc, t2=lnL+y; kc=(t2-lnL)-y; lnL=t2; }

    // pool buffers persist (no cudaFree) — reused next call
    return lnL;
}

// G.8.0 — profile-mixture clean-room lnL launcher. Mirrors gpu_lnl_crosscheck but with R = nmix*ncat regimes;
// per-class Uinv/UinvRowSum/freq + per-regime weight live in GLOBAL memory (mix_* pools). echild/tip/partial/patlh
// reuse the single-model pools (resized via DEVB to the larger mixture sizes). NaN on OOM/CUDA error -> CPU fallback.
static DevBuf gb_mUinv, gb_mUrs, gb_mFreq, gb_mWreg, gb_mLhcat;
static DevBuf gb_mClsinv;   // A1 (+I): per-chunk per-class invariant clsinv[nmix][Pn] (lnL mix launcher)
static DevBuf gb_mDval0, gb_mDval1, gb_mDval2;   // G.8.1b central-edge derivative coeffs (R*ns, GLOBAL)

// G.8.2.5 — pick the pattern-tiling factor for a mixture clean-room launcher. perPatternDoubles = the O(nptn) device
// footprint PER PATTERN, in doubles (e.g. nInternal*R*ns for the lnL partial arena; +nPool*R*ns for the derivative
// preorder pool). nTile = ceil(one-shot O(nptn) bytes / 80% free VRAM); JOLT_NTILE overrides. The result is
// chunk-count-INDEPENDENT (per-pattern values are chunk-independent; the Kahan reductions add patterns in order
// 0..nptn-1), so nTile only trades VRAM for kernel launches, never the answer.
static int mix_pick_ntile(int nptn, size_t perPatternDoubles) {
    if (const char* e = getenv("JOLT_NTILE")) { int t=atoi(e); if(t<1)t=1;
        if(getenv("JOLT_DEBUG")) fprintf(stderr,"[MIX-TILE] JOLT_NTILE=%d (nptn=%d)\n",t,nptn); return t; }
    size_t foot = perPatternDoubles * (size_t)nptn * sizeof(double);
    size_t freeB=0, totB=0; int nTile=1;
    if (cudaMemGetInfo(&freeB,&totB)==cudaSuccess && freeB>0) {
        double budget = 0.80*(double)freeB;
        int T = (int)ceil((double)foot/budget); if(T<1)T=1; nTile=T;
    }
    if (getenv("JOLT_DEBUG")) fprintf(stderr,"[MIX-TILE] nptn=%d perPtnDoubles=%zu O(nptn)foot=%.2fGB freeVRAM=%.1fGB -> nTile=%d (chunk~%d)\n",
        nptn,perPatternDoubles,(double)foot/1.073741824e9,(double)freeB/1.073741824e9,nTile,(nptn+nTile-1)/nTile);
    return nTile;
}
extern "C" double gpu_lnl_crosscheck_mix(
    int nstates, int nptn, int ncat, int nmix, int ntax, int nnodes, int nInternal,
    const double* Uinv, const double* UinvRowSum, const double* freq, const double* wreg,
    const double* echild, const unsigned char* tip, const double* ptn_freq,
    const int* desc_isRoot, const int* desc_nchild, const int* desc_outSlot,
    const int* desc_childNode, const int* desc_childIsLeaf, const int* desc_childLeaf, const int* desc_childSlot,
    double* out_patlh, double* out_lhcat,   // G.8.1: out_lhcat (optional) = per-class L_{p,m}, [nmix][nptn]
    double pinv, const double* clsinv)       // A1 (+I): pinv + per-class invariant clsinv[m][ptn]=w_m*pinv*base_invar_m [nmix][nptn]; pinv<=0 => byte-identical (clsinv may be null)
{
    int ns = nstates;
    if (ns > NS_MAX || nmix < 1 || ncat < 1) { fprintf(stderr,"[GPU-XCHECK-MIX] unsupported ns=%d nmix=%d ncat=%d\n",ns,nmix,ncat); return (double)NAN; }
    int R = nmix*ncat;

    // per-class eigen + per-regime weight -> GLOBAL (overflow __constant__ at 320 regimes)
    DEVB(gb_mUinv, (size_t)nmix*ns*ns*sizeof(double));
    DEVB(gb_mUrs,  (size_t)nmix*ns*sizeof(double));
    DEVB(gb_mFreq, (size_t)nmix*ns*sizeof(double));
    DEVB(gb_mWreg, (size_t)R*sizeof(double));
    GCK(cudaMemcpy(gb_mUinv.p, Uinv,       (size_t)nmix*ns*ns*sizeof(double), cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(gb_mUrs.p,  UinvRowSum, (size_t)nmix*ns*sizeof(double),    cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(gb_mFreq.p, freq,       (size_t)nmix*ns*sizeof(double),    cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(gb_mWreg.p, wreg,       (size_t)R*sizeof(double),          cudaMemcpyHostToDevice));
    double *d_Uinv=(double*)gb_mUinv.p, *d_Urs=(double*)gb_mUrs.p, *d_Freq=(double*)gb_mFreq.p, *d_Wreg=(double*)gb_mWreg.p;

    size_t ecStride = (size_t)R*ns*ns;        // echild per node, regime-strided — PATTERN-INDEPENDENT (built once)

    // G.8.2.5 — PATTERN TILING (mirrors single-model gpu_jolt_optimize G.7.1). The partial arena nInternal*R*ns*nptn
    // dominates VRAM (MEOW80+G4 full-data ~100 GB lnL / ~145 GB derivative); chunking nptn into nTile contiguous pieces
    // and running a full postorder per chunk shrinks every O(nptn) buffer by ~nTile. echild/eigen carry NO pattern axis
    // (built once). The per-pattern patlh[p] is chunk-INDEPENDENT and the Kahan lnL accumulator adds patterns in the
    // SAME order 0..nptn-1 across chunks => BIT-IDENTICAL to one-shot for any nTile. Auto-pick from free VRAM (80%
    // target) or JOLT_NTILE override.
    int nTile = mix_pick_ntile(nptn, (size_t)(nInternal>0?nInternal:1)*(size_t)R*ns + (size_t)(out_lhcat?nmix:0) + 2);
    int chunk0 = (nptn + nTile - 1) / nTile;
    size_t slotSzMax = (size_t)R*ns*chunk0;
    DEVB(gb_echild, (size_t)nnodes*ecStride*sizeof(double));
    DEVB(gb_tip,    (size_t)ntax*chunk0);
    DEVB(gb_partial,(size_t)(nInternal>0?nInternal:1)*slotSzMax*sizeof(double));
    DEVB(gb_patlh,  (size_t)chunk0*sizeof(double));
    double *d_echild=(double*)gb_echild.p, *d_partial=(double*)gb_partial.p, *d_patlh=(double*)gb_patlh.p;
    unsigned char *d_tip=(unsigned char*)gb_tip.p;
    double *d_lhcat = nullptr;
    if (out_lhcat) { DEVB(gb_mLhcat, (size_t)nmix*chunk0*sizeof(double)); d_lhcat=(double*)gb_mLhcat.p; }   // G.8.1
    double *d_clsinv = nullptr;   // A1 (+I): per-chunk per-class invariant [nmix][Pn] (only when pinv>0)
    if (pinv > 0.0 && clsinv) { DEVB(gb_mClsinv, (size_t)nmix*chunk0*sizeof(double)); d_clsinv=(double*)gb_mClsinv.p; }
    GCK(cudaMemcpy(d_echild, echild, (size_t)nnodes*ecStride*sizeof(double), cudaMemcpyHostToDevice));      // pattern-independent

    std::vector<unsigned char> tipChunk((size_t)ntax*chunk0);
    std::vector<double> patlh(chunk0), lhcatChunk((size_t)(out_lhcat?nmix:0)*chunk0);
    std::vector<double> clsinvChunk((size_t)(d_clsinv?nmix:0)*chunk0);   // A1 (+I): host staging for the chunk slice
    double lnL=0.0, kc=0.0;
    int TB=256;
    for (int tchunk=0; tchunk<nTile; tchunk++){
        int pOff=tchunk*chunk0, p1=pOff+chunk0; if(p1>nptn)p1=nptn; int Pn=p1-pOff; if(Pn<=0) break;
        size_t slotSz=(size_t)R*ns*Pn; int GB=(Pn+TB-1)/TB;
        for(int a=0;a<ntax;a++) memcpy(&tipChunk[(size_t)a*Pn], tip+(size_t)a*nptn+pOff, (size_t)Pn);
        GCK(cudaMemcpy(d_tip, tipChunk.data(), (size_t)ntax*Pn, cudaMemcpyHostToDevice));
        if (d_clsinv) {   // A1 (+I): upload this chunk's per-class invariant slice [nmix][Pn] (stride Pn matches the kernel's nptn=Pn)
            for(int m=0;m<nmix;m++) memcpy(&clsinvChunk[(size_t)m*Pn], clsinv+(size_t)m*nptn+pOff, (size_t)Pn*sizeof(double));
            GCK(cudaMemcpy(d_clsinv, clsinvChunk.data(), (size_t)nmix*Pn*sizeof(double), cudaMemcpyHostToDevice)); }
        for (int idx=0; idx<nInternal; idx++){
            int isRoot = desc_isRoot[idx], nchild = desc_nchild[idx];
            double* out = (desc_outSlot[idx] < 0) ? nullptr : (d_partial + (size_t)desc_outSlot[idx]*slotSz);
            const double* ec[3]={nullptr,nullptr,nullptr};
            const double* p[3]={nullptr,nullptr,nullptr};
            const unsigned char* t[3]={nullptr,nullptr,nullptr};
            for (int k=0;k<nchild && k<3;k++){
                int cn = desc_childNode[idx*3+k];
                if (cn>=0) ec[k] = d_echild + (size_t)cn*ecStride;
                if (desc_childIsLeaf[idx*3+k]) t[k] = d_tip + (size_t)desc_childLeaf[idx*3+k]*Pn;   // tip stride = chunk width
                else                          p[k] = d_partial + (size_t)desc_childSlot[idx*3+k]*slotSz;
            }
            dim3 gMix = isRoot ? dim3(GB,1) : dim3(GB,R);   // G.8.2.3: root accumulates over regimes (1D); non-root parallelises them
            k1_node_mix<<<gMix,TB>>>(ns,Pn,ncat,nmix,isRoot,out,d_patlh,d_Uinv,d_Urs,d_Freq,d_Wreg,d_lhcat,pinv,d_clsinv,nchild,
                ec[0],p[0],t[0], ec[1],p[1],t[1], ec[2],p[2],t[2]);
        }
        GCK(cudaDeviceSynchronize());
        GCK(cudaGetLastError());
        GCK(cudaMemcpy(patlh.data(), d_patlh, (size_t)Pn*sizeof(double), cudaMemcpyDeviceToHost));
        if (out_patlh) for (int p2=0; p2<Pn; p2++) out_patlh[pOff+p2] = patlh[p2];
        if (out_lhcat) { GCK(cudaMemcpy(lhcatChunk.data(), d_lhcat, (size_t)nmix*Pn*sizeof(double), cudaMemcpyDeviceToHost));
            for (int m=0;m<nmix;m++) for (int p2=0;p2<Pn;p2++) out_lhcat[(size_t)m*nptn+pOff+p2] = lhcatChunk[(size_t)m*Pn+p2]; }   // G.8.1
        for (int p2=0; p2<Pn; p2++){ double term = ptn_freq[pOff+p2]*patlh[p2];   // continuous Kahan, order 0..nptn-1 => bit-identical
            double y=term-kc, t2=lnL+y; kc=(t2-lnL)-y; lnL=t2; }
    }
    return lnL;
}

// G.8.1b — clean-room single-edge derivative launcher for PROFILE MIXTURES. Mirrors gpu_derv_crosscheck but:
// (1) the descriptor sweep runs k1_node_mix (isRoot=0) so each endpoint writes its R=nmix*ncat regime eigen
// partials; (2) the central-edge coefficients dval0/1/2[R*ns] are built per-CLASS (eval_{m,x}) with weight
// w_m*catProp_c=wreg[r] and uploaded to GLOBAL memory (R*ns exceeds the __constant__ 64-cat budget); (3) leaf
// endpoints synthesize per-class tip eigen via k_leaf_eig_mix. Returns df=Σ_ptn freq·(d1/lh) (un-negated), with
// *out_ddf=Σ freq·(d2/lh−(d1/lh)²) and *out_lnL=tree lnL at t. NaN on CUDA error.
extern "C" double gpu_derv_crosscheck_mix(
    int nstates, int nptn, int ncat, int nmix, int ntax, int nnodes, int nInternal,
    const double* Uinv, const double* UinvRowSum, const double* freq, const double* wreg,
    const double* echild, const unsigned char* tip, const double* ptn_freq,
    const int* desc_isRoot, const int* desc_nchild, const int* desc_outSlot,
    const int* desc_childNode, const int* desc_childIsLeaf, const int* desc_childLeaf, const int* desc_childSlot,
    int nodeSlot, int nodeLeafTax, int dadSlot, int dadLeafTax,
    const double* evalC, const double* catRate, double t,
    double* out_ddf, double* out_lnL)
{
    int ns = nstates;
    if (ns > NS_MAX || nmix < 1 || ncat < 1) { fprintf(stderr,"[GPU-DERV-MIX] unsupported ns=%d nmix=%d ncat=%d\n",ns,nmix,ncat); return (double)NAN; }
    int R = nmix*ncat;
    (void)desc_isRoot;   // all entries isRoot=0 for the derivative sweep

    // per-class eigen + per-regime weight -> GLOBAL (same pools as the lnL mix path)
    DEVB(gb_mUinv, (size_t)nmix*ns*ns*sizeof(double));
    DEVB(gb_mUrs,  (size_t)nmix*ns*sizeof(double));
    DEVB(gb_mFreq, (size_t)nmix*ns*sizeof(double));
    DEVB(gb_mWreg, (size_t)R*sizeof(double));
    GCK(cudaMemcpy(gb_mUinv.p, Uinv,       (size_t)nmix*ns*ns*sizeof(double), cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(gb_mUrs.p,  UinvRowSum, (size_t)nmix*ns*sizeof(double),    cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(gb_mFreq.p, freq,       (size_t)nmix*ns*sizeof(double),    cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(gb_mWreg.p, wreg,       (size_t)R*sizeof(double),          cudaMemcpyHostToDevice));
    double *d_Uinv=(double*)gb_mUinv.p, *d_Urs=(double*)gb_mUrs.p, *d_Freq=(double*)gb_mFreq.p, *d_Wreg=(double*)gb_mWreg.p;

    // central-edge derivative coeffs dval{0,1,2}[r*ns+x], r=m*ncat+c: cof=eval_{m,x}*rate_c; v0=exp(cof*t)*wreg[r];
    // v1=cof*v0; v2=cof*cof*v0. Per-CLASS eigenvalues; π_m is NOT here (it is already in the eigen-space partials).
    std::vector<double> v0((size_t)R*ns), v1((size_t)R*ns), v2((size_t)R*ns);
    for (int m=0;m<nmix;m++) for (int c=0;c<ncat;c++){ int r=m*ncat+c; double rc=catRate[c], wr=wreg[r];
        for (int x=0;x<ns;x++){ double cof=evalC[(size_t)m*ns+x]*rc, e=exp(cof*t)*wr;
            v0[(size_t)r*ns+x]=e; v1[(size_t)r*ns+x]=cof*e; v2[(size_t)r*ns+x]=cof*cof*e; } }
    DEVB(gb_mDval0,(size_t)R*ns*sizeof(double)); DEVB(gb_mDval1,(size_t)R*ns*sizeof(double)); DEVB(gb_mDval2,(size_t)R*ns*sizeof(double));
    GCK(cudaMemcpy(gb_mDval0.p, v0.data(), (size_t)R*ns*sizeof(double), cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(gb_mDval1.p, v1.data(), (size_t)R*ns*sizeof(double), cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(gb_mDval2.p, v2.data(), (size_t)R*ns*sizeof(double), cudaMemcpyHostToDevice));
    double *d_v0=(double*)gb_mDval0.p, *d_v1=(double*)gb_mDval1.p, *d_v2=(double*)gb_mDval2.p;

    size_t ecStride=(size_t)R*ns*ns, slotSz=(size_t)R*ns*nptn;
    DEVB(gb_echild, (size_t)nnodes*ecStride*sizeof(double));
    DEVB(gb_tip,    (size_t)ntax*nptn);
    DEVB(gb_partial,(size_t)(nInternal>0?nInternal:1)*slotSz*sizeof(double));
    DEVB(gb_pdf,    (size_t)nptn*sizeof(double));
    DEVB(gb_pddf,   (size_t)nptn*sizeof(double));
    DEVB(gb_patlh,  (size_t)nptn*sizeof(double));
    double *d_echild=(double*)gb_echild.p, *d_partial=(double*)gb_partial.p;
    double *d_pdf=(double*)gb_pdf.p, *d_pddf=(double*)gb_pddf.p, *d_patlh=(double*)gb_patlh.p;
    unsigned char *d_tip=(unsigned char*)gb_tip.p;
    GCK(cudaMemcpy(d_echild,echild,(size_t)nnodes*ecStride*sizeof(double),cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(d_tip,tip,(size_t)ntax*nptn,cudaMemcpyHostToDevice));

    int TB=256, GB=(nptn+TB-1)/TB;
    for (int idx=0; idx<nInternal; idx++){
        int nchild=desc_nchild[idx];
        double* out=(desc_outSlot[idx]<0)?nullptr:(d_partial+(size_t)desc_outSlot[idx]*slotSz);
        const double* ec[3]={nullptr,nullptr,nullptr}; const double* p[3]={nullptr,nullptr,nullptr}; const unsigned char* tp[3]={nullptr,nullptr,nullptr};
        for (int k=0;k<nchild && k<3;k++){ int cn=desc_childNode[idx*3+k];
            if (cn>=0) ec[k]=d_echild+(size_t)cn*ecStride;
            if (desc_childIsLeaf[idx*3+k]) tp[k]=d_tip+(size_t)desc_childLeaf[idx*3+k]*nptn;
            else                          p[k]=d_partial+(size_t)desc_childSlot[idx*3+k]*slotSz; }
        k1_node_mix<<<dim3(GB,R),TB>>>(ns,nptn,ncat,nmix,/*isRoot=*/0,out,d_patlh,d_Uinv,d_Urs,d_Freq,d_Wreg,/*out_lhcat=*/nullptr,/*pinv=*/0.0,/*clsinv=*/nullptr,nchild,
            ec[0],p[0],tp[0], ec[1],p[1],tp[1], ec[2],p[2],tp[2]);
    }
    GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError());

    // resolve node_eig/dad_eig: internal slot, or synthesize a leaf endpoint's per-class tip eigen (k_leaf_eig_mix)
    const double *node_eig, *dad_eig;
    if (nodeSlot >= 0) node_eig = d_partial + (size_t)nodeSlot*slotSz;
    else { DEVB(gb_nodeleaf, slotSz*sizeof(double));
           k_leaf_eig_mix<<<GB,TB>>>(ns,nptn,ncat,nmix,d_tip+(size_t)nodeLeafTax*nptn,d_Uinv,d_Urs,(double*)gb_nodeleaf.p); node_eig=(double*)gb_nodeleaf.p; }
    if (dadSlot >= 0) dad_eig = d_partial + (size_t)dadSlot*slotSz;
    else { DEVB(gb_dadleaf, slotSz*sizeof(double));
           k_leaf_eig_mix<<<GB,TB>>>(ns,nptn,ncat,nmix,d_tip+(size_t)dadLeafTax*nptn,d_Uinv,d_Urs,(double*)gb_dadleaf.p); dad_eig=(double*)gb_dadleaf.p; }

    k2_derv_mix<<<GB,TB>>>(ns,nptn,R,node_eig,dad_eig,d_v0,d_v1,d_v2,d_pdf,d_pddf,d_patlh);
    GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError());

    std::vector<double> pdf(nptn),pddf(nptn),patlh(nptn);
    GCK(cudaMemcpy(pdf.data(),d_pdf,(size_t)nptn*sizeof(double),cudaMemcpyDeviceToHost));
    GCK(cudaMemcpy(pddf.data(),d_pddf,(size_t)nptn*sizeof(double),cudaMemcpyDeviceToHost));
    GCK(cudaMemcpy(patlh.data(),d_patlh,(size_t)nptn*sizeof(double),cudaMemcpyDeviceToHost));

    double df=0,kdf=0, ddf=0,kddf=0, lnL=0,kl=0;
    for (int p2=0;p2<nptn;p2++){ double f=ptn_freq[p2];
        { double term=f*pdf[p2],  y=term-kdf,  s=df +y; kdf =(s-df )-y; df =s; }
        { double term=f*pddf[p2], y=term-kddf, s=ddf+y; kddf=(s-ddf)-y; ddf=s; }
        { double term=f*patlh[p2],y=term-kl,   s=lnL+y; kl  =(s-lnL)-y; lnL=s; } }
    if (out_ddf) *out_ddf=ddf;
    if (out_lnL) *out_lnL=lnL;
    return df;
}

// G.8.2.1a — clean-room ALL-BRANCH derivative launcher for PROFILE MIXTURES (the Ji-2020 linear-time gradient: one
// postorder + one preorder sweep yields df/ddf for EVERY edge, vs the single-edge gpu_derv_crosscheck_mix's two-
// sub-root split per edge). Rooted at an internal node `root`; out_df[v]/out_ddf[v] hold d(lnL)/db_v and the 2nd
// derivative for edge v->parent (the root entry stays 0). Reuses the validated mixture pieces: k1_node_mix
// (postorder lower partials + root-child upper seeds), k7_pre_mix (preorder upper partials), k2_derv_mix (per-edge
// reduction). NEW vs the lnL/single-edge path: the eigenvectors d_U (k7_pre_mix's up-map) AND the per-node per-
// regime expfac = exp(eval_m[i]·catRate_c·b_u) are uploaded; the O(tree-height) preorder pool recycles slots.
// Returns 0.0 on success (results in out_df/out_ddf), NaN on CUDA error.
static DevBuf gb_mU, gb_expfac, gb_prepool;   // G.8.2.1a: eigenvectors (up-map), per-node expfac, O(depth) preorder pool
static DevBuf gb_baseinv;   // A1 (+I): per-chunk COMBINED invariant base_invar_comb[Pn] (all-branch derivative mix launcher)
extern "C" double gpu_allbranch_derv_crosscheck_mix(
    int nstates, int nptn, int ncat, int nmix, int ntax, int nnodes, int root,
    const double* Uinv, const double* U, const double* UinvRowSum, const double* freq, const double* wreg,
    const double* evalC, const double* catRate,
    const double* echild, const double* expfac, const unsigned char* tip, const double* ptn_freq,
    const int* node_nchild, const int* node_child, const int* node_leaf, const double* node_parentLen,
    double* out_df, double* out_ddf,
    double pinv, const double* base_invar_comb)   // A1 (+I): pinv + COMBINED invariant Σ_m w_m·base_invar_m [nptn]; the +I term is branch-INDEPENDENT so it enters ONLY the 1/Lp denominator (k2_derv_mix_inv). pinv<=0 => byte-identical (k2_derv_mix)
{
    int ns = nstates;
    if (ns > NS_MAX || nmix < 1 || ncat < 1) { fprintf(stderr,"[GPU-ALLDERV-MIX] unsupported ns=%d nmix=%d ncat=%d\n",ns,nmix,ncat); return (double)NAN; }
    int R = nmix*ncat;

    // ---- rebuild topology from flat arrays (node ids = caller's DFS index, rooted at `root`) ----
    std::vector<std::vector<int>> child(nnodes);
    std::vector<int> leaf(nnodes);
    std::vector<double> brlen(node_parentLen, node_parentLen+nnodes);
    for (int u=0;u<nnodes;u++){ leaf[u]=node_leaf[u];
        for (int k=0;k<node_nchild[u] && k<3;k++){ int c=node_child[u*3+k]; if (c>=0) child[u].push_back(c); } }
    std::vector<int> postorder; std::vector<int> slot(nnodes,-1);
    std::function<void(int)> dfs=[&](int u){ for(int c:child[u]) dfs(c); if(leaf[u]<0){ slot[u]=(int)postorder.size(); postorder.push_back(u);} };
    dfs(root); int nInternal=(int)postorder.size();
    int treeH=0; std::function<void(int,int)> ddfs=[&](int u,int d){ if(d>treeH)treeH=d; for(int c:child[u]) ddfs(c,d+1); }; ddfs(root,0);
    int nPool=treeH+2;   // O(depth): preorder holds one upper-partial slot per ancestor on the current path (peak == treeH)
    if(getenv("ALLDERV_DBG")) fprintf(stderr,"[ALLDERV-DBG] entry ns=%d nptn=%d ncat=%d nmix=%d nnodes=%d root=%d nInternal=%d treeH=%d nPool=%d R=%d\n",ns,nptn,ncat,nmix,nnodes,root,nInternal,treeH,nPool,R);

    size_t ecStride=(size_t)R*ns*ns, exStride=(size_t)R*ns;   // slotSz is chunk-scoped (G.8.2.5 tiling)

    // ---- per-class eigen (Uinv down-map, U up-map) + per-regime weight -> GLOBAL ----
    DEVB(gb_mUinv, (size_t)nmix*ns*ns*sizeof(double));
    DEVB(gb_mU,    (size_t)nmix*ns*ns*sizeof(double));
    DEVB(gb_mUrs,  (size_t)nmix*ns*sizeof(double));
    DEVB(gb_mFreq, (size_t)nmix*ns*sizeof(double));     // k1_node_mix isRoot=0 ignores freq, but the signature needs it
    DEVB(gb_mWreg, (size_t)R*sizeof(double));
    GCK(cudaMemcpy(gb_mUinv.p, Uinv,       (size_t)nmix*ns*ns*sizeof(double), cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(gb_mU.p,    U,          (size_t)nmix*ns*ns*sizeof(double), cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(gb_mUrs.p,  UinvRowSum, (size_t)nmix*ns*sizeof(double),    cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(gb_mFreq.p, freq,       (size_t)nmix*ns*sizeof(double),    cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(gb_mWreg.p, wreg,       (size_t)R*sizeof(double),          cudaMemcpyHostToDevice));
    double *d_Uinv=(double*)gb_mUinv.p, *d_U=(double*)gb_mU.p, *d_Urs=(double*)gb_mUrs.p, *d_Freq=(double*)gb_mFreq.p, *d_Wreg=(double*)gb_mWreg.p;

    // G.8.2.5 — PATTERN TILING for the all-branch gradient. The two O(nptn) partial arenas dominate VRAM:
    // gb_partial = nInternal*R*ns*nptn (all postorder lower partials, resident) + gb_prepool = nPool*R*ns*nptn
    // (O(depth) preorder pool). MEOW80+G4 full-data ~145 GB OOMs even an H200. Chunk nptn into nTile contiguous
    // pieces; per chunk run a FULL postorder + preorder proc sweep and accumulate each edge's df/ddf into a
    // CONTINUOUS per-edge Kahan accumulator carried across chunks. echild/expfac/eigen have NO pattern axis
    // (uploaded once). Per-pattern pdf[p]/pddf[p] are chunk-INDEPENDENT and each edge's Kahan reduction adds
    // patterns in order 0..nptn-1 across chunks => BIT-IDENTICAL to one-shot for any nTile (mirrors the lnL launcher).
    int nTile  = mix_pick_ntile(nptn, (size_t)(nInternal+nPool+1)*R*ns + 3);
    int chunk0 = (nptn + nTile - 1) / nTile;
    size_t slotSzMax = (size_t)R*ns*chunk0;

    // ---- echild + expfac uploaded ONCE (pattern-independent); partial arenas sized to chunk0 ----
    DEVB(gb_echild, (size_t)nnodes*ecStride*sizeof(double));
    DEVB(gb_expfac, (size_t)nnodes*exStride*sizeof(double));
    DEVB(gb_tip,    (size_t)ntax*chunk0);
    DEVB(gb_partial,(size_t)(nInternal>0?nInternal:1)*slotSzMax*sizeof(double));
    DEVB(gb_prepool,(size_t)nPool*slotSzMax*sizeof(double));
    DEVB(gb_pdf,    (size_t)chunk0*sizeof(double));
    DEVB(gb_pddf,   (size_t)chunk0*sizeof(double));
    DEVB(gb_patlh,  (size_t)chunk0*sizeof(double));
    DEVB(gb_mDval0, (size_t)R*ns*sizeof(double)); DEVB(gb_mDval1,(size_t)R*ns*sizeof(double)); DEVB(gb_mDval2,(size_t)R*ns*sizeof(double));
    DEVB(gb_nodeleaf, slotSzMax*sizeof(double));   // scratch for a leaf endpoint's lower eigen partial (k_leaf_eig_mix)
    double *d_baseinvar=nullptr;   // A1 (+I): per-chunk combined invariant slice [Pn] (only when pinv>0)
    if (pinv > 0.0 && base_invar_comb) { DEVB(gb_baseinv, (size_t)chunk0*sizeof(double)); d_baseinvar=(double*)gb_baseinv.p; }
    double *d_echild=(double*)gb_echild.p, *d_expfac=(double*)gb_expfac.p, *d_partial=(double*)gb_partial.p, *d_prepool=(double*)gb_prepool.p;
    double *d_pdf=(double*)gb_pdf.p, *d_pddf=(double*)gb_pddf.p, *d_patlh=(double*)gb_patlh.p;
    double *d_v0=(double*)gb_mDval0.p, *d_v1=(double*)gb_mDval1.p, *d_v2=(double*)gb_mDval2.p, *d_tipeig=(double*)gb_nodeleaf.p;
    unsigned char *d_tip=(unsigned char*)gb_tip.p;
    GCK(cudaMemcpy(d_echild,echild,(size_t)nnodes*ecStride*sizeof(double),cudaMemcpyHostToDevice));   // pattern-independent
    GCK(cudaMemcpy(d_expfac,expfac,(size_t)nnodes*exStride*sizeof(double),cudaMemcpyHostToDevice));   // pattern-independent

    int TB=256;
    // chunk-scoped pattern window (referenced by the lambdas below): updated each tile iteration
    int Pn=0, pOff=0, GB=0; size_t slotSz=0;
    std::vector<unsigned char> tipChunk((size_t)ntax*chunk0);
    // per-edge CONTINUOUS Kahan accumulators (carried across chunks => addition order 0..nptn-1 per edge)
    std::vector<double> accDf(nnodes,0.0), accDfK(nnodes,0.0), accDdf(nnodes,0.0), accDdfK(nnodes,0.0);
    for(int v=0;v<nnodes;v++){ out_df[v]=0.0; out_ddf[v]=0.0; }

    // child-args helper (exclude `excl`, -1 for none): fills echild/partial/tip pointers for k1_node_mix
    auto fillChild=[&](int u,int excl,int& nch,const double** ec,const double** p,const unsigned char** t){
        nch=0; for(int k=0;k<3;k++){ec[k]=nullptr;p[k]=nullptr;t[k]=nullptr;}
        for(int c:child[u]){ if(c==excl||nch>=3) continue; ec[nch]=d_echild+(size_t)c*ecStride;
            if(leaf[c]>=0) t[nch]=d_tip+(size_t)leaf[c]*Pn; else p[nch]=d_partial+(size_t)slot[c]*slotSz; nch++; } };

    // per-edge df/ddf reduction: ADD this chunk's patterns continuously into accDf[v]/accDdf[v] (ptn_freq-weighted
    // Kahan). Carrying (acc,accK) across chunks makes the per-edge sum one continuous Kahan sweep over 0..nptn-1.
    auto reduceInto=[&](int v){
        std::vector<double> pdf(Pn),pddf(Pn);
        cudaMemcpy(pdf.data(),d_pdf,(size_t)Pn*sizeof(double),cudaMemcpyDeviceToHost);
        cudaMemcpy(pddf.data(),d_pddf,(size_t)Pn*sizeof(double),cudaMemcpyDeviceToHost);
        double D=accDf[v],kd=accDfK[v],DD=accDdf[v],kdd=accDdfK[v];
        for(int p=0;p<Pn;p++){ double f=ptn_freq[pOff+p];
            { double term=f*pdf[p],  y=term-kd,  s=D +y; kd =(s-D )-y; D =s; }
            { double term=f*pddf[p], y=term-kdd, s=DD+y; kdd=(s-DD)-y; DD=s; } }
        accDf[v]=D; accDfK[v]=kd; accDdf[v]=DD; accDdfK[v]=kdd; };
    // resolve v's lower partial (internal slot, or synthesise a leaf tip eigen into the scratch buffer)
    auto edgeNodePtr=[&](int v)->const double*{
        if(leaf[v]<0) return d_partial+(size_t)slot[v]*slotSz;
        k_leaf_eig_mix<<<GB,TB>>>(ns,Pn,ncat,nmix,d_tip+(size_t)leaf[v]*Pn,d_Uinv,d_Urs,d_tipeig); return d_tipeig; };
    // central-edge coeffs for t=b_v: dval{0,1,2}[r*ns+x]=exp(eval_m[x]·rate_c·t)·wreg[r] (π_m absorbed in the partials)
    auto setDval=[&](double t){ std::vector<double> v0((size_t)R*ns),v1((size_t)R*ns),v2((size_t)R*ns);
        for(int m=0;m<nmix;m++) for(int c=0;c<ncat;c++){ int r=m*ncat+c; double rc=catRate[c], wr=wreg[r];
            for(int x=0;x<ns;x++){ double cof=evalC[(size_t)m*ns+x]*rc, e=exp(cof*t)*wr; v0[(size_t)r*ns+x]=e; v1[(size_t)r*ns+x]=cof*e; v2[(size_t)r*ns+x]=cof*cof*e; } }
        cudaMemcpy(d_v0,v0.data(),(size_t)R*ns*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(d_v1,v1.data(),(size_t)R*ns*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(d_v2,v2.data(),(size_t)R*ns*sizeof(double),cudaMemcpyHostToDevice); };

    // ---- PREORDER pool + proc recursion (the O(depth) slot pool is reset per chunk; proc rebuilt each tile) ----
    std::vector<int> freeSlots; bool poolUnderflow=false; int held=0, maxHeld=0;
    auto acq=[&]()->int{ if(freeSlots.empty()){ poolUnderflow=true; fprintf(stderr,"[ALLDERV-DBG] PREPOOL UNDERFLOW (nPool=%d treeH=%d)\n",nPool,treeH); return 0; } held++; if(held>maxHeld)maxHeld=held; int s=freeSlots.back();freeSlots.pop_back();return s;};
    auto rls=[&](int s){ held--; freeSlots.push_back(s);};
    std::function<void(int,int)> proc=[&](int u,int su){
        if(poolUnderflow) return (double)0;   // (double) to match GCK's return type in this lambda; value discarded by std::function<void>
        for(int v:child[u]){
            int sv=acq(); double* pre=d_prepool+(size_t)sv*slotSz;
            if(u==root){   // root child: upper partial = lower partial of root EXCLUDING v (k1_node_mix isRoot=0; NO π/wreg)
                int nch; const double* ec[3]; const double* p[3]; const unsigned char* t[3]; fillChild(root,v,nch,ec,p,t);
                k1_node_mix<<<dim3(GB,R),TB>>>(ns,Pn,ncat,nmix,/*isRoot=*/0,pre,d_patlh,d_Uinv,d_Urs,d_Freq,d_Wreg,nullptr,/*pinv=*/0.0,/*clsinv=*/nullptr,nch,
                    ec[0],p[0],t[0], ec[1],p[1],t[1], ec[2],p[2],t[2]);
            } else {       // internal parent: propagate pre_u through u with siblings-of-v and the parent branch b_u
                const double* ec[2]={0,0}; const double* sp[2]={0,0}; const unsigned char* st[2]={0,0}; int nsb=0;
                for(int w:child[u]){ if(w==v||nsb>=2) continue; ec[nsb]=d_echild+(size_t)w*ecStride;
                    if(leaf[w]>=0) st[nsb]=d_tip+(size_t)leaf[w]*Pn; else sp[nsb]=d_partial+(size_t)slot[w]*slotSz; nsb++; }
                k7_pre_mix<<<dim3(GB,R),TB>>>(ns,Pn,ncat,nmix,pre,d_prepool+(size_t)su*slotSz,d_expfac+(size_t)u*exStride,d_U,d_Uinv,d_Urs,nsb,
                    ec[0],sp[0],st[0], ec[1],sp[1],st[1]);
            }
            GCK(cudaDeviceSynchronize());
            const double* plv=edgeNodePtr(v); GCK(cudaDeviceSynchronize());
            setDval(brlen[v]);
            // A1 (+I): the invariant is branch-independent => enters ONLY the 1/Lp denominator. Route to k2_derv_mix_inv
            // (Lp = variable + pinv·base_invar_comb) when pinv>0; else the unchanged k2_derv_mix (byte-identical).
            if (d_baseinvar) k2_derv_mix_inv<<<GB,TB>>>(ns,Pn,R,plv,pre,d_v0,d_v1,d_v2,pinv,d_baseinvar,d_pdf,d_pddf,d_patlh);
            else             k2_derv_mix<<<GB,TB>>>(ns,Pn,R,plv,pre,d_v0,d_v1,d_v2,d_pdf,d_pddf,d_patlh);
            GCK(cudaDeviceSynchronize());
            reduceInto(v);
            if(leaf[v]<0) proc(v,sv); rls(sv);   // recurse THEN release (v's pre must stay live for its children)
        }
        return (double)0;   // EVERY control path must return: GCK injects `return (double)NAN`, so this lambda's
                            // deduced return type is double — falling off the end (normal loop exit) was UB and
                            // crashed the first frame to complete its loop (the deepest leaf-only parent).
    };

    // ---- TILE LOOP: full postorder + proc per chunk; per-edge Kahan accumulators carry across chunks ----
    for (int tchunk=0; tchunk<nTile; tchunk++){
        pOff=tchunk*chunk0; int p1=pOff+chunk0; if(p1>nptn)p1=nptn; Pn=p1-pOff; if(Pn<=0) break;
        slotSz=(size_t)R*ns*Pn; GB=(Pn+TB-1)/TB;
        for(int a=0;a<ntax;a++) memcpy(&tipChunk[(size_t)a*Pn], tip+(size_t)a*nptn+pOff, (size_t)Pn);   // tip stride = chunk width
        GCK(cudaMemcpy(d_tip,tipChunk.data(),(size_t)ntax*Pn,cudaMemcpyHostToDevice));
        if (d_baseinvar) GCK(cudaMemcpy(d_baseinvar, base_invar_comb+pOff, (size_t)Pn*sizeof(double), cudaMemcpyHostToDevice));   // A1 (+I): chunk slice (contiguous [pOff..pOff+Pn), kernel reads baseinvar[0..Pn-1])

        // POSTORDER: lower partial pl_u for every internal node (skip root: its lower partial is never used)
        for (int idx=0; idx<nInternal; idx++){ int u=postorder[idx]; if(u==root) continue;
            int nch; const double* ec[3]; const double* p[3]; const unsigned char* t[3]; fillChild(u,-1,nch,ec,p,t);
            k1_node_mix<<<dim3(GB,R),TB>>>(ns,Pn,ncat,nmix,/*isRoot=*/0,d_partial+(size_t)slot[u]*slotSz,d_patlh,d_Uinv,d_Urs,d_Freq,d_Wreg,/*out_lhcat=*/nullptr,/*pinv=*/0.0,/*clsinv=*/nullptr,nch,
                ec[0],p[0],t[0], ec[1],p[1],t[1], ec[2],p[2],t[2]); }
        GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError());

        // PREORDER: reset the O(depth) pool, then proc(root) does upper partials + per-edge reduceInto for this chunk
        freeSlots.clear(); for(int s=nPool-1;s>=0;s--) freeSlots.push_back(s); held=0;
        proc(root,-1);
        if(poolUnderflow) return (double)NAN;
        GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError());
    }
    if(getenv("ALLDERV_DBG")) fprintf(stderr,"[ALLDERV-DBG] tiled proc done (nTile=%d underflow=%d maxHeld=%d treeH=%d nInternal=%d)\n",nTile,(int)poolUnderflow,maxHeld,treeH,nInternal);
    for(int v=0;v<nnodes;v++){ out_df[v]=accDf[v]; out_ddf[v]=accDdf[v]; }
    return 0.0;
}

// G.2.1a — clean-room single-edge derivative launcher. The descriptor list covers BOTH subtrees split by the
// central edge (two sub-roots: the edge endpoints), every entry isRoot=0 so each internal node (incl. the two
// endpoints) writes its eigen-space partial to its slot. node_eig=d_partial[nodeSlot], dad_eig=d_partial[dadSlot]
// are the two endpoint partials EXCLUDING the central edge's transition (which is applied via val0=exp(eval·r·t)).
// Returns df = d(lnL)/dt = Σ_ptn ptn_freq·(d1/lh); *out_ddf = Σ_ptn ptn_freq·(d2/lh−(d1/lh)²); *out_lnL = the
// tree lnL at t (a free cross-check). NaN on CUDA error.
extern "C" double gpu_derv_crosscheck(
    int nstates, int nptn, int ncat, int ntax, int nnodes, int nInternal,
    const double* Uinv, const double* UinvRowSum, const double* freq, const double* catProp,
    const double* echild, const unsigned char* tip, const double* ptn_freq,
    const int* desc_isRoot, const int* desc_nchild, const int* desc_outSlot,
    const int* desc_childNode, const int* desc_childIsLeaf, const int* desc_childLeaf, const int* desc_childSlot,
    int nodeSlot, int nodeLeafTax, int dadSlot, int dadLeafTax,
    const double* eval, const double* catRate, double t,
    double* out_ddf, double* out_lnL)
{
    int ns = nstates;
    if (ns > NS_MAX || ncat > 64) { fprintf(stderr,"[GPU-DERV] unsupported ns=%d ncat=%d\n",ns,ncat); return (double)NAN; }
    (void)desc_isRoot;  // all entries are internal (isRoot=0) for the derivative sweep

    CC_TOSYM(0, g_Uinv, Uinv, sizeof(double)*ns*ns);
    CC_TOSYM(2, g_UinvRowSum, UinvRowSum, sizeof(double)*ns);
    CC_TOSYM(3, g_freq, freq, sizeof(double)*ns);
    CC_TOSYM(4, g_catw, catProp, sizeof(double)*ncat);

    // central-edge derivative coefficients val0/val1/val2 (host) -> __constant__
    std::vector<double> v0((size_t)ncat*ns), v1((size_t)ncat*ns), v2((size_t)ncat*ns);
    for (int c=0;c<ncat;c++){ double rc=catRate[c], pc=catProp[c];
        for (int x=0;x<ns;x++){ double re=rc*eval[x], e=exp(eval[x]*rc*t)*pc;
            v0[c*ns+x]=e; v1[c*ns+x]=re*e; v2[c*ns+x]=re*re*e; } }
    GCK(cudaMemcpyToSymbol(g_val0, v0.data(), sizeof(double)*ncat*ns));
    GCK(cudaMemcpyToSymbol(g_val1, v1.data(), sizeof(double)*ncat*ns));
    GCK(cudaMemcpyToSymbol(g_val2, v2.data(), sizeof(double)*ncat*ns));

    size_t ecStride=(size_t)ncat*ns*ns, slotSz=(size_t)ncat*ns*nptn;
    DEVB(gb_echild, (size_t)nnodes*ecStride*sizeof(double));
    DEVB(gb_tip,    (size_t)ntax*nptn);
    DEVB(gb_partial,(size_t)(nInternal>0?nInternal:1)*slotSz*sizeof(double));
    DEVB(gb_pdf,    (size_t)nptn*sizeof(double));
    DEVB(gb_pddf,   (size_t)nptn*sizeof(double));
    DEVB(gb_patlh,  (size_t)nptn*sizeof(double));
    double *d_echild=(double*)gb_echild.p, *d_partial=(double*)gb_partial.p;
    double *d_pdf=(double*)gb_pdf.p, *d_pddf=(double*)gb_pddf.p, *d_patlh=(double*)gb_patlh.p;
    unsigned char *d_tip=(unsigned char*)gb_tip.p;
    GCK(cudaMemcpy(d_echild,echild,(size_t)nnodes*ecStride*sizeof(double),cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(d_tip,tip,(size_t)ntax*nptn,cudaMemcpyHostToDevice));

    int TB=256, GB=(nptn+TB-1)/TB;
    for (int idx=0; idx<nInternal; idx++){
        int nchild=desc_nchild[idx];
        double* out=(desc_outSlot[idx]<0)?nullptr:(d_partial+(size_t)desc_outSlot[idx]*slotSz);
        const double* ec[3]={nullptr,nullptr,nullptr};
        const double* p[3]={nullptr,nullptr,nullptr};
        const unsigned char* tp[3]={nullptr,nullptr,nullptr};
        for (int k=0;k<nchild && k<3;k++){ int cn=desc_childNode[idx*3+k];
            if (cn>=0) ec[k]=d_echild+(size_t)cn*ecStride;
            if (desc_childIsLeaf[idx*3+k]) tp[k]=d_tip+(size_t)desc_childLeaf[idx*3+k]*nptn;
            else                          p[k]=d_partial+(size_t)desc_childSlot[idx*3+k]*slotSz; }
        launch_k1_node(GB,TB,0,ns,nptn,ncat,/*isRoot=*/0,out,d_patlh,nchild,
            ec[0],p[0],tp[0], ec[1],p[1],tp[1], ec[2],p[2],tp[2]);
    }
    GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError());

    // resolve node_eig / dad_eig: internal endpoint -> its slot; leaf endpoint -> synthesize the tip eigen
    // partial into a scratch slot (k_leaf_eig). Exactly one of (slot>=0, leafTax>=0) holds per endpoint.
    const double *node_eig, *dad_eig;
    if (nodeSlot >= 0) node_eig = d_partial + (size_t)nodeSlot*slotSz;
    else { DEVB(gb_nodeleaf, slotSz*sizeof(double));
           k_leaf_eig<<<GB,TB>>>(ns,nptn,ncat,d_tip+(size_t)nodeLeafTax*nptn,(double*)gb_nodeleaf.p); node_eig=(double*)gb_nodeleaf.p; }
    if (dadSlot >= 0) dad_eig = d_partial + (size_t)dadSlot*slotSz;
    else { DEVB(gb_dadleaf, slotSz*sizeof(double));
           k_leaf_eig<<<GB,TB>>>(ns,nptn,ncat,d_tip+(size_t)dadLeafTax*nptn,(double*)gb_dadleaf.p); dad_eig=(double*)gb_dadleaf.p; }

    k2_derv<<<GB,TB>>>(ns,nptn,ncat,node_eig,dad_eig,d_pdf,d_pddf,d_patlh);
    GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError());

    std::vector<double> pdf(nptn),pddf(nptn),patlh(nptn);
    GCK(cudaMemcpy(pdf.data(),d_pdf,(size_t)nptn*sizeof(double),cudaMemcpyDeviceToHost));
    GCK(cudaMemcpy(pddf.data(),d_pddf,(size_t)nptn*sizeof(double),cudaMemcpyDeviceToHost));
    GCK(cudaMemcpy(patlh.data(),d_patlh,(size_t)nptn*sizeof(double),cudaMemcpyDeviceToHost));

    // ptn_freq-weighted Kahan sums: df, ddf, lnL
    double df=0,kdf=0, ddf=0,kddf=0, lnL=0,kl=0;
    for (int p2=0;p2<nptn;p2++){ double f=ptn_freq[p2];
        { double term=f*pdf[p2],  y=term-kdf,  s=df +y; kdf =(s-df )-y; df =s; }
        { double term=f*pddf[p2], y=term-kddf, s=ddf+y; kddf=(s-ddf)-y; ddf=s; }
        { double term=f*patlh[p2],y=term-kl,   s=lnL+y; kl  =(s-lnL)-y; lnL=s; } }

    // pool buffers persist (no cudaFree) — reused next call
    if (out_ddf) *out_ddf=ddf;
    if (out_lnL) *out_lnL=lnL;
    return df;
}

// ============================================================================================================
// TS.2 Increment 3a — gpu_screen_nni_fold_crosscheck: score an NNI-swapped topology @ OLD lengths from ONE
// resident postorder over the PHYSICAL (unswapped) tree + a RE-PAIRING FOLD — NO swap-aware DFS, NO new kernel.
// Clones gpu_derv_crosscheck's prologue + postorder VERBATIM (the descriptor arrays describe the physical
// two-sub-root tree, so d_partial holds every subtree's resident lower partial). Then, instead of reading
// nodeSlot/dadSlot (the UNSWAPPED endpoint partials), it RE-PAIRS the four surrounding subtrees via two
// k1_node(isRoot=0,nchild=2) folds: node1's swapped directed partial = fold(child n1a, n1b); node2's =
// fold(n2a, n2b). The swap is purely in the fold GROUPING — each child carries its own physical echild matrix
// (ecNode), whose length is UNCHANGED by the swap (a moved Neighbor keeps its length). k2_derv then combines
// the two re-paired endpoint partials across the central edge at the UNCHANGED length t -> the swapped lnL.
// The re-pairing math must reproduce gpuScreenNNICleanRoom (the trusted I2 oracle) to 1e-9.
//   Re-pairing descriptor per child: (ec = echild node index, slot>=0 internal | leaf>=0 tip; exactly one).
//   For an NNI on (node1,node2) swapping S1<->S2: n1a=S2@L2, n1b=Bn@Lb, n2a=S1@L1, n2b=Dn@Ld.
// ============================================================================================================
extern "C" double gpu_screen_nni_fold_crosscheck(
    int nstates, int nptn, int ncat, int ntax, int nnodes, int nInternal,
    const double* Uinv, const double* UinvRowSum, const double* freq, const double* catProp,
    const double* echild, const unsigned char* tip, const double* ptn_freq,
    const int* desc_isRoot, const int* desc_nchild, const int* desc_outSlot,
    const int* desc_childNode, const int* desc_childIsLeaf, const int* desc_childLeaf, const int* desc_childSlot,
    int n1a_ec, int n1a_slot, int n1a_leaf,   int n1b_ec, int n1b_slot, int n1b_leaf,
    int n2a_ec, int n2a_slot, int n2a_leaf,   int n2b_ec, int n2b_slot, int n2b_leaf,
    const double* eval, const double* catRate, double t,
    double* out_ddf, double* out_lnL)
{
    int ns = nstates;
    if (ns > NS_MAX || ncat > 64) { fprintf(stderr,"[GPU-FOLD] unsupported ns=%d ncat=%d\n",ns,ncat); return (double)NAN; }
    (void)desc_isRoot;

    CC_TOSYM(0, g_Uinv, Uinv, sizeof(double)*ns*ns);
    CC_TOSYM(2, g_UinvRowSum, UinvRowSum, sizeof(double)*ns);
    CC_TOSYM(3, g_freq, freq, sizeof(double)*ns);
    CC_TOSYM(4, g_catw, catProp, sizeof(double)*ncat);

    // central-edge coeffs at the UNCHANGED length t (the swap never touches t) -> __constant__
    std::vector<double> v0((size_t)ncat*ns), v1((size_t)ncat*ns), v2((size_t)ncat*ns);
    for (int c=0;c<ncat;c++){ double rc=catRate[c], pc=catProp[c];
        for (int x=0;x<ns;x++){ double re=rc*eval[x], e=exp(eval[x]*rc*t)*pc;
            v0[c*ns+x]=e; v1[c*ns+x]=re*e; v2[c*ns+x]=re*re*e; } }
    GCK(cudaMemcpyToSymbol(g_val0, v0.data(), sizeof(double)*ncat*ns));
    GCK(cudaMemcpyToSymbol(g_val1, v1.data(), sizeof(double)*ncat*ns));
    GCK(cudaMemcpyToSymbol(g_val2, v2.data(), sizeof(double)*ncat*ns));

    size_t ecStride=(size_t)ncat*ns*ns, slotSz=(size_t)ncat*ns*nptn;
    DEVB(gb_echild, (size_t)nnodes*ecStride*sizeof(double));
    DEVB(gb_tip,    (size_t)ntax*nptn);
    DEVB(gb_partial,(size_t)(nInternal>0?nInternal:1)*slotSz*sizeof(double));
    DEVB(gb_pdf,    (size_t)nptn*sizeof(double));
    DEVB(gb_pddf,   (size_t)nptn*sizeof(double));
    DEVB(gb_patlh,  (size_t)nptn*sizeof(double));
    DEVB(gb_n1eig,  slotSz*sizeof(double));
    DEVB(gb_n2eig,  slotSz*sizeof(double));
    double *d_echild=(double*)gb_echild.p, *d_partial=(double*)gb_partial.p;
    double *d_pdf=(double*)gb_pdf.p, *d_pddf=(double*)gb_pddf.p, *d_patlh=(double*)gb_patlh.p;
    double *d_n1eig=(double*)gb_n1eig.p, *d_n2eig=(double*)gb_n2eig.p;
    unsigned char *d_tip=(unsigned char*)gb_tip.p;
    GCK(cudaMemcpy(d_echild,echild,(size_t)nnodes*ecStride*sizeof(double),cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(d_tip,tip,(size_t)ntax*nptn,cudaMemcpyHostToDevice));

    int TB=256, GB=(nptn+TB-1)/TB;
    // ---- resident postorder over the PHYSICAL tree (verbatim from gpu_derv_crosscheck) ----
    for (int idx=0; idx<nInternal; idx++){
        int nchild=desc_nchild[idx];
        double* out=(desc_outSlot[idx]<0)?nullptr:(d_partial+(size_t)desc_outSlot[idx]*slotSz);
        const double* ec[3]={nullptr,nullptr,nullptr};
        const double* p[3]={nullptr,nullptr,nullptr};
        const unsigned char* tp[3]={nullptr,nullptr,nullptr};
        for (int k=0;k<nchild && k<3;k++){ int cn=desc_childNode[idx*3+k];
            if (cn>=0) ec[k]=d_echild+(size_t)cn*ecStride;
            if (desc_childIsLeaf[idx*3+k]) tp[k]=d_tip+(size_t)desc_childLeaf[idx*3+k]*nptn;
            else                          p[k]=d_partial+(size_t)desc_childSlot[idx*3+k]*slotSz; }
        launch_k1_node(GB,TB,0,ns,nptn,ncat,/*isRoot=*/0,out,d_patlh,nchild,
            ec[0],p[0],tp[0], ec[1],p[1],tp[1], ec[2],p[2],tp[2]);
    }
    GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError());

    // ---- RE-PAIRING FOLDS: node1/node2 swapped directed eigen partials (the only new step) ----
    // each child resolves to (ec = echild matrix, p = resident slot | t = tip); exactly one of p/t per child.
    auto ecP = [&](int ecn){ return (ecn>=0)? (const double*)(d_echild+(size_t)ecn*ecStride) : (const double*)nullptr; };
    auto plP = [&](int slot){ return (slot>=0)? (const double*)(d_partial+(size_t)slot*slotSz) : (const double*)nullptr; };
    auto tpP = [&](int leaf){ return (leaf>=0)? (const unsigned char*)(d_tip+(size_t)leaf*nptn) : (const unsigned char*)nullptr; };
    launch_k1_node(GB,TB,0,ns,nptn,ncat,/*isRoot=*/0,d_n1eig,d_patlh,/*nchild=*/2,
        ecP(n1a_ec),plP(n1a_slot),tpP(n1a_leaf), ecP(n1b_ec),plP(n1b_slot),tpP(n1b_leaf), nullptr,nullptr,nullptr);
    launch_k1_node(GB,TB,0,ns,nptn,ncat,/*isRoot=*/0,d_n2eig,d_patlh,/*nchild=*/2,
        ecP(n2a_ec),plP(n2a_slot),tpP(n2a_leaf), ecP(n2b_ec),plP(n2b_slot),tpP(n2b_leaf), nullptr,nullptr,nullptr);
    GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError());

    k2_derv<<<GB,TB>>>(ns,nptn,ncat,d_n1eig,d_n2eig,d_pdf,d_pddf,d_patlh);
    GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError());

    std::vector<double> pdf(nptn),pddf(nptn),patlh(nptn);
    GCK(cudaMemcpy(pdf.data(),d_pdf,(size_t)nptn*sizeof(double),cudaMemcpyDeviceToHost));
    GCK(cudaMemcpy(pddf.data(),d_pddf,(size_t)nptn*sizeof(double),cudaMemcpyDeviceToHost));
    GCK(cudaMemcpy(patlh.data(),d_patlh,(size_t)nptn*sizeof(double),cudaMemcpyDeviceToHost));

    double df=0,kdf=0, ddf=0,kddf=0, lnL=0,kl=0;
    for (int p2=0;p2<nptn;p2++){ double f=ptn_freq[p2];
        { double term=f*pdf[p2],  y=term-kdf,  s=df +y; kdf =(s-df )-y; df =s; }
        { double term=f*pddf[p2], y=term-kddf, s=ddf+y; kddf=(s-ddf)-y; ddf=s; }
        { double term=f*patlh[p2],y=term-kl,   s=lnL+y; kl  =(s-lnL)-y; lnL=s; } }

    if (out_ddf) *out_ddf=ddf;
    if (out_lnL) *out_lnL=lnL;
    return df;
}

// ============================================================================================================
// TS.2 Increment 3b-i — gpu_allbranch_upper_check: ONE fixed-root postorder (resident lowers) + ONE preorder with
// a PERSISTENT per-node upper buffer (d_upper[v], slot = node id — NOT the O(depth) acq/rls pool), then for EVERY
// internal edge (u=parent, v=child) run k2_derv(lower_v, pre_v, t=b_v) and return its lnL. The whole-tree lnL is
// also returned (independent root isRoot=1 reduction). The INVARIANT: every edge's k2_derv lnL == the tree lnL
// (a reversible model's lnL is the contraction of lower_v ⊗ pre_v at the true length b_v, edge-invariant). This
// validates the persistent-upper machinery in its UNSWAPPED configuration — the substrate 3b-ii's re-pairing reuses
// (kj_pre with a swapped sibling) — WITHOUT any re-pairing. Single-model only; nTile=1 (pattern tiling = 3c).
// Reuses k1_node / kj_pre / k2_derv verbatim; NO new kernel. The only new thing = the persistent (not pooled) upper.
//   pre_v: root-child v -> k1_node(isRoot=0) over root's OTHER children (no parent branch); interior v ->
//          kj_pre(pre_u=d_upper[u], expfac_u, siblings = child[u]\{v}). Stored persistently at d_upper[v].
// ============================================================================================================
extern "C" double gpu_allbranch_upper_check(
    int nstates, int nptn, int ncat, int ntax, int nnodes, int nInternal, int root,
    const double* Uinv, const double* U, const double* UinvRowSum, const double* freq, const double* catProp,
    const double* eval, const double* catRate,
    const double* echild, const double* expfac, const unsigned char* tip, const double* ptn_freq,
    const int* node_nchild, const int* node_child, const int* node_leaf, const int* node_slot,
    const double* node_parentLen, const int* post_internal,
    double* out_edge_lnL,   // [nnodes]: per-edge lnL via k2_derv(lower_v,pre_v,b_v); non-edge/root entries left as set by caller
    double* out_tree_lnL)   // whole-tree lnL (independent root isRoot=1 reduction)
{
    int ns = nstates;
    if (ns > NS_MAX || ncat > 64) { fprintf(stderr,"[GPU-UPPER] unsupported ns=%d ncat=%d\n",ns,ncat); return (double)NAN; }
    int TB=256, GB=(nptn+TB-1)/TB, Pn=nptn;

    CC_TOSYM(0, g_Uinv, Uinv, sizeof(double)*ns*ns);
    CC_TOSYM(1, g_U,    U,    sizeof(double)*ns*ns);   // kj_pre needs the eigenvectors (up-map)
    CC_TOSYM(2, g_UinvRowSum, UinvRowSum, sizeof(double)*ns);
    CC_TOSYM(3, g_freq, freq, sizeof(double)*ns);
    CC_TOSYM(4, g_catw, catProp, sizeof(double)*ncat);

    size_t ecStride=(size_t)ncat*ns*ns, exStride=(size_t)ncat*ns, slotSz=(size_t)ncat*ns*nptn;
    DEVB(gb_echild, (size_t)nnodes*ecStride*sizeof(double));
    DEVB(gb_uexpfac,(size_t)nnodes*exStride*sizeof(double));
    DEVB(gb_tip,    (size_t)ntax*nptn);
    DEVB(gb_partial,(size_t)(nInternal>0?nInternal:1)*slotSz*sizeof(double));
    DEVB(gb_upper,  (size_t)nnodes*slotSz*sizeof(double));   // PERSISTENT: one upper slot per node
    DEVB(gb_pdf,    (size_t)nptn*sizeof(double));
    DEVB(gb_pddf,   (size_t)nptn*sizeof(double));
    DEVB(gb_patlh,  (size_t)nptn*sizeof(double));
    DEVB(gb_nodeleaf, slotSz*sizeof(double));   // leaf endpoint lower-eigen scratch (k_leaf_eig)
    double *d_echild=(double*)gb_echild.p, *d_expfac=(double*)gb_uexpfac.p, *d_partial=(double*)gb_partial.p;
    double *d_upper=(double*)gb_upper.p, *d_pdf=(double*)gb_pdf.p, *d_pddf=(double*)gb_pddf.p, *d_patlh=(double*)gb_patlh.p;
    double *d_tipeig=(double*)gb_nodeleaf.p;
    unsigned char *d_tip=(unsigned char*)gb_tip.p;
    GCK(cudaMemcpy(d_echild,echild,(size_t)nnodes*ecStride*sizeof(double),cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(d_expfac,expfac,(size_t)nnodes*exStride*sizeof(double),cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(d_tip,tip,(size_t)ntax*nptn,cudaMemcpyHostToDevice));

    // child-args helper (exclude `excl`, -1 for none): echild/partial/tip pointers for k1_node
    auto fillChild=[&](int u,int excl,int& nch,const double** ec,const double** p,const unsigned char** t){
        nch=0; for(int k=0;k<3;k++){ec[k]=nullptr;p[k]=nullptr;t[k]=nullptr;}
        for(int kk=0; kk<node_nchild[u]; kk++){ int c=node_child[u*3+kk]; if(c==excl||nch>=3) continue;
            ec[nch]=d_echild+(size_t)c*ecStride;
            if(node_leaf[c]>=0) t[nch]=d_tip+(size_t)node_leaf[c]*Pn; else p[nch]=d_partial+(size_t)node_slot[c]*slotSz; nch++; } };
    auto edgeNodePtr=[&](int v)->const double*{
        if(node_leaf[v]<0) return d_partial+(size_t)node_slot[v]*slotSz;
        k_leaf_eig<<<GB,TB>>>(ns,Pn,ncat,d_tip+(size_t)node_leaf[v]*Pn,d_tipeig); return d_tipeig; };
    auto setVal=[&](double t){ std::vector<double> v0(ncat*ns),v1(ncat*ns),v2(ncat*ns);
        for(int c=0;c<ncat;c++){ double rc=catRate[c], pc=catProp[c];
            for(int x=0;x<ns;x++){ double re=rc*eval[x], e=exp(eval[x]*rc*t)*pc; v0[c*ns+x]=e; v1[c*ns+x]=re*e; v2[c*ns+x]=re*re*e; } }
        cudaMemcpyToSymbol(g_val0,v0.data(),sizeof(double)*ncat*ns); cudaMemcpyToSymbol(g_val1,v1.data(),sizeof(double)*ncat*ns); cudaMemcpyToSymbol(g_val2,v2.data(),sizeof(double)*ncat*ns); };
    // ptn_freq-weighted Kahan reduce of the patlh channel (after a k2_derv) -> edge lnL
    auto reduceLnL=[&]()->double{ std::vector<double> pl(nptn); cudaMemcpy(pl.data(),d_patlh,(size_t)nptn*sizeof(double),cudaMemcpyDeviceToHost);
        double L=0,k=0; for(int p=0;p<nptn;p++){ double term=ptn_freq[p]*pl[p], y=term-k, s=L+y; k=(s-L)-y; L=s; } return L; };

    // ---- POSTORDER: resident lower partials (skip root) ----
    for (int idx=0; idx<nInternal; idx++){ int u=post_internal[idx]; if(u==root) continue;
        int nch; const double* ec[3]; const double* p[3]; const unsigned char* t[3]; fillChild(u,-1,nch,ec,p,t);
        launch_k1_node(GB,TB,0,ns,Pn,ncat,/*isRoot=*/0,d_partial+(size_t)node_slot[u]*slotSz,d_patlh,nch,
            ec[0],p[0],t[0], ec[1],p[1],t[1], ec[2],p[2],t[2]); }
    GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError());

    // ---- whole-tree lnL: root isRoot=1 fold over root's children (independent reference) ----
    { int nch; const double* ec[3]; const double* p[3]; const unsigned char* t[3]; fillChild(root,-1,nch,ec,p,t);
      launch_k1_node(GB,TB,0,ns,Pn,ncat,/*isRoot=*/1,/*out=*/nullptr,d_patlh,nch,
          ec[0],p[0],t[0], ec[1],p[1],t[1], ec[2],p[2],t[2]);
      GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError()); }
    double tree_lnL = reduceLnL();
    if (out_tree_lnL) *out_tree_lnL = tree_lnL;

    // ---- PREORDER: persistent per-node upper d_upper[v] + per-edge k2_derv lnL ----
    // proc returns double (NOT void) so GCK's injected `return (double)NAN` typechecks; std::function<void(int)>
    // discards it. EVERY path must return (GCK makes the deduced type double; falling off the end would be UB).
    std::function<void(int)> proc=[&](int u)->double{
        for(int kk=0; kk<node_nchild[u]; kk++){
            int v=node_child[u*3+kk];
            double* pre=d_upper+(size_t)v*slotSz;
            if(u==root){   // root child: upper = lower partial of root EXCLUDING v (k1_node isRoot=0; no parent branch)
                int nch; const double* ec[3]; const double* p[3]; const unsigned char* t[3]; fillChild(root,v,nch,ec,p,t);
                launch_k1_node(GB,TB,0,ns,Pn,ncat,/*isRoot=*/0,pre,d_patlh,nch, ec[0],p[0],t[0], ec[1],p[1],t[1], ec[2],p[2],t[2]);
            } else {       // interior parent: propagate pre_u through u with siblings-of-v and the parent branch b_u
                const double* ec[2]={0,0}; const double* sp[2]={0,0}; const unsigned char* st[2]={0,0}; int nsb=0;
                for(int jj=0; jj<node_nchild[u]; jj++){ int w=node_child[u*3+jj]; if(w==v||nsb>=2) continue;
                    ec[nsb]=d_echild+(size_t)w*ecStride;
                    if(node_leaf[w]>=0) st[nsb]=d_tip+(size_t)node_leaf[w]*Pn; else sp[nsb]=d_partial+(size_t)node_slot[w]*slotSz; nsb++; }
                launch_kj_pre(GB,TB,0,ns,Pn,ncat,pre,d_upper+(size_t)u*slotSz,d_expfac+(size_t)u*exStride,nsb,
                    ec[0],sp[0],st[0], ec[1],sp[1],st[1]);
            }
            GCK(cudaDeviceSynchronize());
            const double* plv=edgeNodePtr(v); GCK(cudaDeviceSynchronize());
            setVal(node_parentLen[v]);
            k2_derv<<<GB,TB>>>(ns,Pn,ncat,plv,pre,d_pdf,d_pddf,d_patlh); GCK(cudaDeviceSynchronize());
            if(out_edge_lnL) out_edge_lnL[v] = reduceLnL();
            if(node_leaf[v]<0) proc(v);   // recurse (d_upper[v] persists for v's children)
        }
        return (double)0;   // normal exit (see note above)
    };
    proc(root);
    GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError());
    return tree_lnL;
}

// ============================================================================================================
// TS.2 Increment 3b-ii — gpu_screen_nni_batch_crosscheck: the BATCHED re-pairing NNI screener (the perf core).
// Clones gpu_allbranch_upper_check's prologue + ONE fixed-root postorder (resident lowers d_partial) + ONE
// persistent-upper preorder (d_upper[v], slot = node id) — BUILT ONCE — then scores a host-provided list of NNI
// moves, each a CHEAP fold reading the resident lowers + persistent uppers (NO re-sweep). Per move (parent u,
// child v, swapping u's other child w with a v-child):
//   node1Eig (u side, toward v) = kj_pre(pre_u=d_upper[u], expfac_u, sibling = swapped-in v-child)   [u != root]
//                               = k1_node({swapped-in v-child, u's staying child})                   [u == root: no upper/expfac]
//   node2Eig (v side, toward u) = k1_node({w, v's staying child})
//   lnL = k2_derv(node1Eig, node2Eig, b_v)   (central length unchanged)
// The ONLY new work vs gpu_allbranch_upper_check is the move loop; NO new kernel (k1_node/kj_pre/k2_derv reused).
// out_move_lnL[m] = swapped-topology lnL, cross-checked per move vs gpuScreenNNIFoldCleanRoom (the 3a oracle).
// The win: 1 postorder shared across ALL moves vs the per-move oracle's M postorders. (nTile=1; AA-scale = 3c.)
// ============================================================================================================
extern "C" double gpu_screen_nni_batch_crosscheck(
    int nstates, int nptn, int ncat, int ntax, int nnodes, int nInternal, int root,
    const double* Uinv, const double* U, const double* UinvRowSum, const double* freq, const double* catProp,
    const double* eval, const double* catRate,
    const double* echild, const double* expfac, const unsigned char* tip, const double* ptn_freq,
    const int* node_nchild, const int* node_child, const int* node_leaf, const int* node_slot,
    const double* node_parentLen, const int* post_internal,
    int nMoves,
    const int* mv_u, const int* mv_uIsRoot, const double* mv_bv,
    const int* n1a_ec, const int* n1a_slot, const int* n1a_leaf,   // node1 fold child A = swapped-in v-child (kj_pre sibling / k1_node child)
    const int* n1b_ec, const int* n1b_slot, const int* n1b_leaf,   // node1 fold child B = u's staying child (u==root k1_node ONLY)
    const int* n2a_ec, const int* n2a_slot, const int* n2a_leaf,   // node2 fold child A = w (u's moved-out child)
    const int* n2b_ec, const int* n2b_slot, const int* n2b_leaf,   // node2 fold child B = v's staying child
    double* out_move_lnL, double* out_tree_lnL)
{
    int ns = nstates;
    if (ns > NS_MAX || ncat > 64) { fprintf(stderr,"[GPU-BATCH] unsupported ns=%d ncat=%d\n",ns,ncat); return (double)NAN; }
    int TB=256, GB=(nptn+TB-1)/TB, Pn=nptn;

    CC_TOSYM(0, g_Uinv, Uinv, sizeof(double)*ns*ns);
    CC_TOSYM(1, g_U,    U,    sizeof(double)*ns*ns);
    CC_TOSYM(2, g_UinvRowSum, UinvRowSum, sizeof(double)*ns);
    CC_TOSYM(3, g_freq, freq, sizeof(double)*ns);
    CC_TOSYM(4, g_catw, catProp, sizeof(double)*ncat);

    size_t ecStride=(size_t)ncat*ns*ns, exStride=(size_t)ncat*ns, slotSz=(size_t)ncat*ns*nptn;
    DEVB(gb_echild, (size_t)nnodes*ecStride*sizeof(double));
    DEVB(gb_uexpfac,(size_t)nnodes*exStride*sizeof(double));
    DEVB(gb_tip,    (size_t)ntax*nptn);
    DEVB(gb_partial,(size_t)(nInternal>0?nInternal:1)*slotSz*sizeof(double));
    DEVB(gb_upper,  (size_t)nnodes*slotSz*sizeof(double));
    DEVB(gb_pmat,   (size_t)nnodes*ecStride*sizeof(double));   // node-space P(b)=echild·Uinv (F39/F40 node-space upper)
    DEVB(gb_pdf,    (size_t)nptn*sizeof(double));
    DEVB(gb_pddf,   (size_t)nptn*sizeof(double));
    DEVB(gb_patlh,  (size_t)nptn*sizeof(double));
    DEVB(gb_nodeleaf, slotSz*sizeof(double));
    DEVB(gb_n1eig,  slotSz*sizeof(double));
    DEVB(gb_n2eig,  slotSz*sizeof(double));
    double *d_echild=(double*)gb_echild.p, *d_expfac=(double*)gb_uexpfac.p, *d_partial=(double*)gb_partial.p;
    double *d_upper=(double*)gb_upper.p, *d_pdf=(double*)gb_pdf.p, *d_pddf=(double*)gb_pddf.p, *d_patlh=(double*)gb_patlh.p;
    double *d_tipeig=(double*)gb_nodeleaf.p, *d_n1eig=(double*)gb_n1eig.p, *d_n2eig=(double*)gb_n2eig.p;
    double *d_pmat=(double*)gb_pmat.p;
    unsigned char *d_tip=(unsigned char*)gb_tip.p;
    GCK(cudaMemcpy(d_echild,echild,(size_t)nnodes*ecStride*sizeof(double),cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(d_expfac,expfac,(size_t)nnodes*exStride*sizeof(double),cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(d_tip,tip,(size_t)ntax*nptn,cudaMemcpyHostToDevice));
    { int rows=nnodes*ncat*ns, gbp=(rows+TB-1)/TB; make_pmat<<<gbp,TB>>>(ns,ncat,nnodes,d_echild,d_pmat);   // P(b) per node·cat
      GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError()); }

    auto fillChild=[&](int u,int excl,int& nch,const double** ec,const double** p,const unsigned char** t){
        nch=0; for(int k=0;k<3;k++){ec[k]=nullptr;p[k]=nullptr;t[k]=nullptr;}
        for(int kk=0; kk<node_nchild[u]; kk++){ int c=node_child[u*3+kk]; if(c==excl||nch>=3) continue;
            ec[nch]=d_echild+(size_t)c*ecStride;
            if(node_leaf[c]>=0) t[nch]=d_tip+(size_t)node_leaf[c]*Pn; else p[nch]=d_partial+(size_t)node_slot[c]*slotSz; nch++; } };
    auto setVal=[&](double t){ std::vector<double> v0(ncat*ns),v1(ncat*ns),v2(ncat*ns);
        for(int c=0;c<ncat;c++){ double rc=catRate[c], pc=catProp[c];
            for(int x=0;x<ns;x++){ double re=rc*eval[x], e=exp(eval[x]*rc*t)*pc; v0[c*ns+x]=e; v1[c*ns+x]=re*e; v2[c*ns+x]=re*re*e; } }
        cudaMemcpyToSymbol(g_val0,v0.data(),sizeof(double)*ncat*ns); cudaMemcpyToSymbol(g_val1,v1.data(),sizeof(double)*ncat*ns); cudaMemcpyToSymbol(g_val2,v2.data(),sizeof(double)*ncat*ns); };
    auto reduceLnL=[&]()->double{ std::vector<double> pl(nptn); cudaMemcpy(pl.data(),d_patlh,(size_t)nptn*sizeof(double),cudaMemcpyDeviceToHost);
        double L=0,k=0; for(int p=0;p<nptn;p++){ double term=ptn_freq[p]*pl[p], y=term-k, s=L+y; k=(s-L)-y; L=s; } return L; };
    // resolve a re-pairing child's (echild, partial|tip) pointers
    auto ecP=[&](int ecn){ return (ecn>=0)? (const double*)(d_echild+(size_t)ecn*ecStride) : (const double*)nullptr; };
    auto plP=[&](int slot){ return (slot>=0)? (const double*)(d_partial+(size_t)slot*slotSz) : (const double*)nullptr; };
    auto tpP=[&](int leaf){ return (leaf>=0)? (const unsigned char*)(d_tip+(size_t)leaf*Pn) : (const unsigned char*)nullptr; };

    // ---- POSTORDER (resident lowers) ----
    for (int idx=0; idx<nInternal; idx++){ int u=post_internal[idx]; if(u==root) continue;
        int nch; const double* ec[3]; const double* p[3]; const unsigned char* t[3]; fillChild(u,-1,nch,ec,p,t);
        launch_k1_node(GB,TB,0,ns,Pn,ncat,/*isRoot=*/0,d_partial+(size_t)node_slot[u]*slotSz,d_patlh,nch,
            ec[0],p[0],t[0], ec[1],p[1],t[1], ec[2],p[2],t[2]); }
    GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError());

    // ---- whole-tree lnL (independent; result-invariance) ----
    { int nch; const double* ec[3]; const double* p[3]; const unsigned char* t[3]; fillChild(root,-1,nch,ec,p,t);
      launch_k1_node(GB,TB,0,ns,Pn,ncat,/*isRoot=*/1,nullptr,d_patlh,nch, ec[0],p[0],t[0], ec[1],p[1],t[1], ec[2],p[2],t[2]);
      GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError()); }
    if (out_tree_lnL) *out_tree_lnL = reduceLnL();

    // ---- PREORDER: persistent per-node upper d_upper[v] stored NODE-space (F39/F40): seed root-children with
    // k1_node_prod (no final Uinv); interior up_v = (P(b_u)·up_u) ⊙ fsib via kj_pre_node(eigOut=0). No eigen round-trip. ----
    std::function<void(int)> proc=[&](int u)->double{
        for(int kk=0; kk<node_nchild[u]; kk++){
            int v=node_child[u*3+kk];
            double* pre=d_upper+(size_t)v*slotSz;
            if(u==root){
                int nch; const double* ec[3]; const double* p[3]; const unsigned char* t[3]; fillChild(root,v,nch,ec,p,t);
                k1_node_prod<<<GB,TB>>>(ns,Pn,ncat,pre,nch, ec[0],p[0],t[0], ec[1],p[1],t[1], ec[2],p[2],t[2]);
            } else {
                const double* ec[2]={0,0}; const double* sp[2]={0,0}; const unsigned char* st[2]={0,0}; int nsb=0;
                for(int jj=0; jj<node_nchild[u]; jj++){ int w=node_child[u*3+jj]; if(w==v||nsb>=2) continue;
                    ec[nsb]=d_echild+(size_t)w*ecStride;
                    if(node_leaf[w]>=0) st[nsb]=d_tip+(size_t)node_leaf[w]*Pn; else sp[nsb]=d_partial+(size_t)node_slot[w]*slotSz; nsb++; }
                launch_kj_pre_node(GB,TB,0,ns,Pn,ncat,/*eigOut=*/0,pre,d_upper+(size_t)u*slotSz,d_pmat+(size_t)u*ecStride,nsb,
                    ec[0],sp[0],st[0], ec[1],sp[1],st[1]);
            }
            GCK(cudaDeviceSynchronize());
            if(node_leaf[v]<0) proc(v);
        }
        return (double)0;
    };
    proc(root);
    GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError());

    // ---- MOVE LOOP: each move = 2 cheap folds + k2_derv off the RESIDENT lowers + PERSISTENT uppers ----
    for (int m=0; m<nMoves; m++){
        int u=mv_u[m];
        if (mv_uIsRoot[m]) {   // u==root: no upper/expfac -> k1_node({swapped-in v-child, u's staying child})
            launch_k1_node(GB,TB,0,ns,Pn,ncat,/*isRoot=*/0,d_n1eig,d_patlh,/*nchild=*/2,
                ecP(n1a_ec[m]),plP(n1a_slot[m]),tpP(n1a_leaf[m]), ecP(n1b_ec[m]),plP(n1b_slot[m]),tpP(n1b_leaf[m]), nullptr,nullptr,nullptr);
        } else {               // interior u: node1Eig = Uinv·((P(b_u)·up_u) ⊙ (P(b_swapchild)·L_swapchild)), eigOut=1
            launch_kj_pre_node(GB,TB,0,ns,Pn,ncat,/*eigOut=*/1,d_n1eig,d_upper+(size_t)u*slotSz,d_pmat+(size_t)u*ecStride,/*nsib=*/1,
                ecP(n1a_ec[m]),plP(n1a_slot[m]),tpP(n1a_leaf[m]), nullptr,nullptr,nullptr);
        }
        launch_k1_node(GB,TB,0,ns,Pn,ncat,/*isRoot=*/0,d_n2eig,d_patlh,/*nchild=*/2,
            ecP(n2a_ec[m]),plP(n2a_slot[m]),tpP(n2a_leaf[m]), ecP(n2b_ec[m]),plP(n2b_slot[m]),tpP(n2b_leaf[m]), nullptr,nullptr,nullptr);
        GCK(cudaDeviceSynchronize());
        setVal(mv_bv[m]);
        k2_derv<<<GB,TB>>>(ns,Pn,ncat,d_n1eig,d_n2eig,d_pdf,d_pddf,d_patlh); GCK(cudaDeviceSynchronize());
        if (out_move_lnL) out_move_lnL[m] = reduceLnL();
    }
    GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError());
    return out_tree_lnL ? *out_tree_lnL : 0.0;
}

// ============================================================================================================
// TS.2 Increment 3c — gpu_screen_nni_tile_crosscheck: the PATTERN-TILED batched re-pairing NNI screener.
// IDENTICAL math to gpu_screen_nni_batch_crosscheck (3b-ii) but nptn is split into nTile contiguous chunks so the
// PERSISTENT per-node upper (gb_upper = nnodes*ncat*ns*nptn — the OOM surface at AA-1M, ~640GB) is sized to chunk0
// instead of nptn. Per chunk: full postorder (resident lowers) + tree-lnL + persistent-upper preorder + the move
// loop, each scoring all M moves over THIS chunk's patterns. Each move's lnL is a CONTINUOUS per-move Kahan sum
// carried across chunks (add order 0..nptn-1) => BIT-IDENTICAL to nTile=1 (== the validated 3b-ii) for any nTile.
// nTile = forced_ntile>0 ? forced_ntile : mix_pick_ntile(nptn, perPtnDoubles); JOLT_NTILE env still overrides the
// auto path. out_ntile reports the chosen nTile. echild/expfac/eigen carry NO pattern axis (uploaded once); the
// move descriptors are pattern-INDEPENDENT (enumerated once, reused every chunk). NO new kernel.
// ============================================================================================================
// TS-KCOUNT (env TS_KCOUNT, default OFF => byte-identical): per-call-site partial-kernel LAUNCH counters to size
// the redundancy split — base postorder sweep (recomputed every round = shareable/redundant) vs per-move folds
// (one per distinct NNI move = NECESSARY) vs the upper sweep vs reopt. Counts LAUNCHES (grid invocations), not
// blocks. Printed per-call + cumulative at the two extern-C drivers. Read-only side counters => no math change.
static const bool g_kcount = (getenv("TS_KCOUNT") != nullptr);
static long g_kc_scr_base = 0, g_kc_scr_upper = 0, g_kc_scr_fold = 0, g_kc_scr_calls = 0;
static long g_kc_reopt_part = 0, g_kc_reopt_calls = 0;

extern "C" double gpu_screen_nni_tile_crosscheck(
    int nstates, int nptn, int ncat, int ntax, int nnodes, int nInternal, int root,
    const double* Uinv, const double* U, const double* UinvRowSum, const double* freq, const double* catProp,
    const double* eval, const double* catRate,
    const double* echild, const double* expfac, const unsigned char* tip, const double* ptn_freq,
    const int* node_nchild, const int* node_child, const int* node_leaf, const int* node_slot,
    const double* node_parentLen, const int* post_internal,
    int nMoves,
    const int* mv_u, const int* mv_uIsRoot, const double* mv_bv,
    const int* n1a_ec, const int* n1a_slot, const int* n1a_leaf,
    const int* n1b_ec, const int* n1b_slot, const int* n1b_leaf,
    const int* n2a_ec, const int* n2a_slot, const int* n2a_leaf,
    const int* n2b_ec, const int* n2b_slot, const int* n2b_leaf,
    int forced_ntile,
    double* out_move_lnL, double* out_tree_lnL, int* out_ntile,
    const double* baseinvar, double pinv)   // A3 (+I): per-pattern invariant base + pinv; pinv<=0 => no +I (bit-identical)
{
    int ns = nstates;
    if (ns > NS_MAX || ncat > 64) { fprintf(stderr,"[GPU-TILE] unsupported ns=%d ncat=%d\n",ns,ncat); return (double)NAN; }
    int TB=256;

    CC_TOSYM(0, g_Uinv, Uinv, sizeof(double)*ns*ns);
    CC_TOSYM(1, g_U,    U,    sizeof(double)*ns*ns);
    CC_TOSYM(2, g_UinvRowSum, UinvRowSum, sizeof(double)*ns);
    CC_TOSYM(3, g_freq, freq, sizeof(double)*ns);
    CC_TOSYM(4, g_catw, catProp, sizeof(double)*ncat);

    // ---- ASYNC (Inc 0): stream count S for the screener move loop. OFF => S=1 (default stream, identical alloc).
    //      ON => S=min(poolK,nMoves) private scratch slots round-robined across g_ts_streams. ts_streams() lazily
    //      creates the never-freed pool; if it fails we fall back to S=1 (serial, still correct). ----
    int S = 1; cudaStream_t* tsS = nullptr;
    if (g_ts_async && nMoves > 0) {
        int poolK = 0; tsS = ts_streams(&poolK);
        if (tsS && poolK >= 1) { S = (poolK < nMoves) ? poolK : nMoves; }
    }
    // ---- pick nTile: persistent upper (nnodes) + lowers (nInternal) + 5 R*ns scratch + 3 per-pattern scalars ----
    // ASYNC: the per-move move-loop scratch (n1eig+n2eig = 2*R*ns + pdf/pddf/patlh = 3 scalars, per pattern) is
    // replicated S× so each stream owns a private slot. Add the (S-1) EXTRA copies to perPtnDoubles so the tiling
    // still fits HBM. S=1 (OFF) adds 0 => perPtnDoubles is byte-identical to today.
    size_t perPtnDoubles = ((size_t)nnodes + (size_t)(nInternal>0?nInternal:1) + 5)*(size_t)ncat*ns + 3
                         + (size_t)(S-1)*(2*(size_t)ncat*ns + 3)
                         // TS RAKE BATCH: B-wide n1/n2 eigen slots (2*ncat*ns each) + B-wide patlh (1) per pattern (OFF => +0)
                         + (g_ts_batchfold ? (size_t)g_ts_batchfold_B*(2*(size_t)ncat*ns + 1) : 0);
    int nTile  = (forced_ntile>0) ? forced_ntile : mix_pick_ntile(nptn, perPtnDoubles);
    if (nTile<1) nTile=1; if (nTile>nptn) nTile=nptn;
    int chunk0 = (nptn + nTile - 1) / nTile;
    size_t ecStride=(size_t)ncat*ns*ns, exStride=(size_t)ncat*ns, slotSzMax=(size_t)ncat*ns*chunk0;
    if (getenv("JOLT_DEBUG")) fprintf(stderr,"[TILE] nptn=%d nTile=%d chunk0=%d perPtnDoubles=%zu upperGB(nt=1)=%.2f\n",
        nptn,nTile,chunk0,perPtnDoubles,(double)nnodes*ncat*ns*nptn*8/1.073741824e9);

    // ---- alloc ONCE at chunk0 max width (echild/expfac have NO pattern axis -> full nnodes-size, uploaded once) ----
    DEVB(gb_echild, (size_t)nnodes*ecStride*sizeof(double));
    DEVB(gb_uexpfac,(size_t)nnodes*exStride*sizeof(double));
    DEVB(gb_pmat,   (size_t)nnodes*ecStride*sizeof(double));   // node-space P(b)=echild·Uinv (same per-node stride as echild)
    DEVB(gb_tip,    (size_t)ntax*chunk0);
    DEVB(gb_partial,(size_t)(nInternal>0?nInternal:1)*slotSzMax*sizeof(double));
    DEVB(gb_upper,  (size_t)nnodes*slotSzMax*sizeof(double));
    // ASYNC (Inc 0): the 5 screener move-loop scratch buffers get S private slots (S=1 OFF => identical alloc).
    // Per-slot strides are UNCHANGED (chunk0 for pdf/pddf/patlh; slotSzMax=ncat*ns*chunk0 for n1eig/n2eig); slot s
    // lives at offset s*<stride>. Move m uses slot (m % S) on stream (m % S) => within a stream slot reuse is
    // serialized (k2 -> D2H -> next move's folds), so no cross-move scratch aliasing.
    DEVB(gb_pdf,    (size_t)S*chunk0*sizeof(double));
    DEVB(gb_pddf,   (size_t)S*chunk0*sizeof(double));
    DEVB(gb_patlh,  (size_t)S*chunk0*sizeof(double));
    DEVB(gb_nodeleaf, slotSzMax*sizeof(double));
    DEVB(gb_n1eig,  (size_t)S*slotSzMax*sizeof(double));
    DEVB(gb_n2eig,  (size_t)S*slotSzMax*sizeof(double));
    DEVB(gb_valall, (size_t)(nMoves>0?nMoves:1)*3*ncat*ns*sizeof(double));   // TS.2.1 K1: per-move {v0,v1,v2} tables (pattern-independent), uploaded ONCE
    DEVB(gb_baseinvar, (size_t)chunk0*sizeof(double));   // A3 (+I): per-pattern invariant base, uploaded per chunk
    // TS RAKE BATCH: B-wide per-move scratch (node1/node2 eigen slots + patlh row). Allocated only when ON; a 1-double
    // stub keeps DEVB happy when OFF (no HBM cost, never read). B = g_ts_batchfold_B.
    DEVB(gb_n1batch, (g_ts_batchfold ? (size_t)g_ts_batchfold_B*slotSzMax : 1)*sizeof(double));
    DEVB(gb_n2batch, (g_ts_batchfold ? (size_t)g_ts_batchfold_B*slotSzMax : 1)*sizeof(double));
    DEVB(gb_patlhbatch, (g_ts_batchfold ? (size_t)g_ts_batchfold_B*chunk0 : 1)*sizeof(double));
    double *d_n1batch=(double*)gb_n1batch.p, *d_n2batch=(double*)gb_n2batch.p, *d_patlhbatch=(double*)gb_patlhbatch.p;
    std::vector<double> plbatch((g_ts_batchfold ? (size_t)g_ts_batchfold_B*chunk0 : 1));
    // TS RAKE BATCH: the move-descriptor arrays (mv_u, n1a_ec, ... ) are HOST pointers (the serial loop reads them on
    // the host); upload the 14 of them ONCE into a packed device buffer [14][nMoves] so the batched kernels can index
    // them by blockIdx.y on the DEVICE. Pattern-chunk-INDEPENDENT => uploaded once, before the tile loop.
    DEVB(gb_mvdesc, ((g_ts_batchfold && nMoves>0) ? (size_t)14*nMoves : 1)*sizeof(int));
    int* d_mvdesc=(int*)gb_mvdesc.p;
    if (g_ts_batchfold && nMoves>0) {
        const int* hdesc[14]={mv_u,mv_uIsRoot, n1a_ec,n1a_slot,n1a_leaf, n1b_ec,n1b_slot,n1b_leaf,
                              n2a_ec,n2a_slot,n2a_leaf, n2b_ec,n2b_slot,n2b_leaf};
        for(int i=0;i<14;i++) GCK(cudaMemcpy(d_mvdesc+(size_t)i*nMoves, hdesc[i], (size_t)nMoves*sizeof(int), cudaMemcpyHostToDevice));
    }
    int *d_mv_u=d_mvdesc+0*(size_t)nMoves,    *d_mv_uIsRoot=d_mvdesc+1*(size_t)nMoves;
    int *d_n1a_ec=d_mvdesc+2*(size_t)nMoves,  *d_n1a_slot=d_mvdesc+3*(size_t)nMoves,  *d_n1a_leaf=d_mvdesc+4*(size_t)nMoves;
    int *d_n1b_ec=d_mvdesc+5*(size_t)nMoves,  *d_n1b_slot=d_mvdesc+6*(size_t)nMoves,  *d_n1b_leaf=d_mvdesc+7*(size_t)nMoves;
    int *d_n2a_ec=d_mvdesc+8*(size_t)nMoves,  *d_n2a_slot=d_mvdesc+9*(size_t)nMoves,  *d_n2a_leaf=d_mvdesc+10*(size_t)nMoves;
    int *d_n2b_ec=d_mvdesc+11*(size_t)nMoves, *d_n2b_slot=d_mvdesc+12*(size_t)nMoves, *d_n2b_leaf=d_mvdesc+13*(size_t)nMoves;
    double *d_baseinvar=(double*)gb_baseinvar.p; bool useInv = (pinv > 0.0 && baseinvar != nullptr);
    double *d_echild=(double*)gb_echild.p, *d_expfac=(double*)gb_uexpfac.p, *d_partial=(double*)gb_partial.p;
    double *d_pmat=(double*)gb_pmat.p;
    double *d_upper=(double*)gb_upper.p, *d_pdf=(double*)gb_pdf.p, *d_pddf=(double*)gb_pddf.p, *d_patlh=(double*)gb_patlh.p;
    double *d_n1eig=(double*)gb_n1eig.p, *d_n2eig=(double*)gb_n2eig.p, *d_valall=(double*)gb_valall.p;
    unsigned char *d_tip=(unsigned char*)gb_tip.p;
    GCK(cudaMemcpy(d_echild,echild,(size_t)nnodes*ecStride*sizeof(double),cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(d_expfac,expfac,(size_t)nnodes*exStride*sizeof(double),cudaMemcpyHostToDevice));
    // derive node-space P(b)=echild·Uinv ONCE (pattern-independent) for the stable node-space upper recurrence
    { int rows=nnodes*ncat*ns, gbp=(rows+TB-1)/TB; make_pmat<<<gbp,TB>>>(ns,ncat,nnodes,d_echild,d_pmat);
      GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError()); }

    // ---- chunk-scoped state (the lambdas capture these by ref; updated at the top of each chunk) ----
    int Pn=0, pOff=0, GB=0; size_t slotSz=0;
    std::vector<unsigned char> tipChunk((size_t)ntax*chunk0);

    auto fillChild=[&](int u,int excl,int& nch,const double** ec,const double** p,const unsigned char** t){
        nch=0; for(int k=0;k<3;k++){ec[k]=nullptr;p[k]=nullptr;t[k]=nullptr;}
        for(int kk=0; kk<node_nchild[u]; kk++){ int c=node_child[u*3+kk]; if(c==excl||nch>=3) continue;
            ec[nch]=d_echild+(size_t)c*ecStride;
            if(node_leaf[c]>=0) t[nch]=d_tip+(size_t)node_leaf[c]*Pn; else p[nch]=d_partial+(size_t)node_slot[c]*slotSz; nch++; } };
    // TS.2.1 K1: precompute EVERY move's central-edge coeff table {v0,v1,v2} ONCE. These are pattern-INDEPENDENT
    // (depend only on mv_bv[m], eval, catRate, catProp), so they upload in a SINGLE H2D copy instead of the old
    // per-move 3x cudaMemcpyToSymbol(g_val*) in the move loop (part8 #5 per-move symbol-upload stall). The move loop
    // reads each move's slice via k2_derv_mix(R=ncat) device args; the kernel arithmetic AND the host-Kahan reduceInto
    // order are UNCHANGED, so the per-move lnL stays bit-identical to the g_val path (gate: --ts-tile-check 28/28).
    {   const int Smv=3*ncat*ns; std::vector<double> valAll((size_t)(nMoves>0?nMoves:1)*Smv);
        for(int m=0;m<nMoves;m++){ double t=mv_bv[m]; double* vm=&valAll[(size_t)m*Smv];
            for(int c=0;c<ncat;c++){ double rc=catRate[c], pc=catProp[c];
                for(int x=0;x<ns;x++){ double re=rc*eval[x], e=exp(eval[x]*rc*t)*pc;
                    vm[c*ns+x]=e; vm[ncat*ns + c*ns+x]=re*e; vm[2*ncat*ns + c*ns+x]=re*re*e; } } }
        if(nMoves>0) GCK(cudaMemcpy(d_valall, valAll.data(), (size_t)nMoves*Smv*sizeof(double), cudaMemcpyHostToDevice)); }
    auto ecP=[&](int ecn){ return (ecn>=0)? (const double*)(d_echild+(size_t)ecn*ecStride) : (const double*)nullptr; };
    auto plP=[&](int slot){ return (slot>=0)? (const double*)(d_partial+(size_t)slot*slotSz) : (const double*)nullptr; };
    auto tpP=[&](int leaf){ return (leaf>=0)? (const unsigned char*)(d_tip+(size_t)leaf*Pn) : (const unsigned char*)nullptr; };
    // CONTINUOUS per-target Kahan: add THIS chunk's Pn patterns into (acc,accK) weighted by ptn_freq[pOff+p].
    // pOff strictly increases & p runs 0..Pn-1 => global add order 0..nptn-1 => bit-identical to nTile=1.
    std::vector<double> plchunk(chunk0);
    // TS.2.1 K1 (env TS_SCREEN_GPUREDUCE): replace the per-(move,chunk) full-patlh D2H + host-Kahan with an on-device
    // kj_reduce3 block reduction -> only nblk block-partials D2H (the part8 #1 reduction-stall fix JOLT already has).
    // The block-tree reduce changes the summation ORDER, so this is RESULT-INVARIANT (screener ranking robust to
    // ~1e-12, §13.9), NOT bit-identical (it drops the nTile-invariance gate). DEFAULT OFF => the proven host-Kahan path
    // is byte-identical. Needs a per-chunk device ptn_freq (uploaded in the tile loop). d_pdf/d_pddf are valid (k2_derv
    // fills all 3 channels); only the patlh channel (h_sredpart[0..GB-1]) is consumed.
    static constexpr bool TS_GPURED = false;   // RETIRED 2026-07-14 (was TS_SCREEN_GPUREDUCE): GPU-side ptnfreq
                                               // reduce experiment; never shipped. Env surface removed (constexpr).
    int GBmax = (chunk0+TB-1)/TB;
    double *d_sptnfreq=nullptr, *d_sredpart=nullptr; std::vector<double> h_sredpart;
    if (TS_GPURED) { DEVB(gb_sptnfreq,(size_t)chunk0*sizeof(double)); d_sptnfreq=(double*)gb_sptnfreq.p;
        DEVB(gb_sredpart,(size_t)3*GBmax*sizeof(double)); d_sredpart=(double*)gb_sredpart.p; h_sredpart.resize(GBmax); }
    auto reduceInto=[&](double& acc,double& accK){
        if (TS_GPURED) {
            kj_reduce3<<<GB,TB,(size_t)3*TB*sizeof(double)>>>(Pn,d_patlh,d_pdf,d_pddf,d_sptnfreq,GB,d_sredpart);
            cudaMemcpy(h_sredpart.data(),d_sredpart,(size_t)GB*sizeof(double),cudaMemcpyDeviceToHost);
            double L=acc,k=accK; for(int b=0;b<GB;b++){ double term=h_sredpart[b], y=term-k, s=L+y; k=(s-L)-y; L=s; }
            acc=L; accK=k; return; }
        cudaMemcpy(plchunk.data(),d_patlh,(size_t)Pn*sizeof(double),cudaMemcpyDeviceToHost);
        double L=acc,k=accK; for(int p=0;p<Pn;p++){ double term=ptn_freq[pOff+p]*plchunk[p], y=term-k, s=L+y; k=(s-L)-y; L=s; }
        acc=L; accK=k; };

    // PERSISTENT UPPER stored NODE-SPACE (positive products): seed root-children with k1_node_prod; interior
    // up_v = (P(b_u)·up_u) ⊙ fsib via kj_pre_node(eigOut=0). No eigen round-trip => no cancellation (F39/F40 fix).
    // TS_EIGEN_UPPER=1 (DIAGNOSTIC ONLY): selects the pre-migration EIGEN upper (k1_node + kj_pre via d_expfac) so a
    // single binary can run the eigen-vs-node-space head-to-head (reliability/wall). Default OFF = node-space production.
    constexpr bool eigUpper = false;   // RETIRED 2026-07-14 (was TS_EIGEN_UPPER): eigen-upper kernel variant;
                                       // never shipped. Env surface removed; compiler strips the launches.
    std::function<void(int)> proc=[&](int u)->double{
        for(int kk=0; kk<node_nchild[u]; kk++){
            int v=node_child[u*3+kk];
            double* pre=d_upper+(size_t)v*slotSz;
            if(u==root){
                int nch; const double* ec[3]; const double* p[3]; const unsigned char* t[3]; fillChild(root,v,nch,ec,p,t);
                if(eigUpper) launch_k1_node(GB,TB,0,ns,Pn,ncat,/*isRoot=*/0,pre,d_patlh,nch, ec[0],p[0],t[0], ec[1],p[1],t[1], ec[2],p[2],t[2]);
                else         k1_node_prod<<<GB,TB>>>(ns,Pn,ncat,pre,nch, ec[0],p[0],t[0], ec[1],p[1],t[1], ec[2],p[2],t[2]);
            } else {
                const double* ec[2]={0,0}; const double* sp[2]={0,0}; const unsigned char* st[2]={0,0}; int nsb=0;
                for(int jj=0; jj<node_nchild[u]; jj++){ int w=node_child[u*3+jj]; if(w==v||nsb>=2) continue;
                    ec[nsb]=d_echild+(size_t)w*ecStride;
                    if(node_leaf[w]>=0) st[nsb]=d_tip+(size_t)node_leaf[w]*Pn; else sp[nsb]=d_partial+(size_t)node_slot[w]*slotSz; nsb++; }
                if(eigUpper) launch_kj_pre(GB,TB,0,ns,Pn,ncat,pre,d_upper+(size_t)u*slotSz,d_expfac+(size_t)u*exStride,nsb,
                    ec[0],sp[0],st[0], ec[1],sp[1],st[1]);
                else         launch_kj_pre_node(GB,TB,0,ns,Pn,ncat,/*eigOut=*/0,pre,d_upper+(size_t)u*slotSz,d_pmat+(size_t)u*ecStride,nsb,
                    ec[0],sp[0],st[0], ec[1],sp[1],st[1]);
            }
            GCK(cudaDeviceSynchronize());
            if(node_leaf[v]<0) proc(v);
        }
        return (double)0;
    };

    // ---- per-move + tree CONTINUOUS Kahan accumulators carried across chunks (init ONCE) ----
    std::vector<double> accMove(nMoves>0?nMoves:1,0.0), accMoveK(nMoves>0?nMoves:1,0.0);
    double accTree=0.0, accTreeK=0.0;

    // ---- TILE LOOP: full postorder + tree-lnL + persistent-upper preorder + move loop, per chunk ----
    for (int tchunk=0; tchunk<nTile; tchunk++){
        pOff=tchunk*chunk0; int p1=pOff+chunk0; if(p1>nptn)p1=nptn; Pn=p1-pOff; if(Pn<=0) break;
        slotSz=(size_t)ncat*ns*Pn; GB=(Pn+TB-1)/TB;
        for(int a=0;a<ntax;a++) memcpy(&tipChunk[(size_t)a*Pn], tip+(size_t)a*nptn+pOff, (size_t)Pn);   // gather chunk columns
        GCK(cudaMemcpy(d_tip,tipChunk.data(),(size_t)ntax*Pn,cudaMemcpyHostToDevice));
        if (useInv) GCK(cudaMemcpy(d_baseinvar, baseinvar+pOff, (size_t)Pn*sizeof(double), cudaMemcpyHostToDevice));   // A3: this chunk's invariant base
        if (TS_GPURED) GCK(cudaMemcpy(d_sptnfreq, ptn_freq+pOff, (size_t)Pn*sizeof(double), cudaMemcpyHostToDevice));   // TS.2.1 K1: this chunk's ptn_freq (for kj_reduce3)

        // POSTORDER (resident lowers for THIS chunk)
        for (int idx=0; idx<nInternal; idx++){ int u=post_internal[idx]; if(u==root) continue;
            int nch; const double* ec[3]; const double* p[3]; const unsigned char* t[3]; fillChild(u,-1,nch,ec,p,t);
            launch_k1_node(GB,TB,0,ns,Pn,ncat,/*isRoot=*/0,d_partial+(size_t)node_slot[u]*slotSz,d_patlh,nch,
                ec[0],p[0],t[0], ec[1],p[1],t[1], ec[2],p[2],t[2]); }
        GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError());

        if (g_kcount) g_kc_scr_base += nInternal;   // base postorder lowers, one k1_node/internal-node, recomputed THIS chunk
        // TREE-LNL (independent root isRoot=1 reduction) -> carried accTree. MUST precede the preorder (proc clobbers d_patlh).
        { int nch; const double* ec[3]; const double* p[3]; const unsigned char* t[3]; fillChild(root,-1,nch,ec,p,t);
          launch_k1_node(GB,TB,0,ns,Pn,ncat,/*isRoot=*/1,nullptr,d_patlh,nch, ec[0],p[0],t[0], ec[1],p[1],t[1], ec[2],p[2],t[2]);
          GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError()); }
        reduceInto(accTree,accTreeK);

        // PREORDER (persistent per-node upper d_upper[v] for THIS chunk)
        proc(root);
        GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError());
        if (g_kcount) g_kc_scr_upper += nInternal;   // base preorder uppers (~1 kj_pre/kj_pre_node per internal node)

        // MOVE LOOP (each move = 2 cheap folds + k2_derv off the resident lowers + persistent uppers) -> carried accMove[m]
        // ---- launch one move m onto stream `st` into scratch slot `s` (slot s owns disjoint regions of the S-wide
        //      n1eig/n2eig/pdf/pddf/patlh buffers). st==0 (default stream) + s==0 reproduces the legacy launch EXACTLY.
        auto launchMove = [&](int m, int s, cudaStream_t st){
            int u=mv_u[m];
            double* p_n1 = d_n1eig + (size_t)s*slotSzMax;   // per-slot scratch bases (slot stride = MAX-chunk width;
            double* p_n2 = d_n2eig + (size_t)s*slotSzMax;   // kernel writes only slotSz<=slotSzMax of it -> disjoint)
            double* p_pdf= d_pdf   + (size_t)s*chunk0;
            double* p_pddf=d_pddf  + (size_t)s*chunk0;
            double* p_pl = d_patlh + (size_t)s*chunk0;
            if (mv_uIsRoot[m]) {   // u==root: no upper -> k1_node({swapped-in child, root-leaf}) (eigen), unchanged
                launch_k1_node(GB,TB,st,ns,Pn,ncat,/*isRoot=*/0,p_n1,p_pl,/*nchild=*/2,
                    ecP(n1a_ec[m]),plP(n1a_slot[m]),tpP(n1a_leaf[m]), ecP(n1b_ec[m]),plP(n1b_slot[m]),tpP(n1b_leaf[m]), nullptr,nullptr,nullptr);
            } else {               // interior u: node1Eig = Uinv·((P(b_u)·up_u) ⊙ (P(b_swapchild)·L_swapchild)), eigOut=1
                if(eigUpper) launch_kj_pre(GB,TB,st,ns,Pn,ncat,p_n1,d_upper+(size_t)u*slotSz,d_expfac+(size_t)u*exStride,/*nsib=*/1,
                    ecP(n1a_ec[m]),plP(n1a_slot[m]),tpP(n1a_leaf[m]), nullptr,nullptr,nullptr);
                else         launch_kj_pre_node(GB,TB,st,ns,Pn,ncat,/*eigOut=*/1,p_n1,d_upper+(size_t)u*slotSz,d_pmat+(size_t)u*ecStride,/*nsib=*/1,
                    ecP(n1a_ec[m]),plP(n1a_slot[m]),tpP(n1a_leaf[m]), nullptr,nullptr,nullptr);
            }
            launch_k1_node(GB,TB,st,ns,Pn,ncat,/*isRoot=*/0,p_n2,p_pl,/*nchild=*/2,   // node2Eig (lower fold), already eigen
                ecP(n2a_ec[m]),plP(n2a_slot[m]),tpP(n2a_leaf[m]), ecP(n2b_ec[m]),plP(n2b_slot[m]),tpP(n2b_leaf[m]), nullptr,nullptr,nullptr);
            const double* dv0=d_valall+(size_t)m*3*ncat*ns;   // TS.2.1 K1: this move's precomputed table slice (no per-move g_val upload)
            if (useInv)   // A3 (+I): add the invariant term pinv*baseinvar[ptn]; non-+I path UNCHANGED (same kernel below)
                k2_derv_mix_inv<<<GB,TB,0,st>>>(ns,Pn,ncat,p_n1,p_n2,dv0,dv0+ncat*ns,dv0+2*ncat*ns,pinv,d_baseinvar,p_pdf,p_pddf,p_pl);
            else
                k2_derv_mix<<<GB,TB,0,st>>>(ns,Pn,ncat,p_n1,p_n2,dv0,dv0+ncat*ns,dv0+2*ncat*ns,p_pdf,p_pddf,p_pl);
        };
        // ---- host Kahan gather of move m's per-pattern lnL row (Pn doubles) into (acc,accK), add order 0..Pn-1,
        //      reading ptn_freq[pOff+p]*row[p]. `row` must hold the SAME floats k2 wrote (== legacy plchunk). ----
        auto gatherRow = [&](const double* row, double& acc, double& accK){
            double L=acc,k=accK; for(int p=0;p<Pn;p++){ double term=ptn_freq[pOff+p]*row[p], y=term-k, s=L+y; k=(s-L)-y; L=s; }
            acc=L; accK=k; };

        // ASYNC is gated to the bit-identical host-Kahan reduce only. TS_SCREEN_GPUREDUCE is a DIFFERENT (non
        // bit-identical, default-OFF) reduce that changes summation order; do NOT combine the two — fall back to the
        // legacy serial path so each reduce keeps its documented semantics. (Validation runs neither GPUREDUCE.)
        if (g_ts_async && S>=1 && tsS && !TS_GPURED) {
            // ASYNC PATH: round-robin move m -> stream s=m%S into slot s, async-D2H slot s's patlh into pinned row m,
            // ONE sync after all moves in the chunk, THEN host Kahan gather over m in the EXACT existing order. The
            // gather reads identical floats (same kernel math; async memcpy is bit-exact) => bit-identical to serial.
            if (!ts_pin_ensure((size_t)nMoves*(size_t)chunk0)) { fprintf(stderr,"[TS-ASYNC] pinned staging alloc failed (%zu doubles)\n",(size_t)nMoves*(size_t)chunk0); return (double)NAN; }
            // CHECK: snapshot the carried-in (pre-this-chunk) accumulators so the serial shadow re-derives the same
            // per-chunk contribution from an identical baseline (the async gather below mutates accMove in place).
            std::vector<double> chkBaseline, chkBaselineK;
            if (g_ts_async_check) { chkBaseline=accMove; chkBaselineK=accMoveK; }
            for (int m=0; m<nMoves; m++){
                int s = m % S; cudaStream_t st = tsS[s];
                launchMove(m, s, st);
                double* p_pl  = d_patlh + (size_t)s*chunk0;
                double* pinRow= g_ts_pin_patlh + (size_t)m*chunk0;
                GCK(cudaMemcpyAsync(pinRow, p_pl, (size_t)Pn*sizeof(double), cudaMemcpyDeviceToHost, st));
            }
            for (int s=0;s<S;s++) GCK(cudaStreamSynchronize(tsS[s]));   // ONE sync barrier (per stream) for the chunk
            GCK(cudaGetLastError());
            // host Kahan gather in the existing fixed order m=0..nMoves-1, p=0..Pn-1 -> carried accMove[m]
            for (int m=0; m<nMoves; m++) gatherRow(g_ts_pin_patlh+(size_t)m*chunk0, accMove[m], accMoveK[m]);

            if (g_ts_async_check) {
                // BIT-EXACT SHADOW CHECK: re-run each move the LEGACY serial way (default stream, slot 0) into
                // accMove2 seeded from the carried-in baseline, then assert accMove==accMove2 bit-exact per move
                // (modeled on the --ts-tile-check harness). Aborts with the offending m on mismatch. Re-runs from
                // the SAME chunk-resident lowers/uppers (read-only, unchanged) — slot 0 device buffers are free to
                // reuse here because the async results are already staged in the pinned rows.
                for (int m=0; m<nMoves; m++){
                    double acc2=chkBaseline[m], acc2K=chkBaselineK[m];   // carried-in value (pre this chunk)
                    launchMove(m, 0, (cudaStream_t)0);                   // legacy: default stream, slot 0
                    GCK(cudaDeviceSynchronize());
                    cudaMemcpy(plchunk.data(), d_patlh, (size_t)Pn*sizeof(double), cudaMemcpyDeviceToHost);
                    gatherRow(plchunk.data(), acc2, acc2K);
                    if (memcmp(&accMove[m],&acc2,sizeof(double))!=0) {
                        fprintf(stderr,"[TS-ASYNC-CHECK] BIT MISMATCH at move m=%d (chunk pOff=%d Pn=%d): async=%.17g serial=%.17g\n",
                                m,pOff,Pn,accMove[m],acc2);
                        abort();
                    }
                }
                if (getenv("JOLT_DEBUG")) fprintf(stderr,"[TS-ASYNC-CHECK] chunk pOff=%d: %d/%d moves bit-identical (S=%d)\n",pOff,nMoves,nMoves,S);
            }
        } else if (g_ts_batchfold && !TS_GPURED) {
            // ---- TS RAKE BATCH PATH: process moves in batches of B; each batch = ONE node1 + ONE node2 + ONE k2
            //      launch over a (GB,nB) grid (blockIdx.y=local move) that fills the SMs, vs nB×3 tiny serial launches.
            //      node1/node2 -> per-move eigen slots; k2 -> per-move patlh row; host Kahan gather in the EXACT serial
            //      order (m ascending, p 0..Pn-1) => bit-identical. CHECK re-runs the serial path and asserts bit-exact. ----
            std::vector<double> chkBaseline, chkBaselineK;
            if (g_ts_batchfold_check) { chkBaseline=accMove; chkBaselineK=accMoveK; }
            int B = g_ts_batchfold_B;
            for (int bs=0; bs<nMoves; bs+=B){
                int nB = (nMoves-bs < B) ? (nMoves-bs) : B;
                dim3 grid(GB, nB);
                screen_node1_batch<<<grid,TB>>>(ns,Pn,ncat,bs, d_n1batch,slotSz, d_echild,ecStride, d_partial,d_tip,
                    d_upper,d_pmat, d_mv_u,d_mv_uIsRoot, d_n1a_ec,d_n1a_slot,d_n1a_leaf, d_n1b_ec,d_n1b_slot,d_n1b_leaf);
                screen_node2_batch<<<grid,TB>>>(ns,Pn,ncat,bs, d_n2batch,slotSz, d_echild,ecStride, d_partial,d_tip,
                    d_n2a_ec,d_n2a_slot,d_n2a_leaf, d_n2b_ec,d_n2b_slot,d_n2b_leaf);
                GCK(cudaDeviceSynchronize());
                screen_k2_batch<<<grid,TB>>>(ns,Pn,ncat,bs, d_n1batch,d_n2batch,slotSz, d_valall,d_patlhbatch,
                    useInv?1:0, pinv, d_baseinvar);
                GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError());
                GCK(cudaMemcpy(plbatch.data(), d_patlhbatch, (size_t)nB*Pn*sizeof(double), cudaMemcpyDeviceToHost));
                for (int j=0;j<nB;j++) gatherRow(plbatch.data()+(size_t)j*Pn, accMove[bs+j], accMoveK[bs+j]);
            }
            if (g_ts_batchfold_check) {
                for (int m=0;m<nMoves;m++){
                    double acc2=chkBaseline[m], acc2K=chkBaselineK[m];
                    launchMove(m, 0, (cudaStream_t)0);   // legacy serial re-run into slot 0
                    GCK(cudaDeviceSynchronize());
                    cudaMemcpy(plchunk.data(), d_patlh, (size_t)Pn*sizeof(double), cudaMemcpyDeviceToHost);
                    gatherRow(plchunk.data(), acc2, acc2K);
                    if (memcmp(&accMove[m],&acc2,sizeof(double))!=0) {
                        fprintf(stderr,"[TS-BATCHFOLD-CHECK] BIT MISMATCH at move m=%d (chunk pOff=%d Pn=%d): batch=%.17g serial=%.17g\n",
                                m,pOff,Pn,accMove[m],acc2);
                        abort();
                    }
                }
                if (getenv("JOLT_DEBUG")) fprintf(stderr,"[TS-BATCHFOLD-CHECK] chunk pOff=%d: %d/%d moves bit-identical (B=%d)\n",pOff,nMoves,nMoves,B);
            }
        } else {
            // LEGACY SERIAL PATH (byte-identical to pre-async): default stream, single scratch, per-move sync+reduce.
            for (int m=0; m<nMoves; m++){
                int u=mv_u[m];
                if (mv_uIsRoot[m]) {   // u==root: no upper -> k1_node({swapped-in child, root-leaf}) (eigen), unchanged
                    launch_k1_node(GB,TB,0,ns,Pn,ncat,/*isRoot=*/0,d_n1eig,d_patlh,/*nchild=*/2,
                        ecP(n1a_ec[m]),plP(n1a_slot[m]),tpP(n1a_leaf[m]), ecP(n1b_ec[m]),plP(n1b_slot[m]),tpP(n1b_leaf[m]), nullptr,nullptr,nullptr);
                } else {               // interior u: node1Eig = Uinv·((P(b_u)·up_u) ⊙ (P(b_swapchild)·L_swapchild)), eigOut=1
                    if(eigUpper) launch_kj_pre(GB,TB,0,ns,Pn,ncat,d_n1eig,d_upper+(size_t)u*slotSz,d_expfac+(size_t)u*exStride,/*nsib=*/1,
                        ecP(n1a_ec[m]),plP(n1a_slot[m]),tpP(n1a_leaf[m]), nullptr,nullptr,nullptr);
                    else         launch_kj_pre_node(GB,TB,0,ns,Pn,ncat,/*eigOut=*/1,d_n1eig,d_upper+(size_t)u*slotSz,d_pmat+(size_t)u*ecStride,/*nsib=*/1,
                        ecP(n1a_ec[m]),plP(n1a_slot[m]),tpP(n1a_leaf[m]), nullptr,nullptr,nullptr);
                }
                launch_k1_node(GB,TB,0,ns,Pn,ncat,/*isRoot=*/0,d_n2eig,d_patlh,/*nchild=*/2,   // node2Eig (lower fold), already eigen
                    ecP(n2a_ec[m]),plP(n2a_slot[m]),tpP(n2a_leaf[m]), ecP(n2b_ec[m]),plP(n2b_slot[m]),tpP(n2b_leaf[m]), nullptr,nullptr,nullptr);
                GCK(cudaDeviceSynchronize());
                const double* dv0=d_valall+(size_t)m*3*ncat*ns;   // TS.2.1 K1: this move's precomputed table slice (no per-move g_val upload)
                if (useInv)   // A3 (+I): add the invariant term pinv*baseinvar[ptn]; non-+I path UNCHANGED (same kernel below)
                    k2_derv_mix_inv<<<GB,TB>>>(ns,Pn,ncat,d_n1eig,d_n2eig,dv0,dv0+ncat*ns,dv0+2*ncat*ns,pinv,d_baseinvar,d_pdf,d_pddf,d_patlh);
                else
                    k2_derv_mix<<<GB,TB>>>(ns,Pn,ncat,d_n1eig,d_n2eig,dv0,dv0+ncat*ns,dv0+2*ncat*ns,d_pdf,d_pddf,d_patlh);
                GCK(cudaDeviceSynchronize());
                reduceInto(accMove[m],accMoveK[m]);
            }
        }
        if (g_kcount) g_kc_scr_fold += (long)nMoves*2;   // per-move folds: node1 + node2 partial launch per move (the dominant cost, NECESSARY — each scores a distinct NNI)
    }
    if (g_kcount) {
        g_kc_scr_calls++;
        long base=(long)nInternal*nTile, upper=(long)nInternal*nTile, fold=(long)nMoves*2*nTile;
        long tot=base+upper+fold;
        fprintf(stderr,"[TS-KCOUNT scr] call=%ld nInternal=%d nMoves=%d nTile=%d | base=%ld upper=%ld FOLD=%ld tot=%ld | fold%%=%.1f base+upper(shareable)%%=%.1f\n",
                g_kc_scr_calls,nInternal,nMoves,nTile,base,upper,fold,tot, tot?100.0*fold/tot:0.0, tot?100.0*(base+upper)/tot:0.0);
        fprintf(stderr,"[TS-KCOUNT cum] scr_base=%ld scr_upper=%ld scr_fold=%ld reopt_part=%ld (scr_calls=%ld reopt_calls=%ld)\n",
                g_kc_scr_base,g_kc_scr_upper,g_kc_scr_fold,g_kc_reopt_part,g_kc_scr_calls,g_kc_reopt_calls);
    }
    if (out_move_lnL) for(int m=0;m<nMoves;m++) out_move_lnL[m]=accMove[m];
    if (out_tree_lnL) *out_tree_lnL=accTree;
    if (out_ntile) *out_ntile=nTile;
    GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError());
    return accTree;
}

// =============================== G.4.2 — JOLT joint-gradient optimiser launcher ===============================
// Mean-rate discrete-Gamma (Yang 1994; IQ-TREE's "MEAN of the portion") — verbatim from the validated
// standalone gpu_k8b_jolt_alpha.cu (G.4.1b). r_c = K*[P(alpha+1, alpha*b_c) - P(alpha+1, alpha*b_{c-1})].
static double jolt_gammp_reg(double a, double x){   // regularized lower incomplete gamma P(a,x) (NR gser/gcf)
    if (x<=0.0) return 0.0; double gln=lgamma(a);
    if (x<a+1.0){ double ap=a,sum=1.0/a,del=sum; for(int n=1;n<=300;n++){ ap+=1.0; del*=x/ap; sum+=del; if(fabs(del)<fabs(sum)*1e-16) break; }
        return sum*exp(-x+a*log(x)-gln); }
    double b=x+1.0-a,c=1e300,d=1.0/b,h=d;
    for(int i=1;i<=300;i++){ double an=-(double)i*((double)i-a); b+=2.0; d=an*d+b; if(fabs(d)<1e-300)d=1e-300; c=b+an/c; if(fabs(c)<1e-300)c=1e-300; d=1.0/d; double del=d*c; h*=del; if(fabs(del-1.0)<1e-16) break; }
    return 1.0-exp(-x+a*log(x)-gln)*h;
}
static double jolt_gammp_inv(double a, double p){    // inverse: x s.t. P(a,x)=p, by bracketed bisection
    if (p<=0.0) return 0.0; if (p>=1.0) return 1e300;
    double lo=0.0,hi=a+10.0*sqrt(a+1.0)+20.0; int guard=0; while(jolt_gammp_reg(a,hi)<p && guard++<200) hi*=2.0;
    for(int it=0;it<200;it++){ double mid=0.5*(lo+hi); if(jolt_gammp_reg(a,mid)<p) lo=mid; else hi=mid; if(hi-lo<1e-13*(mid+1e-13)) break; }
    return 0.5*(lo+hi);
}
static void jolt_discreteGammaMean(double alpha, int K, double* rates){
    if (K==1){ rates[0]=1.0; return; }
    double prev=0.0;
    for(int c=0;c<K;c++){ double hi;
        if(c==K-1) hi=1.0;
        else { double bc=jolt_gammp_inv(alpha,(double)(c+1)/(double)K)/alpha; hi=jolt_gammp_reg(alpha+1.0, alpha*bc); }
        rates[c]=(double)K*(hi-prev); prev=hi; }
}
// G.8.2.1b: host shim so the mixture joint-optimiser's α-override can recompute mean-1 discrete-gamma rates at an
// iterate α (bit-identical to the live RateGamma::computeRatesMean / GAMMA_CUT_MEAN path used in the eligible gate).
extern "C" void gpu_discrete_gamma_mean(double alpha, int K, double* rates){ jolt_discreteGammaMean(alpha, K, rates); }

// JOLT-specific persistent device buffers (separate from the lnL/derv pools; same alloc-once / reuse policy).
static DevBuf gbj_echild, gbj_partial, gbj_patlh, gbj_pdf, gbj_pddf,
              gbj_pretmp, gbj_tipeig, gbj_prepool, gbj_expfac, gbj_rnum, gbj_tip, gbj_baseinvar,
              gbj_ptnfreq, gbj_redpart,   // G.5.0: on-device ptn_freq + per-block reduction partials
              gbj_invlbase, gbj_redR,     // G.5.0 Part B: base-edge 1/L_p + per-category gradR partials
              gbj_wnum, gbj_redW,         // G.5.1b: +R per-category Lc(p) (weight-grad numerator) + its block-reduction partials
              gbj_opgpart, gbj_opgh, gbj_opgvec;   // OPG Phase 1: Gram block-partials [NCH*GBmax], stage-2 result [NCH], K-vector uploads [2*ncat]

// --jolt-diag (A3): per-eval HOST echild-rebuild tax (host loop + 2 blocking H2D in rebuildEchild). Gated by env
// JOLT_DIAG (set by --jolt-diag; the CUDA TU cannot see Params). Accumulated across the LM loop; reported per call.
static bool   g_jdiag_init = false;
static bool   g_jdiag = false;
static double g_jd_echild_sec = 0.0;
static long   g_jd_echild_n = 0;

// JOLT_LBFGS_M (int env, default 0 = OFF): L-BFGS memory size for the per-round brlen reopt. 0 => the diagonal
// LM path below is BYTE-IDENTICAL. >0 replaces ONLY the brlen step DIRECTION with an m-pair L-BFGS two-loop
// recursion (initial Hessian H0 = diag(1/(|g_ddf|+eps)) — i.e. the SAME per-edge curvature the diagonal step uses,
// not gamma*I), capturing the off-diagonal branch coupling the Jacobi/diagonal step ignores (rho~0.6 => ~12 iters).
// Same MLE (same grad=0 stationary point), ~3-5 iters, ZERO new GPU work (the two-loop is host-side O(m*nedge)).
static bool   g_lbfgs_init = false;
static int    g_lbfgs_m = 0;
// OPG Phase 1 (build+VALIDATE the empirical-Fisher Gram; NO step change in this phase). All default-OFF => the binary
// is byte-identical to canonical unless an env var is set. JOLT_NO_OPG is the hard kill-switch (wins over everything).
static bool   g_opg_init = false;
static bool   g_opg_on   = false;   // build the Gram at all (alloc + launch)
static bool   g_opg_gchk = false;   // GATE: Check A (independent per-pattern FD) + Check B (host-vs-device Gram)
static bool   g_opg_lmin = false;   // report lambda_min of the D-scaled reduced Gram (identifiability diagnostic)

extern "C" double gpu_jolt_optimize(
    int nstates, int nptn, int ncat, int ntax, int nnodes, int root,
    const double* Uinv, const double* UinvRowSum, const double* U, const double* eval,
    const double* catProp, const unsigned char* tip, const double* ptn_freq,
    const int* node_nchild, const int* node_child, const int* node_leaf, const double* node_parentLen,
    double alpha0, int optAlpha, int maxiter,
    const double* base_invar, double pinv0, int optPinv, double pinvMin, double pinvMax,
    const double* catRate0, int freeRate,   // G.5.1: +R FreeRate — catRate0=rates[c] (else nullptr); freeRate=1 seeds rates directly (no alpha)
    int nFreeQ, const double* q0, jolt_qdecompose_fn qdecompose, void* qctx, double* out_q,   // G.6: DNA free-Q (eigensystem moves)
    double* out_brlen, double* out_alpha, double* out_pinv, int* out_iters,
    double* out_rates, double* out_props,   // G.5.1b: +R optimised rates/weights (nullptr unless freeRate==1)
    double* out_patlh)   // STAGE 2b: full-nptn accepted-tree per-pattern log|lh_ptn| snapshot (host; nullptr => no snapshot)
{
    int ns = nstates;
    if (ns > NS_MAX || ncat > 64) { fprintf(stderr,"[JOLT] unsupported ns=%d ncat=%d\n",ns,ncat); return (double)NAN; }

    // G.4.2: ModelFinder evaluates candidates ACROSS-MODEL in parallel (phylotesting.cpp:4097 omp parallel),
    // so optimizeParametersJOLT (hence this launcher) can be entered by many threads at once. The single GPU's
    // __constant__ symbols (g_Uinv/g_U/g_val*/g_rscale) and the static DevBuf pool (gbj_*) are PROCESS-GLOBAL
    // device state — concurrent use would clobber. Serialize the whole GPU computation: JOLT models run one at a
    // time on the GPU while the other threads keep optimising CPU-fallback (+I/+R/+FO) candidates. (Cross-model
    // GPU batching — running B models concurrently on the device — is the PHALANX grid.z work, G.4.3, not this.)
    static std::mutex jolt_gpu_mtx;
    std::lock_guard<std::mutex> jolt_lock(jolt_gpu_mtx);
    // --jolt-diag (A3): init the gate + snapshot the per-call echild baseline INSIDE the lock, so the
    // across-model OpenMP concurrency (ModelFinder) can't tear the baseline read or race g_jdiag_init.
    if (!g_jdiag_init) { g_jdiag = (std::getenv("JOLT_DIAG") != nullptr); g_jdiag_init = true; }
    if (!g_lbfgs_init) { const char* e=std::getenv("JOLT_LBFGS_M"); int m=e?atoi(e):0; g_lbfgs_m=(m<0)?0:((m>32)?32:m); g_lbfgs_init=true; }
    // OPG Phase 1 env gate (inside the same mutex as the others, so the ModelFinder OpenMP concurrency can't race it).
    // GRAMCHECK/LMIN imply the Gram must be built; JOLT_NO_OPG kills all of it regardless.
    if (!g_opg_init) {
        const bool nogo = (std::getenv("JOLT_NO_OPG") != nullptr);
        g_opg_gchk = !nogo && (std::getenv("JOLT_OPG_GRAMCHECK") != nullptr);
        g_opg_lmin = !nogo && (std::getenv("JOLT_OPG_LMIN") != nullptr);
        g_opg_on   = !nogo && ((std::getenv("JOLT_OPG") != nullptr) || g_opg_gchk || g_opg_lmin);
        g_opg_init = true;
    }
    double _jd_ech0 = g_jd_echild_sec; long _jd_echn0 = g_jd_echild_n;
    // Inc 2 reopt proof-counters (JOLT_DEBUG): per-call tally of reopt coefficient uploads via the OFF path
    // (cudaMemcpyToSymbol of g_val0/1/2+g_rscale) vs the ON path (ONE cudaMemcpyAsync of [v0|v1|v2|rs] into the
    // valpool). ON should drive memcpyToSymbol->0. Function-local (reset per call); guarded by the GPU mutex above.
    long ts_reopt_mcs = 0;   // OFF-path reopt cudaMemcpyToSymbol count (g_val0/1/2 + g_rscale per edge)
    long ts_reopt_vp  = 0;   // ON-path reopt cudaMemcpyAsync-to-valpool count

    // alpha-independent eigen constants — upload once (the BASE-Q eigensystem). For free-Q (nFreeQ>0) qApply()
    // re-uploads these whenever an exchangeability changes; for fixed-Q this is the only upload.
    CC_TOSYM(0, g_Uinv, Uinv, sizeof(double)*ns*ns);
    CC_TOSYM(1, g_U,    U,    sizeof(double)*ns*ns);
    CC_TOSYM(2, g_UinvRowSum, UinvRowSum, sizeof(double)*ns);

    // G.6 free-Q: MUTABLE working copies of the eigensystem (refreshed per Q change via qApply). For fixed-Q
    // (nFreeQ==0) evalP/UP alias the passed-in const arrays (the lambdas below use evalP/UP, so the fixed-Q path is
    // byte-identical to before). For free-Q they point at the buffers that qApply overwrites in place.
    std::vector<double> evalB, UB, UinvB;
    const double *evalP = eval, *UP = U;
    if (nFreeQ > 0) { evalB.assign(eval, eval+ns); UB.assign(U, U+ns*ns); UinvB.assign(Uinv, Uinv+ns*ns);
                      evalP = evalB.data(); UP = UB.data(); }
    auto qApply = [&](const double* q) -> void {   // re-decompose for a trial Q and re-upload eval/U/Uinv; rebuildEchild()/setVal() then use the new evalP/UP
        // NB: plain cudaMemcpyToSymbol (NOT GCK) — GCK's `return (double)NAN` would return from THIS lambda (void),
        // swallowing the error + UB. Matches rebuildEchild's plain copies; any failure is caught by the final
        // cudaGetLastError() backstop (the sticky last-error persists to the end of gpu_jolt_optimize -> NaN -> CPU).
        qdecompose(qctx, q, evalB.data(), UB.data(), UinvB.data());
        if(!cc_skip_toSymbol(0, UinvB.data(), sizeof(double)*ns*ns)) cudaMemcpyToSymbol(g_Uinv, UinvB.data(), sizeof(double)*ns*ns);
        if(!cc_skip_toSymbol(1, UB.data(),    sizeof(double)*ns*ns)) cudaMemcpyToSymbol(g_U,    UB.data(),    sizeof(double)*ns*ns);
        double rs[NS_MAX]; for(int i=0;i<ns;i++){ double s=0; for(int j=0;j<ns;j++) s+=UinvB[i*ns+j]; rs[i]=s; }
        if(!cc_skip_toSymbol(2, rs, sizeof(double)*ns)) cudaMemcpyToSymbol(g_UinvRowSum, rs, sizeof(double)*ns); };

    // ---- rebuild topology from flat arrays (node ids = caller's DFS index) ----
    std::vector<std::vector<int>> child(nnodes);
    std::vector<int> leaf(nnodes);
    for (int u=0; u<nnodes; u++){ leaf[u]=node_leaf[u];
        for (int k=0; k<node_nchild[u] && k<3; k++){ int c=node_child[u*3+k]; if (c>=0) child[u].push_back(c); } }
    std::vector<double> brlen(node_parentLen, node_parentLen+nnodes);

    std::vector<int> postorder; std::vector<int> slot(nnodes,-1);
    std::function<void(int)> dfs=[&](int u){ for(int c:child[u]) dfs(c); if(leaf[u]<0){ slot[u]=(int)postorder.size(); postorder.push_back(u);} };
    dfs(root); int nInternal=(int)postorder.size();
    int c0=-1; for(int c:child[root]) if(leaf[c]<0){ c0=c; break; } if(c0<0){ fprintf(stderr,"[JOLT] no internal root child\n"); return (double)NAN; }
    std::vector<int> edgeV; for(int u=0;u<nnodes;u++) for(int v:child[u]) edgeV.push_back(v); int nedge=(int)edgeV.size();
    int treeH=0; std::function<void(int,int)> ddfs=[&](int u,int d){ if(d>treeH)treeH=d; for(int c:child[u]) ddfs(c,d+1); }; ddfs(root,0); int nPool=treeH+2;

    size_t ecStride=(size_t)ncat*ns*ns;
    int TB=256;

    // ===== G.7.1 PATTERN TILING — fit the O(nptn) partial arenas on smaller GPUs =====
    // Every JOLT quantity (lnL, df_e, ddf_e, gradR_c) is a SUM OVER PATTERNS, so partitioning the nptn patterns into
    // nTile contiguous chunks, running a full postorder+preorder sweep PER CHUNK, and Kahan-accumulating each chunk's
    // contribution reproduces the one-shot result to rel<=1e-12 (part7 §VII.3 — the same additivity that already
    // underlies the ptn_freq-weighted reductions). Every O(nptn) device arena (the dominant postorder gbj_partial, the
    // preorder pool, scratch, tip/patlh/...) shrinks by ~nTile, so a model needing 886 GB one-shot at AA-10M fits an
    // H200 (141 GB) at nTile>=8 / an A100 (80 GB) at nTile>=12. The chunk-INDEPENDENT echild/expfac/eigen constants are
    // built once per (brlen,alpha,pinv,Q) point (rebuildEchild), NOT per chunk.
    int nTile = 1;
    if (const char* e = getenv("JOLT_NTILE")) { nTile = atoi(e); if (nTile < 1) nTile = 1; }
    else {
        // auto-pick from free VRAM: estimate the one-shot footprint, target 80% of free, round up.
        size_t slot1 = (size_t)ncat*ns*nptn*sizeof(double);
        size_t foot  = (size_t)(nInternal + nPool + 3) * slot1                 // partial + prepool + (pretmp/tipeig+slack)
                     + (size_t)ncat*nptn*sizeof(double)                        // rnum
                     + (size_t)6*nptn*sizeof(double)                           // patlh/pdf/pddf/baseinvar/ptnfreq/invlbase
                     + (size_t)ntax*nptn                                       // tip
                     + (size_t)nnodes*ecStride*sizeof(double);                 // echild (chunk-independent; not tiled)
        size_t freeB=0, totB=0;
        if (cudaMemGetInfo(&freeB,&totB)==cudaSuccess && freeB>0) {
            double budget = 0.80 * (double)freeB;
            int T = (int)ceil((double)foot / budget); if (T<1) T=1;
            nTile = T;
        }
    }
    // FIX 2026-07-15 (JOLT_RTILE): the production +R LM is chunk-safe -- the lnL loop (:2919) AND the gradient loop
    // (:2981) tile and Kahan-accumulate across chunks (accW->WNc, rel<=1e-12 vs one-shot). ONLY the OPTIONAL RGRADCHECK
    // finite-difference DIAGNOSTIC uses full-nptn buffers. The old `if (freeRate) nTile = 1;` disabled tiling for EVERY
    // +R model -> AA-1M +R allocated the full 74.2GB gbj_partial arena and OOM'd on 80GB. Now +R auto-tiles like every
    // other family (fits A100-80GB at nTile~2-3). JOLT_RTILE_OFF restores the old force-nTile=1 for rollback/A-B.
    if (freeRate && (getenv("JOLT_RGRADCHECK") || getenv("JOLT_RTILE_OFF"))) nTile = 1;
    if (getenv("JOLT_DEBUG")) {
        size_t fB=0,tB=0; cudaMemGetInfo(&fB,&tB);
        fprintf(stderr,"[JOLT-TILE] nptn=%d ns=%d ncat=%d nInternal=%d nPool=%d -> nTile=%d (chunk~%d ptn); freeVRAM=%.1f GB\n",
                nptn,ns,ncat,nInternal,nPool,nTile,(nptn+nTile-1)/nTile,(double)fB/1073741824.0); fflush(stderr);
    }

    int    chunk0    = (nptn + nTile - 1) / nTile;   // max chunk width; all per-pattern buffers are sized to this
    size_t slotSzMax = (size_t)ncat*ns*chunk0;
    int    GBmax     = (chunk0 + TB - 1) / TB;
    // current-chunk state (MUTABLE; set by setChunk; the sweep closures capture these by reference). At nTile==1
    // chunk0==nptn / Pn==nptn / pOff==0 / slotSz==slotSzMax / GB==GBmax => byte-identical to the pre-tiling path.
    int    Pn     = nptn;                            // current chunk's pattern count
    int    pOff   = 0;                               // current chunk's first pattern index (into the host inputs)
    size_t slotSz = (size_t)ncat*ns*nptn;            // current chunk's [cat][state][ptn] slot stride
    int    GB     = (nptn + TB - 1) / TB;            // current chunk's grid
    (void)pOff;

    // ===== Inc 2 (async/streams ladder) reopt substrate =====
    // The per-edge coefficient block staged contiguously as [v0|v1|v2|rs]: v0,v1,v2 each ncat*ns doubles, then rscale
    // ncat. Under JOLT_TS_ASYNC the reopt sweep stages this block into a private valpool slot + ONE async H2D and runs
    // kj_derv_fused_args (kernel-arg coeffs) instead of the per-edge cudaMemcpyToSymbol(g_val*/g_rscale)+kj_derv_fused.
    // Inc 2 keeps reopt SERIAL (S=1, one slot, per-edge stream sync) — no overlap, no cross-edge d_rnum+= race; Inc 3
    // round-robins. tsS/S are acquired ONCE here (lazy pool, mirrors the screener at the top of gpuScreenNNIRank); on
    // any pool failure tsS==nullptr => every gated site below falls through to the BYTE-IDENTICAL legacy branch.
    const int valBlk = 3*ncat*ns + ncat;             // doubles per reopt coeff block ([v0|v1|v2|rs])
    cudaStream_t* tsS = nullptr; int S = 1;          // Inc 2: one active reopt slot (Inc 3 will raise S)
    if (g_ts_async) { int poolK=0; tsS = ts_streams(&poolK); if(!tsS||poolK<1){ tsS=nullptr; } }

    DEVB(gbj_echild, (size_t)nnodes*ecStride*sizeof(double));
    DEVB(gbj_partial,(size_t)(nInternal>0?nInternal:1)*slotSzMax*sizeof(double));
    // part8 #3 cleanup: d_theta (the 601 MB theta arena) is DEAD now that kj_derv_fused computes theta in registers.
    DEVB(gbj_patlh,  (size_t)chunk0*sizeof(double)); DEVB(gbj_pdf,(size_t)chunk0*sizeof(double)); DEVB(gbj_pddf,(size_t)chunk0*sizeof(double));
    DEVB(gbj_pretmp, slotSzMax*sizeof(double)); DEVB(gbj_tipeig, slotSzMax*sizeof(double));
    DEVB(gbj_prepool,(size_t)nPool*slotSzMax*sizeof(double));
    DEVB(gbj_expfac, (size_t)nnodes*ncat*ns*sizeof(double));
    DEVB(gbj_rnum,   (size_t)ncat*chunk0*sizeof(double));
    DEVB(gbj_tip,    (size_t)ntax*chunk0);
    DEVB(gbj_baseinvar, (size_t)chunk0*sizeof(double));   // G.4.3b +I: pinv-independent invariant base per pattern
    DEVB(gbj_ptnfreq,   (size_t)chunk0*sizeof(double));   // G.5.0: pattern weights, constant across the optimise call
    double *d_echild=(double*)gbj_echild.p,*d_partial=(double*)gbj_partial.p;
    double *d_patlh=(double*)gbj_patlh.p,*d_pdf=(double*)gbj_pdf.p,*d_pddf=(double*)gbj_pddf.p;
    double *d_pretmp=(double*)gbj_pretmp.p,*d_tipeig=(double*)gbj_tipeig.p,*d_prepool=(double*)gbj_prepool.p;
    double *d_expfac=(double*)gbj_expfac.p,*d_rnum=(double*)gbj_rnum.p,*d_baseinvar=(double*)gbj_baseinvar.p;
    double *d_ptnfreq=(double*)gbj_ptnfreq.p;
    unsigned char* d_tip=(unsigned char*)gbj_tip.p;
    // tip/ptn_freq/base_invar are CONSTANT across the optimise call; setChunk uploads the current chunk's slice. (At
    // nTile==1 this uploads the whole arrays once per sweep — byte-identical device state to the old one-time upload.)
    std::vector<unsigned char> tipChunk((size_t)ntax*chunk0);
    std::vector<double> biFull(nptn, 0.0); if (base_invar) for (int p=0;p<nptn;p++) biFull[p]=base_invar[p];
    // Phase 1a JOLT_MF_RESIDENT (MODELFINDER-FULL-GPU-PLAN.md §F): tip/ptn_freq/base_invar are CONSTANT across this
    // optimise call (comment above) and the three device buffers d_tip/d_ptnfreq/d_baseinvar are written ONLY by
    // setChunk and are read-only to every kernel (verified: no kernel writes gbj_tip/gbj_ptnfreq/gbj_baseinvar).
    // So re-uploading the same chunk is redundant. nsys(173763069)+sqlite: at 1M this re-upload = 86.5% of H2D
    // (451.5s, the 93.5MB d_tip re-sent ~505x/candidate). Guard: skip the re-gather+3xH2D when the requested chunk
    // is already resident. Byte-identical by construction (identical device bytes).
    // loadedChunk is per-CALL (resets each gpu_jolt_optimize invocation) so a fresh candidate always re-uploads.
    // ✅ GRADUATED TO DEFAULT-ON 2026-07-17 (kill-switch JOLT_NO_MF_RESIDENT).
    // EVIDENCE (verified on disk, gems-verify/mfresprof_173781194): DNA-1M (935227 distinct patterns) -m MF,
    // mf_off 2571.527s -> mf_on 1610.467s = 1.597x; bit-identical (best model F81+F+G4 both, lnL -59208019.1016
    // both, treefile md5 identical, fitted params identical).
    // ⚠️ THE SAFETY INVARIANT IS THE MUTEX, NOT THE LOCAL. My first comment credited "loadedChunk is a per-call
    // local, not shared state" -- that is true (:349, captured by-ref) but it would NOT be sufficient on its own:
    // d_tip/d_ptnfreq/d_baseinvar alias the process-global static DevBufs gbj_tip/gbj_ptnfreq/gbj_baseinvar
    // (:2605-2609), and a per-call cursor over a SHARED device buffer is exactly a stale-read race. What actually
    // makes this safe is that gpu_jolt_optimize holds the process-wide `static std::mutex jolt_gpu_mtx` across its
    // ENTIRE body (:2648-2649), so no two candidates are ever in flight on those buffers at once.
    // 🔴 IF jolt_gpu_mtx IS EVER NARROWED OR REMOVED, THIS FLAG BECOMES A SILENT STALE-CHUNK CORRECTNESS BUG.
    // (md5 "26fba2bb" was cited for this job in my earlier notes; it is NOT recorded in the job artifacts and is
    //  therefore dropped rather than repeated -- the 1.597x and the bit-identity are independently verified above.)
    static const bool mf_resident = (getenv("JOLT_NO_MF_RESIDENT") == nullptr);
    int loadedChunk = -1;
    auto setChunk=[&](int t){
        int p0=t*chunk0, p1=p0+chunk0; if(p1>nptn)p1=nptn; int cw=p1-p0;
        Pn=cw; pOff=p0; slotSz=(size_t)ncat*ns*cw; GB=(cw+TB-1)/TB;
        if (mf_resident && t == loadedChunk) return;   // Phase 1a: chunk already resident -> skip redundant re-upload
        for(int a=0;a<ntax;a++) memcpy(&tipChunk[(size_t)a*cw], tip+(size_t)a*nptn+p0, (size_t)cw);
        cudaMemcpy(d_tip, tipChunk.data(), (size_t)ntax*cw, cudaMemcpyHostToDevice);
        cudaMemcpy(d_ptnfreq, ptn_freq+p0, (size_t)cw*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_baseinvar, biFull.data()+p0, (size_t)cw*sizeof(double), cudaMemcpyHostToDevice);
        loadedChunk = t;   // Phase 1a
    };

    DEVB(gbj_redpart, (size_t)3*GBmax*sizeof(double));   // G.5.0: 3 channels x GBmax per-block partial sums
    double* d_redpart=(double*)gbj_redpart.p;
    std::vector<double> h_redpart((size_t)3*GBmax);
    DEVB(gbj_invlbase, (size_t)chunk0*sizeof(double)); double* d_invLbase=(double*)gbj_invlbase.p;  // G.5.0 Part B
    DEVB(gbj_redR,     (size_t)ncat*GBmax*sizeof(double)); double* d_redR=(double*)gbj_redR.p;
    std::vector<double> h_redR((size_t)ncat*GBmax);
    // G.5.1b +R: weight-grad numerator Lc(p) buffer + its per-block reduction (only referenced when freeRate==1 engages
    // the +R joint LM below). Independent cudaMalloc => no effect on any other buffer's layout (byte-identical pools).
    double* d_wnum=nullptr; double* d_redW=nullptr; std::vector<double> h_redW;
    if(freeRate==1){ DEVB(gbj_wnum,(size_t)ncat*chunk0*sizeof(double)); d_wnum=(double*)gbj_wnum.p;
                     DEVB(gbj_redW,(size_t)ncat*GBmax*sizeof(double)); d_redW=(double*)gbj_redW.p; h_redW.assign((size_t)ncat*GBmax,0.0); }
    // ===== OPG Phase 1 state. The ALLOCATION is flag-gated (not just the launch): the DevBuf pool is never freed and
    // the MFP VRAM balloon is an open defect, so an OFF run must not grow VRAM by even one buffer. ncat outside [2,10]
    // => opgNCH=0 => OPG disabled entirely (hard fallback; JOLT_FREERATE_HIGHK can push ncat past 10).
    const int  opgK   = ncat;
    const int  opgNCH = (opgK>=2 && opgK<=10) ? opgK*(2*opgK+1) : 0;
    const bool opgOK  = (freeRate==1) && g_opg_on && (opgNCH>0);
    double *d_opgPart=nullptr,*d_opgH=nullptr,*d_opgCRCP=nullptr,*d_opgBpr=nullptr;
    std::vector<double> opgH, opgHk, opgCR, opgBP;                 // Gram (upper-tri, Kahan across chunks) + the K-vectors
    std::vector<double> opg_capt; std::vector<int> opg_captP; bool opg_captOn=false;   // Check-A analytic s_p capture
    double opg_gram_sec=0.0; long opg_gram_n=0;                    // P5 cost micro-benchmark
    if(opgOK){
        DEVB(gbj_opgpart,(size_t)opgNCH*GBmax*sizeof(double)); d_opgPart=(double*)gbj_opgpart.p;
        DEVB(gbj_opgh,   (size_t)opgNCH*sizeof(double));       d_opgH   =(double*)gbj_opgh.p;
        DEVB(gbj_opgvec, (size_t)2*ncat*sizeof(double));       d_opgCRCP=(double*)gbj_opgvec.p; d_opgBpr=d_opgCRCP+ncat;
        opgH.assign((size_t)opgNCH,0.0); opgHk.assign((size_t)opgNCH,0.0); opgCR.assign(ncat,0.0); opgBP.assign(ncat,0.0);
    }
    std::vector<double> h_echild((size_t)nnodes*ecStride), h_expfac((size_t)nnodes*ncat*ns);
    std::vector<double> patlh(nptn),pdf(nptn),pddf(nptn);
    std::vector<double> catRate(ncat,1.0), catProp_v(catProp, catProp+ncat);
    std::vector<double> meanR(ncat,1.0);   // G.4.3b: mean-1 discrete-gamma rates (alpha-dependent ONLY)
    double curAlpha=alpha0;
    auto applyAlpha=[&](double a){ jolt_discreteGammaMean(a,ncat,meanR.data()); };  // -> meanR (mean 1)
    if (ncat>1 && !freeRate) applyAlpha(curAlpha);   // G.5.1: +R seeds rates directly (below), not from alpha
    // G.4.3b +I: IQ-TREE's RateGammaInvar uses getProp(c)=(1-pinv)/K AND rescales the gamma rates to meanR[c]/(1-pinv)
    // (RateGamma::computeRates preserves the pre-set curScale=K/(1-pinv) => mean-1 rates / (1-pinv); so the OVERALL
    // mean rate incl. invariant sites at rate 0 stays 1). Both the rate (UP by 1/(1-pinv)) and the prop (DOWN by
    // (1-pinv)) move with pinv. catProp[c] arrives as (1-pinv0)/K, so the pinv-free base prop is bprop=catProp/(1-pinv0).
    // For non-+I (optPinv=0): f=1 => catRate=meanR, catProp_v=catProp (byte-identical to the pre-+I +G path).
    std::vector<double> bprop(ncat);
    for(int c=0;c<ncat;c++) bprop[c] = optPinv ? catProp[c]/(1.0-pinv0) : catProp[c];
    // G.5.1 +R FreeRate: seed the PINV-FREE basis meanR=ρ_c (mean-1 rates, Σ w·ρ=1), bprop=w_c (weights), so
    // applyPinv(pinv0) reproduces catRate=getRate, catProp_v=getProp. RateFreeInvar: getRate=ρ/(1-p), getProp=(1-p)w
    // (model/ratefreeinvar.cpp:48-60) => meanR=catRate0*f0, bprop=catProp/f0 with f0=(1-pinv0). G.5.1d (2b). For pure
    // +R (optPinv=0) f0=1 => BYTE-IDENTICAL to before (meanR=rates, bprop=weights, applyPinv(0) identity).
    if (freeRate) { double f0 = optPinv ? (1.0-pinv0) : 1.0;
                    for(int c=0;c<ncat;c++){ meanR[c]=catRate0[c]*f0; bprop[c]=catProp[c]/f0; } }
    double curPinv = optPinv ? pinv0 : 0.0;
    auto applyPinv=[&](double p){ double f = optPinv ? (1.0-p) : 1.0;
        for(int c=0;c<ncat;c++){ catRate[c]=meanR[c]/f; catProp_v[c]=f*bprop[c]; } };
    applyPinv(curPinv);

    auto childArgs=[&](int u,int excl,int& nch,const double** ec,const double** p,const unsigned char** t){
        nch=0; for(int k=0;k<3;k++){ec[k]=p[k]=nullptr;t[k]=nullptr;}
        for(int c:child[u]){ if(c==excl) continue; ec[nch]=d_echild+(size_t)c*ecStride;
            if(leaf[c]>=0) t[nch]=d_tip+(size_t)leaf[c]*Pn; else p[nch]=d_partial+(size_t)slot[c]*slotSz; nch++; } };   // G.7.1: tip stride = current chunk width Pn
    auto sibArg=[&](int w,const double*& ec,const double*& sp,const unsigned char*& st){
        ec=d_echild+(size_t)w*ecStride; sp=nullptr; st=nullptr;
        if(leaf[w]>=0) st=d_tip+(size_t)leaf[w]*Pn; else sp=d_partial+(size_t)slot[w]*slotSz; };
    auto rebuildEchild=[&](){
        std::chrono::steady_clock::time_point _jd_e0; if(g_jdiag) _jd_e0 = std::chrono::steady_clock::now();   // --jolt-diag
        for(int c=0;c<nnodes;c++){ if(c==root){ for(size_t z=0;z<ecStride;z++) h_echild[(size_t)c*ecStride+z]=0.0; continue; }
            for(int cat=0;cat<ncat;cat++){ double len=brlen[c]*catRate[cat]; double ex[NS_MAX]; for(int i=0;i<ns;i++) ex[i]=exp(evalP[i]*len);
                double* e=&h_echild[(size_t)c*ecStride+(size_t)cat*ns*ns]; for(int x=0;x<ns;x++) for(int i=0;i<ns;i++) e[x*ns+i]=UP[x*ns+i]*ex[i];
                for(int i=0;i<ns;i++) h_expfac[(size_t)c*ncat*ns+cat*ns+i]=ex[i]; } }
        cudaMemcpy(d_echild,h_echild.data(),(size_t)nnodes*ecStride*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(d_expfac,h_expfac.data(),(size_t)nnodes*ncat*ns*sizeof(double),cudaMemcpyHostToDevice);
        if(g_jdiag){ g_jd_echild_sec += std::chrono::duration<double>(std::chrono::steady_clock::now()-_jd_e0).count(); g_jd_echild_n++; } };   // --jolt-diag: echild tax
    auto postorderFill=[&](){
        for(int idx=0; idx<nInternal; idx++){ int u=postorder[idx]; if(u==root) continue;
            int nch; const double* ec[3]; const double* p[3]; const unsigned char* t[3]; childArgs(u,-1,nch,ec,p,t);
            launch_k1_node(GB,TB,0,ns,Pn,ncat,0,d_partial+(size_t)slot[u]*slotSz,d_patlh,nch,ec[0],p[0],t[0],ec[1],p[1],t[1],ec[2],p[2],t[2]); }
        cudaDeviceSynchronize(); };
    // Inc 2 (S3): the central-edge coefficient tables, split into a PURE host-build helper (setValBuild) and the
    // legacy upload (setVal). setValBuild fills v0/v1/v2 with the EXACT same FP64 math (same exp(evalP[x]*rc*t)*pcw
    // factor order) the old single lambda used; setVal still does the 3 cudaMemcpyToSymbol of those SAME bytes, so
    // the OFF path is BYTE-IDENTICAL. The ON reopt path calls setValBuild directly (no constant-memory upload) and
    // stages the bytes into the valpool. v0/v1/v2 must be sized ncat*ns by the caller.
    auto setValBuild=[&](double t,std::vector<double>& v0,std::vector<double>& v1,std::vector<double>& v2){
        for(int c=0;c<ncat;c++){ double rc=catRate[c],pcw=catProp_v[c]; for(int x=0;x<ns;x++){ double re=rc*evalP[x],e=exp(evalP[x]*rc*t)*pcw;
            v0[c*ns+x]=e; v1[c*ns+x]=re*e; v2[c*ns+x]=re*re*e; } } };
    auto setVal=[&](double t){ std::vector<double> v0(ncat*ns),v1(ncat*ns),v2(ncat*ns);
        setValBuild(t,v0,v1,v2);
        cudaMemcpyToSymbol(g_val0,v0.data(),sizeof(double)*ncat*ns); cudaMemcpyToSymbol(g_val1,v1.data(),sizeof(double)*ncat*ns); cudaMemcpyToSymbol(g_val2,v2.data(),sizeof(double)*ncat*ns);
        ts_reopt_mcs += 3; };   // Inc 2: OFF-path proof-counter (3 constant-memory uploads per setVal)
    auto reduceDerv=[&](double& lnL,double& df,double& ddf){
        // G.5.0: on-device ptn_freq-weighted block reduction -> 3*GB partials D2H (88 KB), replacing the old
        // 3x nptn D2H (~22 MB/call x ~197 edges/sweep) + single-thread host Kahan. Final cross-block combine kept
        // on host in the SAME channel order (Kahan) for bit-reproducibility -> stable LM accept/reject.
        kj_reduce3<<<GB,TB,(size_t)3*TB*sizeof(double)>>>(Pn,d_patlh,d_pdf,d_pddf,d_ptnfreq,GB,d_redpart);
        cudaMemcpy(h_redpart.data(),d_redpart,(size_t)3*GB*sizeof(double),cudaMemcpyDeviceToHost);
        double L=0,kc=0,D=0,kd=0,DD=0,kdd=0;
        for(int b=0;b<GB;b++){
            { double term=h_redpart[b],            y=term-kc, s=L +y; kc =(s-L )-y; L =s; }
            { double term=h_redpart[(size_t)GB+b], y=term-kd, s=D +y; kd =(s-D )-y; D =s; }
            { double term=h_redpart[(size_t)2*GB+b],y=term-kdd,s=DD+y; kdd=(s-DD)-y; DD=s; } }
        lnL=L; df=D; ddf=DD; };
    // part8 #3: resolve v's eigen node-partial pointer (synthesising a leaf tip vector into d_tipeig if needed)
    // WITHOUT materialising theta — kj_derv_fused reads node+dad directly. Replaces edgeThetaInto on the fused path.
    auto edgeNodePtr=[&](int v)->const double*{
        if(leaf[v]<0) return d_partial+(size_t)slot[v]*slotSz;
        k_leaf_eig<<<GB,TB>>>(ns,Pn,ncat,d_tip+(size_t)leaf[v]*Pn,d_tipeig); return d_tipeig; };

    // ============ G.5.1a — +R weight-gradient FINITE-DIFFERENCE self-check (gated; then declines to CPU) ============
    // Validates gz_c = WN_c − w_c·N (softmax weight gradient, PART IX §IX.8) against central FD on the REAL GPU path,
    // BEFORE any +R optimiser branch is wired (the G.4.0b discipline: prove the new gradient first). Weights enter only
    // through g_val0 (setVal), not the partials/echild, and Lc(p) is edge-invariant — so WN_c comes from the base edge
    // and each FD perturbation re-runs only setVal + one base-edge kj_derv_fused (cheap). +R always declines to CPU here
    // (the LM branch is G.5.1b); this hook only runs the check under JOLT_RGRADCHECK.
    if (freeRate) {
        if (getenv("JOLT_RGRADCHECK")) {
            setChunk(0);   // G.7.1: nTile==1 for +R — upload the (whole) tip/ptn_freq/base_invar slice the old one-time copy used to do
            rebuildEchild(); postorderFill();
            int nch; const double* ec[3]; const double* p[3]; const unsigned char* tp[3]; childArgs(root,c0,nch,ec,p,tp);
            launch_k1_node(GB,TB,0,ns,nptn,ncat,0,d_pretmp,d_patlh,nch,ec[0],p[0],tp[0],ec[1],p[1],tp[1],ec[2],p[2],tp[2]); cudaDeviceSynchronize();
            const double* pl0=edgeNodePtr(c0); cudaDeviceSynchronize();
            cudaMemset(d_rnum,0,(size_t)ncat*nptn*sizeof(double));   // reuse the rnum buffer as wnum (rnum unused here)
            setVal(brlen[c0]);
            kj_derv_fused<<<GB,TB>>>(ns,nptn,ncat,pl0,d_pretmp,0.0,d_baseinvar,d_patlh,d_pdf,d_pddf,nullptr,d_rnum); cudaDeviceSynchronize();
            double lnL0,dtmp,ddtmp; reduceDerv(lnL0,dtmp,ddtmp);
            kj_invl<<<GB,TB>>>(nptn,d_patlh,d_invLbase); cudaDeviceSynchronize();
            kj_reduce_gradnum<<<GB,TB,(size_t)TB*sizeof(double)>>>(nptn,ncat,d_rnum,d_invLbase,d_ptnfreq,GB,d_redR); cudaDeviceSynchronize();
            cudaMemcpy(h_redR.data(),d_redR,(size_t)ncat*GB*sizeof(double),cudaMemcpyDeviceToHost);
            std::vector<double> WN(ncat,0.0); double sumWN=0;
            for(int c=0;c<ncat;c++){ long double a=0; for(int b=0;b<GB;b++) a+=(long double)h_redR[(size_t)c*GB+b]; WN[c]=(double)a; sumWN+=WN[c]; }
            double Ntot=0; for(int pp=0;pp<nptn;pp++) Ntot+=ptn_freq[pp];
            std::vector<double> w(catProp_v), gz(ncat); double sumgz=0;
            for(int c=0;c<ncat;c++){ gz[c]=WN[c]-w[c]*Ntot; sumgz+=gz[c]; }
            auto lnlW=[&](const std::vector<double>& wv)->double{   // lnL re-eval at perturbed weights (partials unchanged)
                std::vector<double> save=catProp_v; catProp_v=wv; setVal(brlen[c0]);
                kj_derv_fused<<<GB,TB>>>(ns,nptn,ncat,pl0,d_pretmp,0.0,d_baseinvar,d_patlh,d_pdf,d_pddf,nullptr,nullptr); cudaDeviceSynchronize();
                double l,a2,b2; reduceDerv(l,a2,b2); catProp_v=save; return l; };
            auto softmax=[&](const std::vector<double>& z){ double mx=z[0]; for(int c=1;c<ncat;c++) if(z[c]>mx)mx=z[c];
                std::vector<double> o(ncat); double s=0; for(int c=0;c<ncat;c++){o[c]=exp(z[c]-mx); s+=o[c];} for(int c=0;c<ncat;c++)o[c]/=s; return o; };
            double eps=1e-4, maxrel=0;
            for(int d=0; d<ncat; d++){
                std::vector<double> zp(ncat),zm(ncat); for(int c=0;c<ncat;c++){ zp[c]=log(w[c]); zm[c]=zp[c]; }
                zp[d]+=eps; zm[d]-=eps;
                double lp=lnlW(softmax(zp)), lm=lnlW(softmax(zm));
                double fd=(lp-lm)/(2.0*eps); double rel=fabs(gz[d]-fd)/(fabs(fd)+1e-30);
                if(rel>maxrel) maxrel=rel;
                fprintf(stderr,"[RGRADCHECK] c=%d WN=%.6e w=%.6f gz=%.6e FD=%.6e rel=%.3e\n",d,WN[d],w[d],gz[d],fd,rel);
            }
            double relWN=fabs(sumWN-Ntot)/Ntot;
            fprintf(stderr,"[RGRADCHECK] ncat=%d lnL0=%.6f sumWN=%.6f N=%.0f relWN=%.3e sumGz=%.3e maxrel=%.3e -> %s\n",
                ncat,lnL0,sumWN,Ntot,relWN,sumgz,maxrel,(maxrel<1e-4 && relWN<1e-9)?"RGRADCHECK PASS":"RGRADCHECK FAIL");
            fflush(stderr);
        }
        // G.5.1b: freeRate==1 ENGAGES the +R joint LM (falls through to the gradient infra + LM loop below). freeRate==2
        // is the RGRADCHECK-only diagnostic (ncat>maxcat / ineligible regime the gate let through ONLY for the FD check)
        // -> decline to CPU after the check, preserving the historical G.5.1a behaviour for the high-ncat validation path.
        // L5 (2026-07-09): freeRate==3 = SEED-ONLY +R for the brlen-only path. The rates/weights are seeded above (meanR
        // from catRate0, :2779) and held FIXED: every rate/weight (y/z) LM arm, the weight-grad buffers, gaugeFix, and the
        // out_rates/out_props write-back below are ==1-gated, so freeRate==3 falls through to run the SAME branch-only
        // diagonal LM as +G4 (applyAlpha is !freeRate-gated => skipped => rates never reconstructed from alpha). See
        // phylotreegpu.cpp freeRateBrlenOK.
        if (freeRate != 1 && freeRate != 3) return (double)NAN;
    }

    long nGradSweeps=0,nLnLEval=0;
    // part8 #2 base-sweep skip: record the (brlen,alpha,pinv) the device echild/partial were last built for (by
    // evalLnL) so computeGradient can skip the redundant rebuild+postorder when its base already matches. Values are
    // COPIES of the same candidate vectors (no recompute) => exact == is reliable; any mismatch falls back to rebuild.
    std::vector<double> devB; double devA=1e300, devP=1e300; bool devValid=false;
    // STAGE 2b (GPU-BOOTSTRAP-UFBOOT-PLAN §3 Stage 2b): out_patlh snapshot state. `snapPatlh` is set ONLY for the
    // final re-eval snapshot call (nTile>1 / no-accept / L-BFGS) => the tile-loop D2H below is skipped during the LM
    // loop => byte-identical. `snapDone` records that the cheap snapshot-on-accept fast path already filled out_patlh
    // (nTile==1 case) so the final re-eval is skipped. Both stay inert unless out_patlh != nullptr (i.e. -B + opt-in).
    double* snapPatlh = nullptr; bool snapDone = false;
    auto evalLnL=[&](const std::vector<double>& cand_b,double cand_a,double cand_pinv,const double* cand_q)->double{
        if(nFreeQ>0 && cand_q) qApply(cand_q);   // G.6: re-decompose+reupload the trial Q -> rebuildEchild() below uses the new evalP/UP
        if(ncat>1 && !freeRate) applyAlpha(cand_a); applyPinv(cand_pinv); brlen=cand_b;   // G.5.1b: +R seeds rates directly -> never re-derive from alpha (would clobber meanR)
        rebuildEchild();   // G.7.1: chunk-INDEPENDENT (echild/expfac carry no nptn) — build once per eval, reused across all chunks
        double Lacc=0,Lk=0;   // G.7.1: Kahan accumulator of lnL over the pattern chunks (exact additivity, rel<=1e-12 vs one-shot)
        for(int t=0;t<nTile;t++){
            setChunk(t); postorderFill();
            int nch; const double* ec[3]; const double* p[3]; const unsigned char* tp[3]; childArgs(root,c0,nch,ec,p,tp);
            launch_k1_node(GB,TB,0,ns,Pn,ncat,0,d_pretmp,d_patlh,nch,ec[0],p[0],tp[0],ec[1],p[1],tp[1],ec[2],p[2],tp[2]); cudaDeviceSynchronize();
            const double* pl0=edgeNodePtr(c0); cudaDeviceSynchronize();   // part8 #3: fused — no theta materialisation
            // Inc 2: evalLnL's base-edge derv is derv-ONLY (rnum=wnum=nullptr) — it never reads g_rscale. Under ON we
            // host-build only v0/v1/v2 into the valpool slot's [v0|v1|v2] region (the rscale tail is left unread because
            // kj_derv_fused_args dereferences drscale only when rnum!=nullptr) and run kj_derv_fused_args; OFF stays the
            // BYTE-IDENTICAL setVal+kj_derv_fused. CHECK re-runs the legacy serial derv and memcmps (l) bit-exact.
            double l,d,dd;
            if (g_ts_async && tsS && ts_valpool_ensure((size_t)S*valBlk)) {
                cudaStream_t st=tsS[0]; double* vp=g_ts_valpool;   // slot 0 (reopt serial in Inc 2)
                std::vector<double> v0(ncat*ns),v1(ncat*ns),v2(ncat*ns); setValBuild(brlen[c0],v0,v1,v2);
                std::vector<double> hostVal(3*ncat*ns);
                for(int j=0;j<ncat*ns;j++){ hostVal[j]=v0[j]; hostVal[ncat*ns+j]=v1[j]; hostVal[2*ncat*ns+j]=v2[j]; }
                GCK(cudaMemcpyAsync(vp,hostVal.data(),(size_t)3*ncat*ns*sizeof(double),cudaMemcpyHostToDevice,st)); ts_reopt_vp++;
                kj_derv_fused_args<<<GB,TB,0,st>>>(ns,Pn,ncat,pl0,d_pretmp,cand_pinv,d_baseinvar,d_patlh,d_pdf,d_pddf,nullptr,nullptr,
                    vp, vp+ncat*ns, vp+2*ncat*ns, vp+3*ncat*ns);
                GCK(cudaStreamSynchronize(st));
                reduceDerv(l,d,dd);
                if (g_ts_async_check) {
                    double sl,sd,sdd; setVal(brlen[c0]); kj_derv_fused<<<GB,TB>>>(ns,Pn,ncat,pl0,d_pretmp,cand_pinv,d_baseinvar,d_patlh,d_pdf,d_pddf,nullptr,nullptr); cudaDeviceSynchronize();
                    reduceDerv(sl,sd,sdd);
                    if (memcmp(&l,&sl,sizeof(double))!=0) {
                        fprintf(stderr,"[TS-ASYNC-CHECK] evalLnL BIT MISMATCH (chunk pOff=%d Pn=%d): async lnL=%.17g serial=%.17g\n",pOff,Pn,l,sl);
                        abort();
                    }
                }
            } else {
                setVal(brlen[c0]); kj_derv_fused<<<GB,TB>>>(ns,Pn,ncat,pl0,d_pretmp,cand_pinv,d_baseinvar,d_patlh,d_pdf,d_pddf,nullptr,nullptr); cudaDeviceSynchronize();   // BYTE-IDENTICAL legacy
                reduceDerv(l,d,dd);
            }
            double y=l-Lk, s=Lacc+y; Lk=(s-Lacc)-y; Lacc=s;   // Kahan add this chunk's lnL contribution
            if(snapPatlh) cudaMemcpy(snapPatlh+pOff, d_patlh, (size_t)Pn*sizeof(double), cudaMemcpyDeviceToHost);   // STAGE 2b: this tile's per-pattern log|lh| (d_patlh synced by reduceDerv above); pOff/Pn set by setChunk
        }
        devB=cand_b; devA=cand_a; devP=cand_pinv; devValid=true;   // part8 #2: echild matches this base (full postorder present on device only when nTile==1)
        return Lacc; };

    std::vector<double> g_df(nedge,0.0),g_ddf(nedge,0.0),gradR(ncat,0.0);   // G.5.0 Part B: invL/rnumH now on-device
    // G.5.1b +R: weight-grad outputs filled by computeGradient when freeRate==1. WNc[c]=Σ_p Lc(p)/L_p·freq (edge-invariant,
    // taken at the base edge — mirrors the RGRADCHECK path above), gzR[c]=WNc[c]-w_c·N (softmax weight gradient). N once.
    std::vector<double> WNc(freeRate==1?ncat:0,0.0), gzR(freeRate==1?ncat:0,0.0);
    double rN=0.0; if(freeRate==1){ for(int p=0;p<nptn;p++) rN+=ptn_freq[p]; }
    auto computeGradient=[&](double& lnLout,double& galphaOut){
        applyPinv(curPinv);   // G.4.3b: align catRate=meanR/(1-curPinv) and catProp_v to the base pinv before the sweep
        if(opgOK){ for(int q=0;q<opgNCH;q++){ opgH[q]=0.0; opgHk[q]=0.0; } }   // OPG: H accumulates over THIS sweep's chunks only
        // part8 #2 base-sweep skip (tiling-aware): echild/expfac are chunk-INDEPENDENT, so skip the rebuild when the
        // device echild already matches this base point (built by the immediately-preceding accepted evalLnL). The
        // POSTORDER partials, however, are present on device for ALL patterns only when nTile==1 (one chunk); with
        // nTile>1 they hold just the LAST chunk, so postorderFill must rerun per chunk below.
        // NB the skip tracks (brlen,alpha,pinv) only — a Q-FD step (G.6) moves the eigensystem WITHOUT moving those,
        // so for free-Q it is unsafe; disabled when nFreeQ>0.
        bool devMatch = (nFreeQ==0) && devValid && devA==curAlpha && devP==curPinv && (int)devB.size()==nnodes;
        if(devMatch) for(int z=0;z<nnodes;z++) if(devB[z]!=brlen[z]){ devMatch=false; break; }
        if(!devMatch) rebuildEchild();
        bool postValid = devMatch && (nTile==1);
        nGradSweeps++;
        // G.7.1: cross-chunk Kahan accumulators for the per-pattern sums (df_e, ddf_e, lnL, and the raw rate-grad
        // numerator per category). Deterministic chunk order 0..nTile-1 => reproducible; rel<=1e-12 vs the one-shot sum.
        std::vector<double> accDf(nedge,0.0),accDfK(nedge,0.0),accDdf(nedge,0.0),accDdfK(nedge,0.0);
        std::vector<double> accR(ncat,0.0),accRk(ncat,0.0);
        std::vector<double> accW(freeRate==1?ncat:0,0.0),accWk(freeRate==1?ncat:0,0.0);   // G.5.1b +R weight-grad numerator (Kahan across chunks)
        double Lacc=0,Lk=0;
        for(int t=0;t<nTile;t++){
            setChunk(t);
            if(!postValid) postorderFill();
            cudaMemset(d_rnum,0,(size_t)ncat*Pn*sizeof(double));
            std::vector<int> freeSlots; for(int s=nPool-1;s>=0;s--) freeSlots.push_back(s);
            auto acq=[&](){int s=freeSlots.back();freeSlots.pop_back();return s;}; auto rls=[&](int s){freeSlots.push_back(s);};
            std::vector<double> dfC(nnodes,0.0),ddfC(nnodes,0.0); bool gotL=false; double lnLfirst=0;
            long reoptEdges=0, reoptChkOk=0;   // Inc 2 S5: per-chunk reopt edge tally + CHECK bit-identical tally (mirrors the screener's per-chunk count)
            std::function<void(int,int)> proc=[&](int u,int su){
                for(int v:child[u]){
                    int sv=acq(); double* pre=d_prepool+(size_t)sv*slotSz;
                    if(u==root){ int nch; const double* ec[3]; const double* p[3]; const unsigned char* tp[3]; childArgs(root,v,nch,ec,p,tp);
                        launch_k1_node(GB,TB,0,ns,Pn,ncat,0,pre,d_patlh,nch,ec[0],p[0],tp[0],ec[1],p[1],tp[1],ec[2],p[2],tp[2]); }
                    else { const double* ec[2]={0,0}; const double* sp[2]={0,0}; const unsigned char* st[2]={0,0}; int nsb=0;
                        for(int w:child[u]){ if(w==v||nsb>=2) continue; sibArg(w,ec[nsb],sp[nsb],st[nsb]); nsb++; }
                        launch_kj_pre(GB,TB,0,ns,Pn,ncat,pre,d_prepool+(size_t)su*slotSz,d_expfac+(size_t)u*ncat*ns,nsb,ec[0],sp[0],st[0],ec[1],sp[1],st[1]); }
                    if (g_kcount) g_kc_reopt_part++;   // reopt: one upper-partial launch (k1_node root / kj_pre interior) per edge per LM iter
                    cudaDeviceSynchronize();
                    const double* plv=edgeNodePtr(v);   // part8 #3: fused theta+derv+ratenum, no d_theta round-trip
                    double bv=brlen[v]; std::vector<double> rs(ncat); for(int c=0;c<ncat;c++) rs[c]=bv/(catRate[c]*catProp_v[c]);
                    // Inc 2: gated reopt coeff delivery. ON => host-build [v0|v1|v2|rs], ONE async H2D into valpool slot
                    // vp, kj_derv_fused_args (kernel-arg coeffs). OFF => BYTE-IDENTICAL legacy (memcpyToSymbol+setVal+
                    // kj_derv_fused). vp is declared here so the shared +R wnum re-derv below can reuse THIS edge's slot
                    // (the +R landmine: the legacy wnum kj_derv_fused reads __constant__ g_val0, which the ON path never
                    // uploads -> stale; under ON we route the wnum re-derv through kj_derv_fused_args with the SAME vp).
                    bool reoptAsync = (g_ts_async && tsS && ts_valpool_ensure((size_t)S*valBlk));
                    double* vp = nullptr;
                    double l,d,dd;
                    if (reoptAsync) {
                        int s=0; cudaStream_t st=tsS[s];               // Inc 2: reopt still SERIAL; one slot. (Inc 3 round-robins.)
                        vp = g_ts_valpool + (size_t)s*valBlk;
                        std::vector<double> v0(ncat*ns),v1(ncat*ns),v2(ncat*ns); setValBuild(bv,v0,v1,v2);
                        std::vector<double> hostVal(valBlk);           // stage contiguously [v0|v1|v2|rs] for ONE async copy
                        for(int j=0;j<ncat*ns;j++){ hostVal[j]=v0[j]; hostVal[ncat*ns+j]=v1[j]; hostVal[2*ncat*ns+j]=v2[j]; }
                        for(int c=0;c<ncat;c++) hostVal[3*ncat*ns+c]=rs[c];
                        cudaMemcpyAsync(vp,hostVal.data(),(size_t)valBlk*sizeof(double),cudaMemcpyHostToDevice,st); ts_reopt_vp++;   // bare (NOT GCK): proc is std::function<void> — GCK's `return (double)NAN` would not typecheck; errors caught by the final cudaGetLastError() backstop
                        kj_derv_fused_args<<<GB,TB,0,st>>>(ns,Pn,ncat,plv,pre,curPinv,d_baseinvar,d_patlh,d_pdf,d_pddf,d_rnum,nullptr,
                            vp, vp+ncat*ns, vp+2*ncat*ns, vp+3*ncat*ns);
                        cudaStreamSynchronize(st);                     // keep reopt serial in Inc 2 (no cross-edge d_rnum+= race)
                        reduceDerv(l,d,dd);
                        if (g_ts_async_check) {
                            // S5 reopt CHECK shadow: re-run THIS edge the LEGACY serial way (constant memory + kj_derv_fused
                            // on the default stream) into a shadow, then memcmp the reduced (l,d,dd) bit-exact. d_rnum was
                            // already advanced by the async kernel above, so the shadow derv passes nullptr for rnum (it must
                            // NOT double-accumulate); the (l,d,dd) channels come from d_patlh/d_pdf/d_pddf which the shadow
                            // recomputes identically. Mismatch => print BIT MISMATCH + edge id and abort().
                            double sl,sd,sdd; cudaMemcpyToSymbol(g_rscale,rs.data(),sizeof(double)*ncat); setVal(bv); cudaDeviceSynchronize();
                            kj_derv_fused<<<GB,TB>>>(ns,Pn,ncat,plv,pre,curPinv,d_baseinvar,d_patlh,d_pdf,d_pddf,nullptr,nullptr); cudaDeviceSynchronize();
                            reduceDerv(sl,sd,sdd);
                            if (memcmp(&l,&sl,sizeof(double))!=0 || memcmp(&d,&sd,sizeof(double))!=0 || memcmp(&dd,&sdd,sizeof(double))!=0) {
                                fprintf(stderr,"[TS-ASYNC-CHECK] reopt BIT MISMATCH at edge v=%d (chunk pOff=%d Pn=%d): async(l,d,dd)=%.17g,%.17g,%.17g serial=%.17g,%.17g,%.17g\n",
                                        v,pOff,Pn,l,d,dd,sl,sd,sdd);
                                abort();
                            }
                            reoptChkOk++;
                        }
                    } else {
                        cudaMemcpyToSymbol(g_rscale,rs.data(),sizeof(double)*ncat); setVal(bv); cudaDeviceSynchronize();   // BYTE-IDENTICAL legacy
                        kj_derv_fused<<<GB,TB>>>(ns,Pn,ncat,plv,pre,curPinv,d_baseinvar,d_patlh,d_pdf,d_pddf,d_rnum,nullptr); cudaDeviceSynchronize();
                        ts_reopt_mcs += 1;   // Inc 2: OFF-path proof-counter for the per-edge g_rscale upload (setVal counts the 3 g_val* uploads)
                        reduceDerv(l,d,dd);
                    }
                    reoptEdges++;
                    dfC[v]=d; ddfC[v]=dd;
                    if(!gotL){ lnLfirst=l;
                        // G.5.0 Part B: 1/L_p on-device from the base-edge patlh (was a host D2H + exp loop over nptn).
                        kj_invl<<<GB,TB>>>(Pn,d_patlh,d_invLbase); gotL=true;
                        // G.5.1b +R: per-category likelihood Lc(p) (weight-grad numerator), edge-invariant => captured once
                        // at this first edge of the chunk (re-derv with the wnum output; setVal/plv/pre/g_rscale still set
                        // for THIS edge from the call above). kj_invl ran first so d_invLbase holds the correct base 1/L_p
                        // BEFORE this derv overwrites d_patlh/d_pdf/d_pddf (already reduced for edge v at line above).
                        if(freeRate==1){
                            cudaMemset(d_wnum,0,(size_t)ncat*Pn*sizeof(double));
                            // +R LANDMINE FIX: under ON the legacy kj_derv_fused (reading __constant__ g_val0) would read a
                            // STALE g_val0 (the ON path never uploaded it). Route the wnum re-derv through kj_derv_fused_args
                            // with THIS edge's still-resident vp (same [v0|v1|v2|rs] bytes) => bit-identical to the OFF path.
                            if (reoptAsync) {
                                cudaStream_t st=tsS[0];
                                kj_derv_fused_args<<<GB,TB,0,st>>>(ns,Pn,ncat,plv,pre,curPinv,d_baseinvar,d_patlh,d_pdf,d_pddf,nullptr,d_wnum,
                                    vp, vp+ncat*ns, vp+2*ncat*ns, vp+3*ncat*ns);
                                cudaStreamSynchronize(st);   // bare (NOT GCK): proc returns void
                            } else {
                                kj_derv_fused<<<GB,TB>>>(ns,Pn,ncat,plv,pre,curPinv,d_baseinvar,d_patlh,d_pdf,d_pddf,nullptr,d_wnum); cudaDeviceSynchronize();
                            }
                            kj_reduce_gradnum<<<GB,TB,(size_t)TB*sizeof(double)>>>(Pn,ncat,d_wnum,d_invLbase,d_ptnfreq,GB,d_redW);
                            cudaMemcpy(h_redW.data(),d_redW,(size_t)ncat*GB*sizeof(double),cudaMemcpyDeviceToHost);
                            for(int c=0;c<ncat;c++){ long double a=0; for(int b=0;b<GB;b++) a+=(long double)h_redW[(size_t)c*GB+b];
                                double term=(double)a; double y=term-accWk[c], s=accW[c]+y; accWk[c]=(s-accW[c])-y; accW[c]=s; } } }
                    if(leaf[v]<0) proc(v,sv); rls(sv);
                } };
            proc(root,-1); cudaDeviceSynchronize();
            // Inc 2 S5: per-chunk reopt CHECK summary (mirrors the screener's "[TS-ASYNC-CHECK] ... bit-identical").
            // reoptChkOk counts edges where the async (l,d,dd) matched the legacy serial shadow bit-exact; any mismatch
            // already abort()ed inside proc, so reaching here means reoptChkOk==reoptEdges.
            if (g_ts_async && g_ts_async_check)
                fprintf(stderr,"[TS-ASYNC-CHECK] reopt edges %ld/%ld bit-identical (chunk pOff=%d Pn=%d)\n",reoptChkOk,reoptEdges,pOff,Pn);
            // accumulate this chunk's per-edge df/ddf and base-edge lnL (Kahan):
            for(int e=0;e<nedge;e++){
                double td=dfC[edgeV[e]];  { double y=td-accDfK[e],  s=accDf[e] +y; accDfK[e] =(s-accDf[e]) -y; accDf[e] =s; }
                double t2=ddfC[edgeV[e]]; { double y=t2-accDdfK[e], s=accDdf[e]+y; accDdfK[e]=(s-accDdf[e])-y; accDdf[e]=s; } }
            { double y=lnLfirst-Lk, s=Lacc+y; Lk=(s-Lacc)-y; Lacc=s; }
            // G.5.0 Part B: per-category block reduction of ptn_freq*rnum[c]*invL (on-device); accumulate the RAW
            // numerator across chunks (the catProp_v[c] factor is applied once after the chunk loop).
            kj_reduce_gradnum<<<GB,TB,(size_t)TB*sizeof(double)>>>(Pn,ncat,d_rnum,d_invLbase,d_ptnfreq,GB,d_redR);
            cudaMemcpy(h_redR.data(),d_redR,(size_t)ncat*GB*sizeof(double),cudaMemcpyDeviceToHost);
            for(int c=0;c<ncat;c++){ long double a=0; for(int b=0;b<GB;b++) a+=(long double)h_redR[(size_t)c*GB+b];
                double term=(double)a; double y=term-accRk[c], s=accR[c]+y; accRk[c]=(s-accR[c])-y; accR[c]=s; }
            // ===== OPG Phase 1 (insertion E): Gram accumulation for THIS chunk. This is the ONLY point where all four
            // inputs are simultaneously valid: d_rnum is complete after proc(root,-1) above (it is memset per chunk),
            // and d_wnum / d_invLbase were filled at THIS chunk's first edge. Buffers are Pn-strided (Pn shrinks on the
            // last chunk) and block partials are GB-strided, NOT GBmax. Phase 1 makes NO step change: opgH is only
            // validated and reported.
            if(opgOK){
                for(int c=0;c<ncat;c++){ opgCR[c]=catRate[c]*catProp_v[c]; opgBP[c]=bprop[c]; }
                cudaMemcpy(d_opgCRCP,opgCR.data(),(size_t)ncat*sizeof(double),cudaMemcpyHostToDevice);
                cudaMemcpy(d_opgBpr, opgBP.data(),(size_t)ncat*sizeof(double),cudaMemcpyHostToDevice);
                std::chrono::steady_clock::time_point _o0=std::chrono::steady_clock::now();
                const size_t opgSh=(size_t)opgNCH*(size_t)(TB>>5)*sizeof(double);
                if(!jolt_opg_gram_launch(ncat,GB,TB,opgSh,Pn,d_rnum,d_wnum,d_invLbase,d_ptnfreq,d_opgCRCP,d_opgBpr,GB,d_opgPart)){
                    fprintf(stderr,"[OPG] gram dispatch failed for ncat=%d\n",ncat); fflush(stderr);
                } else {
                    kj_opg_reduce2<<<opgNCH,TB,(size_t)TB*sizeof(double)>>>(GB,d_opgPart,d_opgH);
                    cudaDeviceSynchronize();
                    opg_gram_sec += std::chrono::duration<double>(std::chrono::steady_clock::now()-_o0).count(); opg_gram_n++;
                    std::vector<double> hch((size_t)opgNCH);
                    cudaMemcpy(hch.data(),d_opgH,(size_t)opgNCH*sizeof(double),cudaMemcpyDeviceToHost);
                    for(int q=0;q<opgNCH;q++){ double y=hch[q]-opgHk[q], s2=opgH[q]+y; opgHk[q]=(s2-opgH[q])-y; opgH[q]=s2; }
                    // Check B (KERNEL) + Check-A capture. Gated on opg_captOn so the O(Pn*NCH) host reference runs on
                    // exactly ONE sweep (driven from the gate block), not on all ~400 LM iterations.
                    if(g_opg_gchk && opg_captOn){
                        const int N2=2*ncat;
                        std::vector<double> hr((size_t)ncat*Pn),hw((size_t)ncat*Pn),hi((size_t)Pn),hf((size_t)Pn);
                        cudaMemcpy(hr.data(),d_rnum,     (size_t)ncat*Pn*sizeof(double),cudaMemcpyDeviceToHost);
                        cudaMemcpy(hw.data(),d_wnum,     (size_t)ncat*Pn*sizeof(double),cudaMemcpyDeviceToHost);
                        cudaMemcpy(hi.data(),d_invLbase, (size_t)Pn*sizeof(double),      cudaMemcpyDeviceToHost);
                        cudaMemcpy(hf.data(),d_ptnfreq,  (size_t)Pn*sizeof(double),      cudaMemcpyDeviceToHost);
                        std::vector<long double> ref((size_t)opgNCH,0.0L); std::vector<double> sv((size_t)N2);
                        for(int p=0;p<Pn;p++){
                            double il=hi[p], Rp=0.0;
                            for(int c=0;c<ncat;c++) Rp += il*hw[(size_t)c*Pn+p];
                            for(int c=0;c<ncat;c++){ sv[c]=opgCR[c]*il*hr[(size_t)c*Pn+p]; sv[ncat+c]=il*hw[(size_t)c*Pn+p]-opgBP[c]*Rp; }
                            int q=0; for(int a=0;a<N2;a++) for(int b=a;b<N2;b++){ ref[q]+=(long double)hf[p]*sv[a]*sv[b]; q++; }
                            int gp=pOff+p;   // Check-A: stash analytic s_p for the sampled GLOBAL patterns in this chunk
                            for(size_t k=0;k<opg_captP.size();k++) if(opg_captP[k]==gp)
                                for(int t=0;t<N2;t++) opg_capt[k*(size_t)N2+(size_t)t]=sv[t];
                        }
                        double mx=0; for(int q=0;q<opgNCH;q++) mx=fmax(mx,fabs((double)ref[q]));
                        double wr=0; int wq=-1, nb=0;
                        for(int q=0;q<opgNCH;q++){ double r=(double)ref[q], ad=fabs(hch[q]-r);
                            double rel=ad/(fabs(r)+1e-300);
                            if(!((rel<1e-10)||(ad<1e-12*mx))) nb++;                       // hybrid: rel, or negligible vs the largest channel
                            if(rel>wr && fabs(r)>1e-8*mx){ wr=rel; wq=q; } }
                        fprintf(stderr,"[OPGGRAM-B] pOff=%d Pn=%d nTile=%d NCH=%d worst_ch=%d worstrel=%.3e nbad=%d -> %s\n",
                                pOff,Pn,nTile,opgNCH,wq,wr,nb,(nb==0)?"GRAMB PASS":"GRAMB FAIL");
                        fflush(stderr);
                    }
                }
            }
        }
        for(int e=0;e<nedge;e++){ g_df[e]=accDf[e]; g_ddf[e]=accDdf[e]; }
        for(int c=0;c<ncat;c++) gradR[c]=catProp_v[c]*accR[c];
        // G.5.1b/G.5.1d: softmax weight gradient gz_c = WN_c − w_c·(Σ_p freq·S_p/L_p), w_c=bprop[c]. For pure +R the
        // normalizer Σ freq·S/L = N = rN. For +I, L_p=(1-p)S_p+p·I_p, and Σ_k WN_k = Σ freq·(1-p)S/L = N − p·Σ freq·I/L
        // EQUALS that normalizer (·(1-p) cancels into WN) — so use Σ_k WN_k under +I (NO extra reduction needed). At
        // pinv=0: bprop==catProp_v and ΣWN==rN mathematically, but rN is kept to stay BYTE-IDENTICAL (avoids ΣaccW vs Σfreq sum-order ~1e-12 drift).
        if(freeRate==1){ double sumWN=0; for(int c=0;c<ncat;c++) sumWN+=accW[c];
            double wnorm = optPinv ? sumWN : rN;
            for(int c=0;c<ncat;c++){ WNc[c]=accW[c]; gzR[c]=WNc[c]-bprop[c]*wnorm; } }
        double ga=0;
        // alpha gradient: ga = Σ_c (d catRate[c]/dα)·gradR[c]; catRate[c]=meanR[c]/f so the perturbed mean-1 rate
        // rp[c] must be scaled by 1/f too (else mixing scaled/unscaled rates -> wrong alpha grad on the +I path).
        if(ncat>1 && !freeRate){ double f = optPinv ? (1.0-curPinv) : 1.0; double rp[64]; jolt_discreteGammaMean(curAlpha+1e-5,ncat,rp);
            for(int c=0;c<ncat;c++) ga+=((rp[c]/f-catRate[c])/1e-5)*gradR[c]; }   // G.5.1b: no alpha for +R (rates are free params, not gamma-derived)
        lnLout=Lacc; galphaOut=ga; };

    // ---- single joint LM diagonal-Newton optimise from the provided (warm) start ----
    std::vector<double> cand(nnodes,0.0), base;
    std::vector<double> startB(node_parentLen, node_parentLen+nnodes);   // distinct from brlen (evalLnL overwrites brlen)
    curAlpha=alpha0;
    // G.6 free-Q: qcur = running free-Q vector (seeded from q0); set the device to base Q before the first eval so
    // computeGradient's rebuildEchild (skip disabled for free-Q) reads the correct eigensystem.
    const double MINQ=1e-4, MAXQ=100.0;   // == MIN_RATE / MAX_RATE (modelmarkov.h)
    std::vector<double> qcur(nFreeQ>0?nFreeQ:0), qPrev(nFreeQ>0?nFreeQ:0,0.0), gqPrev(nFreeQ>0?nFreeQ:0,0.0);
    if(nFreeQ>0){ for(int k=0;k<nFreeQ;k++) qcur[k]=q0[k]; qApply(qcur.data()); }
    // ===== G.5.1b +R FreeRate joint-LM state (all empty/untouched unless freeRate==1) =====
    // y_c=log(r_c) (log-rate space, so the LM moves rates multiplicatively); z_c = softmax weight-logit. gaugeFix pins
    // Σ w·r=1 by rescaling rates and folding the scale into branch lengths (lnL-invariant) — keeps the FreeRate rate<->
    // branch-scale degeneracy fixed (identifiability + reproducibility, ref gpu_k8c_jolt_freerate.cu:385).
    std::vector<double> zR(freeRate==1?ncat:0,0.0), ryPrev(freeRate==1?ncat:0,0.0), rgyPrev(freeRate==1?ncat:0,0.0),
                        rzPrev(freeRate==1?ncat:0,0.0), rgzPrev(freeRate==1?ncat:0,0.0);
    std::vector<double> baseR_save(freeRate==1?ncat:0,0.0), baseW_save(freeRate==1?ncat:0,0.0);
    // L7 Stage B (2026-07-09): EM closed-form weight M-step. When JOLT_REM is set AND pure +R (optPinv==0), the softmax-
    // GRADIENT weight arm is replaced by the EM JUMP w_c = WNc[c]/Σ_c WNc[c], where WNc[c]=Σ_p freq_p·γ_pc is the EM
    // posterior sum ALREADY reduced for the weight gradient (:3086; Σ_c WNc = N for pure +R, so w sums to 1). Block-
    // alternation: EM-optimal weights (given base rates/branches) combined with the diagonal-LM branch/rate step, kept
    // monotone by the accept-if-better test. The most-coupled block (Σw=1 simplex) then converges in ~1 step instead of
    // many damped gradient steps => fewer outer iters at high K (the measured 97-149s R5-R10 handicap, job 173435343).
    // Default-OFF => byte-identical (the softmax-gradient path is untouched). Pure +R only (pinv=0); +I+R keeps the
    // gradient arm (its EM normaliser is 1-pinv, deferred). Plan §2c.4 Stage B.
    static constexpr bool JOLT_REM_EN = false;   // RETIRED 2026-07-14 (was JOLT_REM): EM closed-form weight M-step,
                                                 // measured NEUTRAL (1.0x, insurance only). Env surface removed.
    // ---- +R GAUGE INSTRUMENTATION (RDIAG; host-side counters only -> numerics UNCHANGED whether or not it prints) ----
    // gaugeFix rescales (catRate/=m, brlen*=m). That product is what the likelihood depends on (P = exp(Q·rate·brlen)),
    // so the transform is EXACTLY lnL-invariant in exact arithmetic -- EXCEPT for the >20.0 clamp below, which breaks
    // invariance for any branch already sitting at the cap (the LM trial itself caps at 20.0, :3285, so branches CAN
    // rest exactly there; with m>1 the branch re-clamps to 20.0 while its rate was still divided by m => a real 1/m
    // change in rate*brlen on that branch). rd_clamps/rd_maxm measure whether that actually fires -- the one
    // source-visible non-invariance, and the open question in the root-cause chain.
    long   rd_clamps = 0;    // # of times the >20.0 clamp truncated a branch during a gauge
    double rd_maxm   = 0.0;  // worst |m-1| seen (how far the mean rate drifts before being gauged back)
    auto gaugeFix=[&](){ double m=0; for(int c=0;c<ncat;c++) m+=catProp_v[c]*catRate[c];   // m = overall mean rate (Σ catProp_v·catRate; the +I invariant class adds 0) — pin =1 (same constraint for +R and +I+R)
        if(fabs(m-1.0)>rd_maxm) rd_maxm=fabs(m-1.0);
        if(m>0){ for(int c=0;c<ncat;c++) catRate[c]/=m; for(int v=0;v<nnodes;v++){ brlen[v]*=m; if(brlen[v]>20.0){ brlen[v]=20.0; rd_clamps++; } } }
        double f = optPinv ? (1.0-curPinv) : 1.0;   // G.5.1d (2b): meanR is the PINV-FREE rate ρ=catRate·(1-pinv); applyPinv(curPinv) then reproduces catRate. f==1 (pure +R) => byte-identical.
        for(int c=0;c<ncat;c++) meanR[c]=catRate[c]*f; };
    auto softmaxApply=[&](const std::vector<double>& z,std::vector<double>& w){
        double mx=z[0]; for(int c=1;c<ncat;c++) if(z[c]>mx) mx=z[c];
        double s=0; for(int c=0;c<ncat;c++){ w[c]=exp(z[c]-mx); s+=w[c]; } for(int c=0;c<ncat;c++) w[c]/=s;
        double tot=0; for(int c=0;c<ncat;c++){ if(w[c]<1e-4) w[c]=1e-4; tot+=w[c]; } for(int c=0;c<ncat;c++) w[c]/=tot; };
    // G.5.1e WARM-SEED (fixes the real-data rate-collapse trap; avian GTR+F+I+R2 job 172452425 collapsed both R2 rates
    // to 1.411). The pre-fix start reset meanR to a flat ~1 (off the constructor catRate=1.0) — a SYMMETRIC stationary
    // point: for ncat≥2 with equal rates the rate-separation gradient can vanish, trapping +R at the collapsed (sub-
    // optimal) optimum. Instead START from the CPU's SEPARATED rate estimates meanR=ρ_c (seeded at :1820 from getRate),
    // gauged to the mean-1 base Σ bprop·meanR=1 with the scale folded into startB. Unifies the pure-+R and +I+R paths
    // (operates on the pinv-free basis meanR/bprop, so no optPinv branch). NB this CHANGES the +R convergence path vs the
    // pre-warm-seed binary (no longer bit-identical for +R) — validated by GPU≥CPU + rate-separation, not bit-identity.
    if(freeRate==1){ for(int c=0;c<ncat;c++) zR[c]=log(bprop[c]);   // weight-logits from the pinv-free weights w_c (== log(catProp_v) at pinv=0; softmax shift-invariant)
        double m=0; for(int c=0;c<ncat;c++) m+=bprop[c]*meanR[c];   // mean-1 base constraint Σ w·ρ (≈1 if getRate/getProp are in convention)
        if(m>0){ for(int c=0;c<ncat;c++) meanR[c]/=m; for(int v=0;v<nnodes;v++){ startB[v]*=m; if(startB[v]>20.0) startB[v]=20.0; } } }
    double lnL=evalLnL(startB,curAlpha,curPinv,nullptr); nLnLEval++;
    // ===== GATE-0.5 part 2: RATE-ARM FD CHECK (JOLT_RGRADCHECK_RATE, default-OFF, byte-identical when unset) =====
    // The planned OPG empirical-Fisher optimiser SQUARES g_y (the log-rate gradient); the shipped RGRADCHECK (:2907)
    // validates ONLY the weight arm (perturbs z). Mode-L, JOLT's predecessor, died on the RATE gradient (10^54). This
    // FD-checks g_y[c]=catRate[c]*gradR[c] (the shipping formula, :3361) against a central difference in y=log(meanR).
    // 🔴 The rate FD MUST re-evaluate via evalLnL (which does the FULL rebuildEchild+postorder — rates feed echild
    // len=brlen*catRate :2860). The weight FD's partials-frozen lnlW() shortcut would FALSE-PASS for rates. Gauge-free:
    // we perturb the unconstrained meanR; gaugeFix runs only on accept (:3431), so the base state here is pre-gauge.
    // Regime-aware gate: echild-rebuild + central-diff cancellation land ~1e-4 even when correct (blue-team recipe).
    if(freeRate==1 && getenv("JOLT_RGRADCHECK_RATE")){
        const double rgr_eps=1e-4;
        std::vector<double> rgr_meanR_save=meanR;
        double rgr_lg,rgr_ga; computeGradient(rgr_lg,rgr_ga);            // fills gradR[] + sets catRate=meanR/f at base
        std::vector<double> rgr_gy(ncat);
        for(int c=0;c<ncat;c++) rgr_gy[c]=catRate[c]*gradR[c];           // analytic d lnL/dy (the :3361 formula)
        double rgr_maxrel=0, rgr_maxabs=0; int rgr_worst=-1, rgr_nfail=0;
        for(int d=0; d<ncat; d++){
            double y0=log(meanR[d]);
            meanR[d]=exp(y0+rgr_eps); double lp=evalLnL(startB,curAlpha,curPinv,nullptr); nLnLEval++;
            meanR[d]=exp(y0-rgr_eps); double lm=evalLnL(startB,curAlpha,curPinv,nullptr); nLnLEval++;
            meanR[d]=rgr_meanR_save[d];
            double fd=(lp-lm)/(2.0*rgr_eps);
            double ad=fabs(rgr_gy[d]-fd);
            double rel=ad/(fabs(fd)+1e-30);
            if(rel>rgr_maxrel){ rgr_maxrel=rel; rgr_worst=d; }
            if(ad>rgr_maxabs) rgr_maxabs=ad;
            if(!((rel<1e-5)||(ad<1e-2))) rgr_nfail++;                     // per-cat HYBRID (blue-team): well-scaled OR near-zero g_y
            fprintf(stderr,"[RGRADRATE] c=%d meanR=%.6e w=%.6e g_y=%.6e FD=%.6e rel=%.3e abs=%.3e\n",
                    d, rgr_meanR_save[d], bprop[d], rgr_gy[d], fd, rel, ad);
        }
        fprintf(stderr,"[RGRADRATE] ncat=%d optPinv=%d pinv0=%.6f worst_c=%d maxrel=%.3e maxabs=%.3e nfail=%d -> %s\n",
                ncat, (int)optPinv, curPinv, rgr_worst, rgr_maxrel, rgr_maxabs, rgr_nfail, (rgr_nfail==0)?"RGRADRATE PASS":"RGRADRATE FAIL");
        fflush(stderr);
        meanR=rgr_meanR_save;
        lnL=evalLnL(startB,curAlpha,curPinv,nullptr); nLnLEval++;        // re-sync device to base for the LM loop
    }
    // ===== GATE-0.5 coverage: WEIGHT-OPT FD CHECK (JOLT_RGRADCHECK_WOPT, default-OFF, byte-identical when unset) =====
    // The shipped RGRADCHECK (:2907) validates a STANDALONE weight-grad reimpl with pinv HARDCODED 0 (:2916/:2928) and the
    // pure-+R formula gz=WN-w*N — it never exercises the OPTIMISER's real gzR (:3150, which the OPG will square) nor the
    // +I+R path (wnorm=sumWN, not rN). This FD-checks the REAL gzR via computeGradient, perturbing z=log(bprop) (the
    // optimiser's OWN softmax parameterisation, :3209) and re-evaluating via evalLnL at curPinv (so +I is live). bprop is
    // the PINV-FREE weight (applyPinv sets catProp_v=f*bprop :2847), so we perturb bprop — perturbing catProp_v would be
    // clobbered by evalLnL's applyPinv. At pinv=0 gzR==gz (part-1's validated formula) => this reproduces part 1 AND
    // extends it to +I+R. Regime-aware <1e-3 gate (same echild-rebuild/central-diff floor as the rate arm).
    if(freeRate==1 && getenv("JOLT_RGRADCHECK_WOPT")){
        const double wgr_eps=1e-4;
        std::vector<double> wgr_bprop_save=bprop;
        double wgr_sb=0; for(int c=0;c<ncat;c++) wgr_sb+=wgr_bprop_save[c];   // SEV-4a: FD centres at softmax(log bprop)==bprop only if Σ==1
        if(fabs(wgr_sb-1.0)>1e-9) fprintf(stderr,"[RGRADWOPT] WARN sum(bprop)=%.12f != 1 (base-point shift)\n",wgr_sb);
        double wgr_lg,wgr_ga; computeGradient(wgr_lg,wgr_ga);            // fills gzR[] (the real +I-aware weight grad :3150)
        std::vector<double> wgr_gz=gzR;                                  // snapshot analytic d lnL/dz
        auto wgr_softmax=[&](const std::vector<double>& z){ double mx=z[0]; for(int c=1;c<ncat;c++) if(z[c]>mx)mx=z[c];
            std::vector<double> o(ncat); double s=0; for(int c=0;c<ncat;c++){o[c]=exp(z[c]-mx); s+=o[c];} for(int c=0;c<ncat;c++)o[c]/=s; return o; };
        double wgr_maxrel=0, wgr_maxabs=0; int wgr_worst=-1, wgr_nfail=0;
        for(int d=0; d<ncat; d++){
            std::vector<double> zp(ncat),zm(ncat); for(int c=0;c<ncat;c++){ zp[c]=log(wgr_bprop_save[c]); zm[c]=zp[c]; }
            zp[d]+=wgr_eps; zm[d]-=wgr_eps;
            bprop=wgr_softmax(zp); double lp=evalLnL(startB,curAlpha,curPinv,nullptr); nLnLEval++;
            bprop=wgr_softmax(zm); double lm=evalLnL(startB,curAlpha,curPinv,nullptr); nLnLEval++;
            bprop=wgr_bprop_save;
            double fd=(lp-lm)/(2.0*wgr_eps);
            double ad=fabs(wgr_gz[d]-fd);                                 // absolute residual (round-off floor ~deltaL/eps, worse at 1M sites)
            double rel=ad/(fabs(fd)+1e-30);
            if(rel>wgr_maxrel){ wgr_maxrel=rel; wgr_worst=d; }
            if(ad>wgr_maxabs) wgr_maxabs=ad;
            if(!((rel<1e-5)||(ad<1e-2))) wgr_nfail++;                     // per-cat HYBRID: well-scaled (rel) OR near-zero gz (abs); a wnorm=rN +I bug => O(1e4)
            fprintf(stderr,"[RGRADWOPT] c=%d bprop=%.6e gz=%.6e FD=%.6e rel=%.3e abs=%.3e\n", d, wgr_bprop_save[d], wgr_gz[d], fd, rel, ad);
        }
        fprintf(stderr,"[RGRADWOPT] ncat=%d optPinv=%d pinv0=%.6f worst_c=%d maxrel=%.3e maxabs=%.3e nfail=%d -> %s\n",
                ncat, (int)optPinv, curPinv, wgr_worst, wgr_maxrel, wgr_maxabs, wgr_nfail, (wgr_nfail==0)?"RGRADWOPT PASS":"RGRADWOPT FAIL");
        fflush(stderr);
        bprop=wgr_bprop_save;
        lnL=evalLnL(startB,curAlpha,curPinv,nullptr); nLnLEval++;        // re-sync device to base for the LM loop
    }
    // ===== OPG Phase 1 GATE (JOLT_OPG_GRAMCHECK / JOLT_OPG_LMIN, default-OFF). NO step change in this phase. =====
    if(opgOK && (g_opg_gchk || g_opg_lmin)){
        const int N2=2*ncat;
        auto opg_soft=[&](const std::vector<double>& z){ double mx=z[0]; for(int c=1;c<ncat;c++) if(z[c]>mx)mx=z[c];
            std::vector<double> o(ncat); double s=0; for(int c=0;c<ncat;c++){o[c]=exp(z[c]-mx); s+=o[c];} for(int c=0;c<ncat;c++)o[c]/=s; return o; };
        // ONE capture sweep: fills opgH over all chunks AND (under GRAMCHECK) the analytic s_p + runs Check B.
        const int KP=16;
        opg_captP.assign((size_t)KP,0); for(int k=0;k<KP;k++) opg_captP[k]=(int)((long long)k*nptn/KP);
        opg_capt.assign((size_t)KP*N2,0.0);
        opg_captOn=true; { double lg,ga; computeGradient(lg,ga); } opg_captOn=false;
        std::vector<double> ana=opg_capt;
        if(g_opg_gchk){
            // ---- Check A (FORMULA, INDEPENDENT). Central-difference the per-pattern log-likelihood patlh[p]=log L_p
            // through the TILE-AWARE snapPatlh path (D2H is +pOff/Pn per chunk => nTile>1 is covered for free) and
            // compare to the analytic s_p. Check B restates the section-1 formula on the host, so it CANNOT catch a
            // wrong formula; this can. log L_p ~ O(1-10) => the FD floor here is ~1e-11 abs, ~1e5x sharper than the
            // aggregate gradient gate, so rel<1e-8 has real margin.
            std::vector<double> Pp((size_t)nptn),Pm((size_t)nptn);
            std::vector<double> mR=meanR, bP=bprop;
            double eps=1e-4;   // JOLT_OPG_EPS: 3-point ladder (1e-3/1e-4/1e-5) pins the FD's own error floor, which is
            if(const char* ee=getenv("JOLT_OPG_EPS")){ double v=atof(ee); if(v>0.0) eps=v; }   // what the Phase-3 lambda floor is derived from
            double wrel=0,wabs=0; int wc=-1,wk=-1,nb=0;
            for(int d=0; d<N2; d++){
                if(d<ncat){ double y0=log(mR[d]);
                    meanR[d]=exp(y0+eps); snapPatlh=Pp.data(); (void)evalLnL(startB,curAlpha,curPinv,nullptr); nLnLEval++;
                    meanR[d]=exp(y0-eps); snapPatlh=Pm.data(); (void)evalLnL(startB,curAlpha,curPinv,nullptr); nLnLEval++;
                    snapPatlh=nullptr; meanR=mR;
                } else { int j=d-ncat;
                    std::vector<double> zp(ncat),zm(ncat);
                    for(int c=0;c<ncat;c++){ zp[c]=log(bP[c]); zm[c]=zp[c]; }
                    zp[j]+=eps; zm[j]-=eps;
                    bprop=opg_soft(zp); snapPatlh=Pp.data(); (void)evalLnL(startB,curAlpha,curPinv,nullptr); nLnLEval++;
                    bprop=opg_soft(zm); snapPatlh=Pm.data(); (void)evalLnL(startB,curAlpha,curPinv,nullptr); nLnLEval++;
                    snapPatlh=nullptr; bprop=bP;
                }
                for(int k=0;k<KP;k++){
                    int p=opg_captP[k]; double fd=(Pp[(size_t)p]-Pm[(size_t)p])/(2.0*eps);
                    double an=ana[(size_t)k*(size_t)N2+(size_t)d];
                    double ad=fabs(an-fd), rel=ad/(fabs(fd)+1e-300);
                    // per-coord hybrid gate. abs floor 1e-5 sits ABOVE the central-diff truncation floor (~2.5e-7 at
                    // eps=1e-4, verified to scale as eps^2 on the eps-ladder => the analytic IS the true derivative) and
                    // ~1e4x-1e5x BELOW any real per-pattern bug (a wrong R_p at pinv=0.19 => O(0.01-0.1) abs). rel<1e-5
                    // catches large-score coords; abs<1e-5 rescues the many patterns whose per-coord score is ~0 (where
                    // rel is a meaningless 0/0). The old 1e-9 abs floor was 250x below truncation => flapped (Phase-1 v1).
                    if(!((rel<1e-5)||(ad<1e-5))) nb++;
                    if(rel>wrel && fabs(fd)>1e-12){ wrel=rel; wc=d; wk=k; }
                    if(ad>wabs) wabs=ad;
                }
            }
            meanR=mR; bprop=bP;
            fprintf(stderr,"[OPGGRAM-A] ncat=%d optPinv=%d pinv0=%.6f KP=%d coords=%d worst_coord=%d worst_k=%d worstrel=%.3e worstabs=%.3e nbad=%d -> %s\n",
                    ncat,(int)optPinv,curPinv,KP,N2,wc,wk,wrel,wabs,nb,(nb==0)?"GRAMA PASS":"GRAMA FAIL");
            fflush(stderr);
        }
        if(g_opg_lmin){
            // ---- lambda_min of the D-scaled reduced Gram. STEP space (2k-1): project ONLY the softmax null n_z (at
            // fixed brlen n_y is large-curvature and genuinely informative). DIAGNOSTIC space (2k-2): project BOTH
            // gauges -- identifiability is a question about the gauge-fixed manifold, and keeping n_y inflates
            // lambda_max, biasing the ratio toward "unidentifiable".
            std::vector<double> Hf((size_t)N2*N2,0.0);
            { int q=0; for(int a=0;a<N2;a++) for(int b=a;b<N2;b++){ Hf[(size_t)a*N2+b]=opgH[q]; Hf[(size_t)b*N2+a]=opgH[q]; q++; } }
            const int km=ncat-1;
            std::vector<double> C((size_t)ncat*km,0.0);           // Helmert: orthonormal basis of {v: sum v = 0}
            for(int j=0;j<km;j++){ int m=j+1; double d1=1.0/sqrt((double)m*((double)m+1.0));
                for(int i=0;i<=j;i++) C[(size_t)i*km+j]=d1;
                C[(size_t)m*km+j]=-(double)m*d1; }
            auto report=[&](const char* tag,bool projY){
                const int ry = projY?km:ncat, rz=km, nr=ry+rz;
                if(nr<1) return;
                std::vector<double> Q((size_t)N2*nr,0.0);
                for(int i=0;i<ncat;i++) for(int j=0;j<ry;j++) Q[(size_t)i*nr+j] = projY? C[(size_t)i*km+j] : ((i==j)?1.0:0.0);
                for(int i=0;i<ncat;i++) for(int j=0;j<rz;j++) Q[(size_t)(ncat+i)*nr+(size_t)(ry+j)] = C[(size_t)i*km+j];
                std::vector<double> T((size_t)N2*nr,0.0), Hr((size_t)nr*nr,0.0);
                for(int a=0;a<N2;a++) for(int j=0;j<nr;j++){ double s=0; for(int b=0;b<N2;b++) s+=Hf[(size_t)a*N2+b]*Q[(size_t)b*nr+j]; T[(size_t)a*nr+j]=s; }
                for(int i=0;i<nr;i++) for(int j=0;j<nr;j++){ double s=0; for(int a=0;a<N2;a++) s+=Q[(size_t)a*nr+i]*T[(size_t)a*nr+j]; Hr[(size_t)i*nr+j]=s; }
                std::vector<double> D((size_t)nr,0.0); double dmax=0;
                for(int i=0;i<nr;i++){ D[i]=Hr[(size_t)i*nr+i]; dmax=fmax(dmax,D[i]); }
                for(int i=0;i<nr;i++) D[i]=fmax(D[i],1e-12*dmax);
                for(int i=0;i<nr;i++) for(int j=0;j<nr;j++) Hr[(size_t)i*nr+j] /= sqrt(D[i]*D[j]);   // correlation matrix => dimensionless spectrum
                std::vector<double> ev((size_t)nr,0.0);
                jolt_jacobi_eig(Hr.data(),nr,ev.data());
                double lo=ev[0],hi=ev[0]; for(int i=1;i<nr;i++){ lo=fmin(lo,ev[i]); hi=fmax(hi,ev[i]); }
                fprintf(stderr,"[OPGLMIN] %-9s ncat=%d dim=%d lmin=%.6e lmax=%.6e ratio=%.6e\n",tag,ncat,nr,lo,hi,(hi>0.0?lo/hi:0.0));
            };
            report("step2k-1",false);
            report("diag2k-2",true);
            fflush(stderr);
        }
        if(opg_gram_n>0) fprintf(stderr,"[OPGCOST] ncat=%d NCH=%d gram_launches=%ld gram_sec=%.6f mean_ms=%.4f\n",
                                 ncat,opgNCH,opg_gram_n,opg_gram_sec,1000.0*opg_gram_sec/(double)opg_gram_n);
        fflush(stderr);
        lnL=evalLnL(startB,curAlpha,curPinv,nullptr); nLnLEval++;   // re-sync device to base for the LM loop
    }
    // T5 BISECTION PROBE (JOLT_IR3, default-OFF): the +I+R constant +10.00-nat joltLnL-vs-CPU offset (iplus 173879879:
    // jolt=cpu+10.00 exactly, mirror==cpu, only freeRate&&optPinv, pinv-invariant). Force maxiter=0 for freeRate&&optPinv
    // so gpu_jolt_optimize RETURNS THIS SEED lnL and writes back the SEED brlen/rates/props => DEVCHECK's cpuLnL is
    // computed at the SAME seed state. If joltLnL(seed)-cpuLnL(seed)=+10 the offset is in the SEED evalLnL (seeding of
    // catRate/catProp_v or the eval itself, reproducible without the LM loop); if =0 it ACCUMULATES in the LM loop
    // (gaugeFix/applyPinv on accept). Bisects loop-vs-seed in ONE build. No production effect (flag-gated).
    if (getenv("JOLT_IR3") && freeRate==1 && optPinv) maxiter = 0;
    double mu=1.0, tol=1e-7; int it=0,nRej=0; bool conv=false;
    double aPrev=0,gaPrev=0; bool haveSec=false;
    double pPrev=0,gpPrev=0;   // G.4.3b: pinv secant curvature (mirrors the alpha secant)
    // ---- L-BFGS state (brlen subvector, edge space) — JOLT_LBFGS_M>0; lbM=0 keeps the diagonal path byte-identical ----
    const int    lbM   = (freeRate==1) ? 0 : g_lbfgs_m;   // G.5.1b: +R always uses the diagonal path (the L-BFGS brlen direction does NOT carry the y/z arms)
    const double lbEps = 1e-9;                    // H0 = diag(1/(|g_ddf|+lbEps)); matches the diagonal-path mu floor
    std::vector<std::vector<double>> lbS, lbY;    // ring buffer of (s,y) pairs, each length nedge (newest at back)
    std::vector<double> lbRho;                    // 1/(y.s) per pair
    std::vector<double> prevB, prevG;             // previous accepted iterate (edge space) + its g_df, for the next pair
    std::vector<double> lbDir(lbM>0?nedge:0,0.0), lbQ(lbM>0?nedge:0,0.0), lbAlf;   // two-loop scratch
    // ②a +I+R WRITEBACK RECONCILIATION (grounded: JOB C irsweep 173890952 — JOLT reaches the true MLE, =pure-CPU MLE
    // -7541972.276 exact, but the state WRITTEN BACK is ~10 nats worse => self-check rel~1.3e-6 trips 1e-6 => CPU fallback;
    // mfgauge 173893718: gaugeFix INVARIANT d=0. VALIDATED job 173898475: -m MF worst|jolt-cpu| 10.00->1e-4 DNA+AA).
    // Root cause (Plan-agent + blue-team, source-verified): the pinv-FD (:3212-3215) calls evalLnL(base,baseA,baseP±1e-4)
    // whose applyPinv leaves catRate/catProp_v at the (1-baseP∓1e-4) scaling; when nFreeQ==0 nothing re-evals at baseP
    // before the baseR_save/baseW_save capture (:3238), so the reject-EXIT restore (:3333) writes props summing to
    // 1-1e-4 => ~1e-4*N_var ≈ 10-nat deficit at 100k sites. ⚠️ TRIGGER IS nFreeQ==0, NOT "AA": AA AND equal/empirical-freq
    // DNA (JC+I+R, F81+F+I+R) hit it; only FREE-Q DNA (HKY/TN/GTR) is clean because its Q-FD's last eval (:3226) re-evals
    // at baseP first. The corrupted catRate also feeds g_y=catRate·gradR (:3241), so the fix (before :3238) repairs both.
    // TWO default-OFF fixes, A/B'd by gems_irbestwb.sh; FDFIX is the graduation candidate (red-team: BESTWB alone fails
    // the zero-accept exit where bs_have==false; blue-team: BESTWB redundant once FDFIX lands):
    // GRADUATED DEFAULT-ON (job 173898475: -m MF gap 10->1e-4 DNA+AA, selection unchanged, DNA exact-MLE incl. avian
    // GTR+F+I+R4). Disable via JOLT_IR_NOFDFIX for A/B.
    // 🔴 MERGE 2026-07-17 -- JOLT_NO_PINVFIX IS KEPT AS A KILL-SWITCH ALIAS, DELIBERATELY. mfresident implemented
    // this same fix under the name JOLT_NO_PINVFIX, and the PASSING promotion gate (job 174010664,
    // gems_pinvfix_rfavor_euk.sh) both (a) asserts `strings|grep JOLT_NO_PINVFIX` as its proof-of-build and (b) uses
    // `JOLT_NO_PINVFIX=1` as its OFF arm. Dropping the name would have been WORSE THAN A COMPILE ERROR: the sentinel
    // check would false-FAIL a correct binary, and -- if that were "fixed" -- the OFF arm would silently become a
    // second ON arm, both arms would agree, and the gate would print SAME ✅ **having tested nothing**. A gate that
    // passes without testing anything is the worst failure mode available to us. The alias costs one getenv.
    // ⚠️ Retire this alias ONLY together with those scripts' sentinel + OFF-arm env, in the same commit.
    static const bool JOLT_IR_FDFIX_EN  = (getenv("JOLT_IR_NOFDFIX") == nullptr &&
                                           getenv("JOLT_NO_PINVFIX") == nullptr);
    static const bool JOLT_IR_BESTWB_EN = (getenv("JOLT_IR_BESTWB") != nullptr);   // defense-in-depth ONLY, default-OFF (red-team: INCOMPLETE — skips the zero-accept exit; never ship as the sole fix). snapshot best-accept, restore before writeback
    double bs_lnL=-1e300; bool bs_have=false; double bs_alpha=0.0, bs_pinv=0.0;
    std::vector<double> bs_brlen, bs_catRate, bs_catProp, bs_meanR, bs_bprop, bs_qcur;
    // ④ AA +I+R convergence gap — ACCEPTED, DOCUMENTED limitation (user decision 2026-07-15). JOLT's diagonal-LM optimum on
    // AA LG+I+R4 (-te = -7541972.346) sits 0.07 nats below the CPU MLE (-7541972.276) = 0.14 BIC, SELECTION-IRRELEVANT
    // (best-fit unchanged in every gate). It lives in this diagonal-LM's noisy FD model-arm secants (ddY/ddZ/ddP) + a shared
    // mu that ratchets 1e3->1e9 (CONVTRACE probe 173907382) — NOT ②a (the +I+R writeback fix, which stays graduated). The
    // curvFloor band-aid (κ·rN) was REVERTED: it fixed AA R4 but REGRESSED high-ncat +R (LG+I+R8 -0.41, R10 -2.6; gate
    // 173910972), and NO hardcoded constant generalizes across ncat/models/data. The principled parameter-free fix (OPG
    // empirical-Fisher — see research/Modelfinder/JOLT-OPG-FISHER-OPTIMIZER.md) is DEFERRED: not worth an optimizer redesign
    // unless the gap ever flips a REAL-DATA selection (untested). The arms below use fabs(dd) (byte-identical pre-④). The
    // JOLT_IR_TOL / FREEZE_MODEL / FREEZE_BRLEN / CONVTRACE flags are retained diagnostics (all default-OFF, byte-identical).
    if(const char* _e=getenv("JOLT_IR_TOL")) tol=atof(_e);                                                            // H-tol diagnostic
    const bool   JOLT_IR_FREEZE_MODEL = (getenv("JOLT_IR_FREEZE_MODEL") != nullptr);                                   // H-couple attribution: brlen+alpha only
    const bool   JOLT_IR_FREEZE_BRLEN = (getenv("JOLT_IR_FREEZE_BRLEN") != nullptr);                                   // H-couple attribution: model arms only
    const bool   JOLT_IR_CONVTRACE    = (getenv("JOLT_IR_CONVTRACE") != nullptr);                                      // H-diag: per-iter mu/rej/exit print
    for(it=1; it<=maxiter; it++){
        base=brlen; double baseA=curAlpha, baseP=curPinv; if(ncat>1 && !freeRate) applyAlpha(baseA);
        double lg,ga; computeGradient(lg,ga);
        // L-BFGS history: form the (s,y) pair for the prev->current ACCEPTED-iterate transition, then save current as prev.
        // ACTIVE-SET (red-team DEFECT B): exclude any edge at a brlen box bound at EITHER endpoint — its constrained
        // gradient would otherwise inject a spurious large y_e that corrupts the free-edge direction for up to lbM iters.
        if(lbM>0){
            if(!prevB.empty()){
                std::vector<double> s(nedge,0.0), y(nedge,0.0); double ys=0;
                for(int e=0;e<nedge;e++){ double bnew=base[edgeV[e]];
                    bool atBound = (bnew<=1e-6)||(bnew>=20.0)||(prevB[e]<=1e-6)||(prevB[e]>=20.0);
                    if(!atBound){ s[e]=bnew-prevB[e]; y[e]=prevG[e]-g_df[e]; ys+=s[e]*y[e]; } }
                if(ys>1e-12){   // curvature condition (SPD-preserving): push only a positively-curved pair
                    if((int)lbS.size()>=lbM){ lbS.erase(lbS.begin()); lbY.erase(lbY.begin()); lbRho.erase(lbRho.begin()); }
                    lbS.push_back(std::move(s)); lbY.push_back(std::move(y)); lbRho.push_back(1.0/ys); }
            }
            if((int)prevB.size()!=nedge){ prevB.assign(nedge,0.0); prevG.assign(nedge,0.0); }
            for(int e=0;e<nedge;e++){ prevB[e]=base[edgeV[e]]; prevG[e]=g_df[e]; }
        }
        // G.4.3b pinv gradient by FORWARD FINITE DIFFERENCE (robust to the rate<->prop<->pinv coupling that the
        // 1/(1-pinv) rate rescaling introduces; an analytic form would need the rate-derivative term). One extra
        // postorder lnL eval (cheap vs computeGradient's full preorder). lg = lnL at the base point (from above).
        double gradPinv=0.0;
        if(optPinv==1){ double ep=1e-4, pp=baseP+ep, dep;   // L6: ==1 only (optPinv==2 = +I apply-don't-step => pinv held FIXED, no gradient)
            if(pp>pinvMax){ pp=baseP-ep; if(pp<pinvMin)pp=pinvMin; }   // backward FD at the upper boundary (audit #4: avoid a stuck zero gradient)
            dep=pp-baseP;
            if(fabs(dep)>1e-9){ double lpe=evalLnL(base,baseA,pp,nullptr); nLnLEval++; gradPinv=(lpe-lg)/dep; } }
        // G.6 free-Q gradient by FORWARD FINITE DIFFERENCE (mirrors pinv; the CPU optimises Q by FD too, via BFGS).
        // Each free exchangeability: perturb in rate-class space (qApply re-decomposes the 4x4 Q + re-uploads), one
        // extra lnL eval. lg = base lnL. The device is left at base Q afterwards (qApply(qcur)).
        std::vector<double> gradQ(nFreeQ>0?nFreeQ:0,0.0), ddQ(nFreeQ>0?nFreeQ:0,-1e6);
        if(nFreeQ>0){
            std::vector<double> qp(qcur);
            for(int k=0;k<nFreeQ;k++){
                double save=qp[k], hq=1e-4*fabs(save); if(hq==0.0)hq=1e-4; double qpk=save+hq;
                if(qpk>MAXQ){ qpk=save-hq; if(qpk<MINQ)qpk=MINQ; }   // backward FD at the upper bound
                double dq=qpk-save;
                if(fabs(dq)>1e-12){ qp[k]=qpk; double lq=evalLnL(base,baseA,baseP,qp.data()); nLnLEval++; gradQ[k]=(lq-lg)/dq; qp[k]=save; }
            }
            qApply(qcur.data());   // restore the device eigensystem to base Q (the last FD eval left it at a perturbation)
        }
        // G.5.1b +R: log-rate / softmax-logit gradients + per-component secant curvature (mirrors the alpha ddA path).
        // g_y=r·gradR (chain rule for y=log r); g_z=gzR (softmax weight grad). haveSec is still the PRE-update flag here
        // (set true at the bottom of the iteration) so iteration 1 uses the -1e6 floor exactly like alpha. baseR/W saved
        // for the reject restore (the trial staging below overwrites meanR/bprop in place).
        std::vector<double> baseY(freeRate==1?ncat:0),baseZ(freeRate==1?ncat:0),g_y(freeRate==1?ncat:0),g_z(freeRate==1?ncat:0);
        std::vector<double> ddY(freeRate==1?ncat:0,-1e6),ddZ(freeRate==1?ncat:0,-1e6);
        if(freeRate==1){
            // ===== +I+R EXPORT-NORMALISATION FIX (default ON; kill-switch JOLT_IR_NOFDFIX restores old behaviour) =====
            // BUG (root-caused rdiag 173984235 + red-team, source-confirmed): the pinv forward-FD at :3220-3223 calls
            // evalLnL(..,pp,..) with pp=baseP+ep (ep=1e-4), whose applyPinv(pp) OVERWRITES catRate/catProp_v to the
            // pp-perturbed values and leaves them there. For a model with NO free exchangeabilities (nFreeQ==0, e.g. JC)
            // the Q-FD block at :3228 is skipped, so NOTHING resets catRate/catProp_v before the capture just below.
            // Then baseR_save/baseW_save capture the PERTURBED values, and a reject-exit (:3324-3328) restores + exports
            // them: out_props=(1-pp)*bprop (Σ=1-baseP-ep) with out_pinv=baseP => Σprop+pinv = 1-ep = 1-1e-4, i.e. the
            // exported +I+R model is UNDER-NORMALISED by exactly ep. That costs ~ep*Nsites nats (10 @100K / 100 @1M,
            // ncat-INDEPENDENT -- driven by ep, not by the gauge m), which is precisely the measured JC+I+R gap. The
            // gauge (catRate/=m, brlen*=m) is EXACTLY lnL-invariant and was NOT the cause (my earlier diagnosis; retracted).
            // FIX: re-derive catRate/catProp_v from the base pinv before capturing the base state. meanR/bprop are still
            // at base here (the FD's applyPinv reads them, never writes them), so applyPinv(baseP) reconstructs the exact
            // base catRate=meanR/(1-baseP), catProp_v=(1-baseP)*bprop -- Σprop+pinv=1. This also repairs the g_y gradient
            // just below (:3248 reads catRate). Zero GPU cost (host-only). optPinv!=1 (pure +R / fixed-pinv) never runs
            // the pinv-FD => catProp_v never perturbed => the guard makes this a no-op there => pure +R BYTE-IDENTICAL.
            // ===== MERGE RESOLUTION 2026-07-17 -- ONE fix, NOT a dedupe =====
            // mfdevcheck and mfresident independently implemented THIS SAME FIX with different names AND different
            // guards. Shipping the graduated NAME (JOLT_IR_FDFIX_EN / kill-switch JOLT_IR_NOFDFIX, job 173898475)
            // together with mfresident's GUARD (optPinv==1). Dropping either duplicate wholesale would have been a
            // real defect, in BOTH directions:
            //   * drop mfresident  -> ships the UNGUARDED form (applyPinv at optPinv==2 does an (x*(1-p))/(1-p)
            //     round-trip that is NOT byte-exact).
            //   * drop mfdevcheck -> loses the graduated, gate-validated name/default.
            //
            // 🔴 REACHABILITY -- CORRECTED 2026-07-17 (red-team). My first TWO justifications for this guard were
            // BOTH WRONG, in opposite directions. The code is right; neither reason I gave for it was:
            //   (a) "the merge makes optPinv==2 reachable via pureinvar's `optPinv = jolt_fixp ? 2 : 1`
            //       (phylotreegpu.cpp:2258)" -- true when written, but the SAME merge then demoted fixed-pinvar to
            //       opt-in (JOLT_FIXINVAR), so that route is now DEFAULT-OFF.
            //   (b) "load-bearing regardless, because brlen-only (:2274) reaches optPinv==2 too" -- FALSE.
            //       freeRateOK is defined `&& !brlenOnly` (:2204-2205) and the call site passes
            //       `freeRateOK ? 1 : (freeRateBrlenOK ? 3 : ...)` (:2405) => brlenOnly gives freeRate 3/2/0,
            //       NEVER 1 => it cannot enter this `if(freeRate==1)` block at all.
            // TRUTH: with fixed-pinvar opt-in, this guard is UNREACHABLE-BY-DEFAULT => shipped behaviour is exactly
            // the form job 173898475 validated. KEEP IT: one comparison, a strict improvement the moment
            // JOLT_FIXINVAR=1 is set or fixed-pinvar is re-graduated, and deleting it would silently re-arm the trap.
            // Documented as unreachable rather than justified by a reachability story that does not survive the source.
            // Guard semantics (mfresident's, verified): optPinv!=1 (pure +R / fixed-pinv) never runs the pinv-FD,
            // so catProp_v is never perturbed => applyPinv would be a no-op-in-intent but NOT byte-exact => guard
            // it out and pure +R / fixed-pinv stay BYTE-IDENTICAL.
            if(optPinv==1 && JOLT_IR_FDFIX_EN) applyPinv(baseP);
            baseR_save=catRate; baseW_save=catProp_v;
            // G.5.1d (2b): the log-rate arm lives in the PINV-FREE basis y=log(meanR=ρ) (the trial staging writes meanR=exp(y)).
            // g_y = d lnL/dy = meanR·dL/dmeanR = meanR·(gradR/(1-p)) = catRate·gradR (the (1-p) cancels). pure +R: meanR==catRate => byte-identical.
            for(int c=0;c<ncat;c++){ baseY[c]=log(meanR[c]); baseZ[c]=zR[c]; g_y[c]=catRate[c]*gradR[c]; g_z[c]=gzR[c]; }
            if(haveSec) for(int c=0;c<ncat;c++){
                if(fabs(baseY[c]-ryPrev[c])>1e-9) ddY[c]=(g_y[c]-rgyPrev[c])/(baseY[c]-ryPrev[c]);
                if(fabs(baseZ[c]-rzPrev[c])>1e-9) ddZ[c]=(g_z[c]-rgzPrev[c])/(baseZ[c]-rzPrev[c]); }
            for(int c=0;c<ncat;c++){ ryPrev[c]=baseY[c]; rgyPrev[c]=g_y[c]; rzPrev[c]=baseZ[c]; rgzPrev[c]=g_z[c]; }
        }
        double ddA=(haveSec && fabs(baseA-aPrev)>1e-9)?(ga-gaPrev)/(baseA-aPrev):-1e6;
        double ddP=(haveSec && fabs(baseP-pPrev)>1e-12)?(gradPinv-gpPrev)/(baseP-pPrev):-1e6;
        for(int k=0;k<nFreeQ;k++) ddQ[k]=(haveSec && fabs(qcur[k]-qPrev[k])>1e-12)?(gradQ[k]-gqPrev[k])/(qcur[k]-qPrev[k]):-1e6;
        aPrev=baseA; gaPrev=ga; pPrev=baseP; gpPrev=gradPinv;
        for(int k=0;k<nFreeQ;k++){ qPrev[k]=qcur[k]; gqPrev[k]=gradQ[k]; }
        haveSec=true;
        bool acc=false; double muIn=mu; int nRej0=nRej;   // ④ CONVTRACE: mu at iteration entry + rejects-this-iter
        if(lbM>0){
            // ---- L-BFGS brlen direction (computed ONCE; textbook step-length line search on t) — red-team DEFECT A fix:
            // a real step length t (NOT the mu ladder), so t*dir->0 always reaches acceptance for the SPD-ascent dir
            // (H_lbfgs is SPD when every stored y.s>0 and H0 SPD => g_df.dir>0 unless g_df==0, i.e. converged). ----
            if(!lbS.empty()){
                int K=(int)lbS.size(); lbAlf.assign(K,0.0);
                for(int e=0;e<nedge;e++) lbQ[e]=g_df[e];                                   // seed q=+g_df (= -grad f); dir=r (no final negation)
                for(int i=K-1;i>=0;i--){ const std::vector<double>& s=lbS[i]; double sdot=0; for(int e=0;e<nedge;e++) sdot+=s[e]*lbQ[e];
                    double a=lbRho[i]*sdot; lbAlf[i]=a; const std::vector<double>& y=lbY[i]; for(int e=0;e<nedge;e++) lbQ[e]-=a*y[e]; }
                for(int e=0;e<nedge;e++) lbDir[e]=lbQ[e]/(fabs(g_ddf[e])+lbEps);             // r = H0 . q
                for(int i=0;i<K;i++){ const std::vector<double>& y=lbY[i]; double ydot=0; for(int e=0;e<nedge;e++) ydot+=y[e]*lbDir[e];
                    double bb=lbRho[i]*ydot, ab=lbAlf[i]-bb; const std::vector<double>& s=lbS[i]; for(int e=0;e<nedge;e++) lbDir[e]+=ab*s[e]; }
            } else {
                for(int e=0;e<nedge;e++) lbDir[e]=g_df[e]/(fabs(g_ddf[e])+lbEps);            // empty history == the diagonal DIRECTION (lbEps-damped; iter-1 differs from the OFF mu-ladder by design)
            }
            // alpha/pinv/Q keep their diagonal+secant directions (computed once at the lbEps floor), scaled by the same t.
            double daB=(optAlpha && ncat>1)? ga/(fabs(ddA)+lbEps):0.0;
            double dpB=(optPinv==1)? gradPinv/(fabs(ddP)+lbEps):0.0;   // L6: pinv step only for ==1 (==2 holds fixed)
            std::vector<double> dqB(nFreeQ>0?nFreeQ:0,0.0);
            for(int k=0;k<nFreeQ;k++) dqB[k]=gradQ[k]/(fabs(ddQ[k])+lbEps);
            double t=1.0;
            for(int bt=0; bt<30; bt++){
                cand=base; for(int e=0;e<nedge;e++){ int v=edgeV[e]; double nb=base[v]+t*lbDir[e]; if(nb<1e-6)nb=1e-6; if(nb>20.0)nb=20.0; cand[v]=nb; }
                double ca=baseA; if(optAlpha && ncat>1){ ca=baseA+t*daB; if(ca<0.02)ca=0.02; if(ca>50.0)ca=50.0; }
                double cp=baseP; if(optPinv==1){ cp=baseP+t*dpB; if(cp<pinvMin)cp=pinvMin; if(cp>pinvMax)cp=pinvMax; }   // L6: ==2 => cp stays baseP (pinv fixed)
                std::vector<double> cq(nFreeQ>0?nFreeQ:0);
                for(int k=0;k<nFreeQ;k++){ double nq=qcur[k]+t*dqB[k]; if(nq<MINQ)nq=MINQ; if(nq>MAXQ)nq=MAXQ; cq[k]=nq; }
                double ln=evalLnL(cand,ca,cp, nFreeQ>0?cq.data():nullptr); nLnLEval++;
                if(ln>lnL+1e-9){ double dl=ln-lnL; brlen=cand; curAlpha=ca; curPinv=cp; if(nFreeQ>0) qcur=cq; lnL=ln; acc=true;
                    if(out_patlh&&nTile==1){ cudaMemcpy(out_patlh,d_patlh,(size_t)nptn*sizeof(double),cudaMemcpyDeviceToHost); snapDone=true; }   // STAGE 2b snapshot-on-accept: d_patlh holds THIS just-accepted trial (nTile==1 => all nptn), nothing overwrites it before break
                    if(dl<tol)conv=true; break; }
                else { t*=0.5; nRej++; } }
        } else {
            for(int bt=0; bt<14; bt++){
                cand=base; if(!JOLT_IR_FREEZE_BRLEN) for(int e=0;e<nedge;e++){ int v=edgeV[e]; double dn=fabs(g_ddf[e])+mu; double nb=base[v]+g_df[e]/dn; if(nb<1e-6)nb=1e-6; if(nb>20.0)nb=20.0; cand[v]=nb; }   // ④ FREEZE_BRLEN: hold brlen at base
                double ca=baseA; if(optAlpha && ncat>1 && !JOLT_IR_FREEZE_MODEL){ double da=ga/(fabs(ddA)+mu); ca=baseA+da; if(ca<0.02)ca=0.02; if(ca>50.0)ca=50.0; }
                double cp=baseP; if(optPinv==1 && !JOLT_IR_FREEZE_MODEL){ double dp=gradPinv/(fabs(ddP)+mu); cp=baseP+dp; if(cp<pinvMin)cp=pinvMin; if(cp>pinvMax)cp=pinvMax; }   // L6: ==2 => cp stays baseP
                std::vector<double> cq(nFreeQ>0?nFreeQ:0);
                for(int k=0;k<nFreeQ;k++){ double dn=fabs(ddQ[k])+mu; double nq=qcur[k]+gradQ[k]/dn; if(nq<MINQ)nq=MINQ; if(nq>MAXQ)nq=MAXQ; cq[k]=nq; }
                // G.5.1b +R: log-rate / softmax-weight arms at the SAME mu (mirror the alpha/pinv diagonal arms). Stage the
                // trial (cr,cw) into the PINV-FREE basis meanR/bprop so evalLnL's applyPinv(cp) (f=1-cp; ==1 for pure +R)
                // evaluates catRate=cr/(1-cp), catProp_v=(1-cp)cw — the +I-correct rates/props. cr lives in meanR=ρ space (baseY=log meanR).
                std::vector<double> cr,cw,cz;
                if(freeRate==1){ cr.resize(ncat); cw.resize(ncat); cz.resize(ncat);
                    bool remW = (JOLT_REM_EN && optPinv==0);   // L7 Stage B: EM weight M-step (pure +R only)
                    for(int c=0;c<ncat;c++){ double ny=baseY[c]+(JOLT_IR_FREEZE_MODEL?0.0:g_y[c]/(fabs(ddY[c])+mu)); double r=exp(ny);   // FREEZE_MODEL holds rate
                        if(r<1e-4)r=1e-4; if(r>1000.0)r=1000.0; cr[c]=r;
                        if(!remW) cz[c]=baseZ[c]+(JOLT_IR_FREEZE_MODEL?0.0:g_z[c]/(fabs(ddZ[c])+mu)); }   // gradient arm (OFF path / +I+R); FREEZE_MODEL holds weight
                    if(remW){ double wn=0; for(int c=0;c<ncat;c++) wn+=WNc[c];   // EM closed-form: w_c = WNc[c]/Σ WNc
                        double tt=0; for(int c=0;c<ncat;c++){ cw[c]=(wn>0.0? WNc[c]/wn : bprop[c]); if(cw[c]<1e-4)cw[c]=1e-4; tt+=cw[c]; }
                        for(int c=0;c<ncat;c++){ cw[c]/=tt; cz[c]=log(cw[c]); } }   // floor+renorm (== softmaxApply MIN_PROP); logit state for zR consistency
                    else softmaxApply(cz,cw);
                    for(int c=0;c<ncat;c++){ meanR[c]=cr[c]; bprop[c]=cw[c]; } }
                double ln=evalLnL(cand,ca,cp, nFreeQ>0?cq.data():nullptr); nLnLEval++;
                if(ln>lnL+1e-9){ double dl=ln-lnL; brlen=cand; curAlpha=ca; curPinv=cp; if(nFreeQ>0) qcur=cq;
                    // G.5.1d (2b): accept the staged pinv-free meanR/bprop, then DERIVE catRate/catProp_v via applyPinv(curPinv)
                    // (== cr/cw at pinv=0, so pure +R byte-identical), then gauge. (Old: catRate=cr direct — wrong under +I.)
                    if(freeRate==1){ for(int c=0;c<ncat;c++){ meanR[c]=cr[c]; bprop[c]=cw[c]; } zR=cz; applyPinv(curPinv); gaugeFix(); }
                    // GAUGE-INVARIANCE PROBE (JOLT_GAUGE_TRACE, default-OFF): `ln` was computed at the PRE-gauge state
                    // (cand, catRate=cr/(1-cp)); applyPinv+gaugeFix just re-expressed it (catRate/=m, brlen*=m) which is
                    // SUPPOSED to be lnL-invariant (rate*brlen preserved, +I term rate/brlen-independent). Re-eval at the
                    // JUST-GAUGED state and compare: d = ln_gauged - ln. d~=0 => gaugeFix invariant (drift is elsewhere);
                    // d~=-10 => gaugeFix corrupts THIS accept (the AA -te iriters 173889582 writeback -10). Dumps the
                    // normalizers (Sprop should be (1-pinv); SpropRate=m_post should be ~1 after the gauge). Measurement
                    // only (extra evalLnL); leaves device state at the gauged accept, which the next iter's base=brlen expects.
                    if (getenv("JOLT_GAUGE_TRACE") && freeRate==1 && optPinv==1) {
                        double ln_g = evalLnL(brlen, curAlpha, curPinv, nFreeQ>0?qcur.data():nullptr); nLnLEval++;
                        double sp=0, spr=0; for(int c=0;c<ncat;c++){ sp+=catProp_v[c]; spr+=catProp_v[c]*catRate[c]; }
                        fprintf(stderr,"[GAUGE] it=%d ln=%.6f ln_gauged=%.6f d=%.6f | Sprop=%.9f SpropRate=%.9f pinv=%.8f b_c0=%.6f cr0=%.6f cp0=%.6f\n",
                                it, ln, ln_g, ln_g-ln, sp, spr, curPinv, brlen[c0], catRate[0], catProp_v[0]);
                    }
                    lnL=ln; mu=fmax(mu*0.5,1e-9); acc=true;
                    if(out_patlh&&nTile==1){ cudaMemcpy(out_patlh,d_patlh,(size_t)nptn*sizeof(double),cudaMemcpyDeviceToHost); snapDone=true; }   // STAGE 2b snapshot-on-accept: d_patlh holds THIS just-accepted trial (nTile==1 => all nptn); host-only work above it (gaugeFix) never touches d_patlh
                    // ②a defensive snapshot: this accept is the post-gauge, self-consistent state (JOLT_GAUGE_TRACE d=0) that evaluates to ln=lnL. Capture it so loop exit writes back THIS state, not the pinv-FD-perturbed reject-EXIT restore.
                    if(JOLT_IR_BESTWB_EN && freeRate==1 && ln>bs_lnL){ bs_lnL=ln; bs_have=true;
                        bs_brlen=brlen; bs_alpha=curAlpha; bs_pinv=curPinv; bs_catRate=catRate; bs_catProp=catProp_v; bs_meanR=meanR; bs_bprop=bprop; if(nFreeQ>0) bs_qcur=qcur; }
                    if(dl<tol)conv=true; break; }
                else { mu*=4.0; nRej++; } }
        }
        if(JOLT_IR_CONVTRACE && freeRate==1) fprintf(stderr,"[IRCONV] it=%d muIn=%.3e muOut=%.3e rej_it=%d acc=%d conv=%d lnL=%.6f exit=%s\n", it, muIn, mu, nRej-nRej0, acc?1:0, conv?1:0, lnL, acc?(conv?"CONV(dl<tol)":"accept"):"REJECT-EXIT");   // ④ H-diag
        if(!acc){ brlen=base; curAlpha=baseA; curPinv=baseP;
            // G.5.1d (2b): restore catRate/catProp_v AND the pinv-free basis. baseR_save/baseW_save are catRate/catProp_v at
            // base; meanR=ρ=catRate·(1-baseP), bprop=w=catProp_v/(1-baseP). f==1 (pure +R) => meanR=baseR_save (byte-identical).
            if(freeRate==1){ catRate=baseR_save; catProp_v=baseW_save; double f=optPinv?(1.0-baseP):1.0;
                for(int c=0;c<ncat;c++){ meanR[c]=baseR_save[c]*f; bprop[c]=baseW_save[c]/f; } }
            if(nFreeQ>0) qApply(qcur.data()); break; }
        if(conv) break; }

    // ②a RETRACTED (JOB C irsweep 173890952): the old JOLT_IR_REEVAL re-eval was HARMFUL — it re-scored the DRIFTED
    // written-back state (props sum 1-1e-4, root cause :3227/:3322) and DEGRADED the reported optimum to MLE-10 (AA
    // -7541982 vs the true MLE -7541972 that JOLT already reaches). REMOVED. The +I+R writeback is repaired at the root
    // (JOLT_IR_FDFIX @ the pinv-FD capture) and/or by restoring the best-accept snapshot here (JOLT_IR_BESTWB, default-OFF).
    if(JOLT_IR_BESTWB_EN && freeRate==1 && bs_have){   // restore best-accept {brlen,alpha,pinv,catRate,catProp_v,meanR,bprop,qcur} so out_* == the params that scored joltLnL (the MLE) => CPU self-check passes at the MLE
        brlen=bs_brlen; curAlpha=bs_alpha; curPinv=bs_pinv; catRate=bs_catRate; catProp_v=bs_catProp; meanR=bs_meanR; bprop=bs_bprop; if(nFreeQ>0) qcur=bs_qcur; }

    if (cudaGetLastError()!=cudaSuccess) return (double)NAN;   // any launch/sync error -> caller falls back to CPU

    // ===== +I+R EXPORT-NORMALISATION VERIFY (JOLT_RDIAG diagnostic; NO effect on lnL or params) =====
    // The real fix lives at the base-state capture above (JOLT_NO_PINVFIX). Here we only VERIFY the invariant the bug
    // violated: for a +I+R model the exported weights + pinv must sum to 1. Red-team's exposing measurement -- pre-fix
    // this reads ~1-1e-4 on a reject-terminal JC+I+R; post-fix it must read 1.0. Print-only, gated, side-effect-free.
    // (My earlier re-evaluate-and-return "fix" here was REMOVED: it made joltLnL==cpuLnL at the WRONG (under-normalised)
    //  point, shipping an invalid model AND suppressing the CPU-fallback safety net. The normalisation fix makes the
    //  self-check pass on a CORRECT model, so no re-eval is needed.)
    if (freeRate==1 && getenv("JOLT_RDIAG")) {
        double sp = optPinv ? curPinv : 0.0;
        for(int c=0;c<ncat;c++) sp += catProp_v[c];
        fprintf(stderr, "[RDIAG-GPU] ncat=%d optPinv=%d curPinv=%.6f  Sum(props)+pinv=%.10f  (must be 1.0)  clamp=%ld\n",
                ncat, optPinv, curPinv, sp, rd_clamps);
        fflush(stderr);
    }

    for(int v=0;v<nnodes;v++) out_brlen[v]=brlen[v];
    if(out_alpha) *out_alpha=curAlpha;
    if(out_pinv)  *out_pinv = optPinv ? curPinv : pinv0;
    if(nFreeQ>0 && out_q) for(int k=0;k<nFreeQ;k++) out_q[k]=qcur[k];
    if(freeRate==1 && out_rates && out_props) for(int c=0;c<ncat;c++){ out_rates[c]=catRate[c]; out_props[c]=catProp_v[c]; }   // G.5.1b: optimised +R rates/weights
    if(out_iters) *out_iters=it;
    // --jolt-diag (A3) + LINE-SEARCH WASTE (red-team C1, 2026-06-26): nRej = rejected backtracks (each a WASTED full
    // postorder over all internal nodes, discarded), nLnLEval = total evalLnL postorders. reject_frac = nRej/nLnLEval
    // measures the line-search-efficiency lever (the 2nd exact-ish surface on the 94% partials, distinct from maxiter).
    if(g_jdiag) printf("JOLT-DIAG-CU echild=%.6f n=%ld iters=%d nRej=%d nLnLEval=%ld nptn=%d\n", g_jd_echild_sec-_jd_ech0, g_jd_echild_n-_jd_echn0, it, nRej, nLnLEval, nptn);
    // Inc 2 S6: direct proof the per-edge reopt constant-memory storm collapses under ON. OFF: memcpyToSymbol=NNN,
    // valpool_async=0. ON: memcpyToSymbol=0, valpool_async=NNN (the 2.6M storm -> 0). JOLT_DEBUG-gated.
    if(getenv("JOLT_DEBUG")) fprintf(stderr,"[TS-ASYNC] reopt memcpyToSymbol=%ld valpool_async=%ld\n", ts_reopt_mcs, ts_reopt_vp);
    if (g_kcount) { g_kc_reopt_calls++;
        fprintf(stderr,"[TS-KCOUNT reopt] call=%ld reopt_part_launches(cum)=%ld | cum: scr_base=%ld scr_upper=%ld scr_fold=%ld reopt_part=%ld -> shareable(base+upper+reopt-rebuild) vs FOLD(necessary)\n",
                g_kc_reopt_calls,g_kc_reopt_part,g_kc_scr_base,g_kc_scr_upper,g_kc_scr_fold,g_kc_reopt_part); }

    // STAGE 2b: robust fallback snapshot. Reached only when out_patlh != nullptr (-B + opt-in) AND the cheap
    // snapshot-on-accept above did NOT fire (snapDone false) — i.e. nTile>1 (accept only captured the last tile), OR
    // no LM step was ever accepted (out_patlh unwritten), OR the L-BFGS path is active. Re-eval the ACCEPTED params
    // (brlen/curAlpha/curPinv/qcur are the accepted values at this point, restored on a reject exit) with snapPatlh
    // set, so evalLnL's per-tile D2H fills out_patlh over ALL tiles from the accepted tree. Warm (reuses the reopt's
    // echild/partial/prepool arenas) — no separate clean-room rebuild. lnL (the return value) is left unchanged.
    if (out_patlh && !snapDone) {
        snapPatlh = out_patlh;
        (void)evalLnL(brlen, curAlpha, curPinv, nFreeQ>0 ? qcur.data() : nullptr);
        snapPatlh = nullptr;
        if (cudaGetLastError()!=cudaSuccess) return (double)NAN;
    }
    return lnL;
}
