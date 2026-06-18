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
#include <cstdlib>      // G.7.1: getenv/atoi for JOLT_NTILE

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
        const double* ec2, const double* p2, const unsigned char* t2) {
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
    if (isRoot) patlh[ptn] = log(fabs(lh));
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
        int nchild,
        const double* ec0, const double* p0, const unsigned char* t0,
        const double* ec1, const double* p1, const unsigned char* t1,
        const double* ec2, const double* p2, const unsigned char* t2) {
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    int R = nmix*ncat;
    double lh = 0.0, clh = 0.0;   // clh = running per-class accumulator (G.8.1)
    for (int r=0;r<R;r++){
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
#define DEVB(b, bytes) do{ if(!devbuf_ensure((b),(size_t)(bytes))){ \
    fprintf(stderr,"[GPU] devbuf_ensure failed (%zu bytes) at %s:%d\n",(size_t)(bytes),__FILE__,__LINE__); \
    return (double)NAN; } }while(0)

extern "C" double gpu_lnl_crosscheck(
    int nstates, int nptn, int ncat, int ntax, int nnodes, int nInternal,
    const double* Uinv, const double* UinvRowSum, const double* freq, const double* catProp,
    const double* echild, const unsigned char* tip, const double* ptn_freq,
    const int* desc_isRoot, const int* desc_nchild, const int* desc_outSlot,
    const int* desc_childNode, const int* desc_childIsLeaf, const int* desc_childLeaf, const int* desc_childSlot,
    double* out_patlh)
{
    int ns = nstates;
    if (ns > NS_MAX || ncat > 64) { fprintf(stderr,"[GPU-XCHECK] unsupported ns=%d ncat=%d\n",ns,ncat); return (double)NAN; }

    GCK(cudaMemcpyToSymbol(g_Uinv, Uinv, sizeof(double)*ns*ns));
    GCK(cudaMemcpyToSymbol(g_UinvRowSum, UinvRowSum, sizeof(double)*ns));
    GCK(cudaMemcpyToSymbol(g_freq, freq, sizeof(double)*ns));
    GCK(cudaMemcpyToSymbol(g_catw, catProp, sizeof(double)*ncat));

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
        k1_node<<<GB,TB>>>(ns,nptn,ncat,isRoot,out,d_patlh,nchild,
            ec[0],p[0],t[0], ec[1],p[1],t[1], ec[2],p[2],t[2]);
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
extern "C" double gpu_lnl_crosscheck_mix(
    int nstates, int nptn, int ncat, int nmix, int ntax, int nnodes, int nInternal,
    const double* Uinv, const double* UinvRowSum, const double* freq, const double* wreg,
    const double* echild, const unsigned char* tip, const double* ptn_freq,
    const int* desc_isRoot, const int* desc_nchild, const int* desc_outSlot,
    const int* desc_childNode, const int* desc_childIsLeaf, const int* desc_childLeaf, const int* desc_childSlot,
    double* out_patlh, double* out_lhcat)   // G.8.1: out_lhcat (optional) = per-class L_{p,m}, [nmix][nptn]
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

    size_t ecStride = (size_t)R*ns*ns;        // echild per node, regime-strided
    size_t slotSz   = (size_t)R*ns*nptn;      // partial per internal slot, regime-strided
    DEVB(gb_echild, (size_t)nnodes*ecStride*sizeof(double));
    DEVB(gb_tip,    (size_t)ntax*nptn);
    DEVB(gb_partial,(size_t)(nInternal>0?nInternal:1)*slotSz*sizeof(double));
    DEVB(gb_patlh,  (size_t)nptn*sizeof(double));
    double *d_echild=(double*)gb_echild.p, *d_partial=(double*)gb_partial.p, *d_patlh=(double*)gb_patlh.p;
    unsigned char *d_tip=(unsigned char*)gb_tip.p;
    double *d_lhcat = nullptr;
    if (out_lhcat) { DEVB(gb_mLhcat, (size_t)nmix*nptn*sizeof(double)); d_lhcat=(double*)gb_mLhcat.p; }   // G.8.1
    GCK(cudaMemcpy(d_echild, echild, (size_t)nnodes*ecStride*sizeof(double), cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(d_tip, tip, (size_t)ntax*nptn, cudaMemcpyHostToDevice));

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
        k1_node_mix<<<GB,TB>>>(ns,nptn,ncat,nmix,isRoot,out,d_patlh,d_Uinv,d_Urs,d_Freq,d_Wreg,d_lhcat,nchild,
            ec[0],p[0],t[0], ec[1],p[1],t[1], ec[2],p[2],t[2]);
    }
    GCK(cudaDeviceSynchronize());
    GCK(cudaGetLastError());

    std::vector<double> patlh(nptn);
    GCK(cudaMemcpy(patlh.data(), d_patlh, (size_t)nptn*sizeof(double), cudaMemcpyDeviceToHost));
    if (out_patlh) for (int p2=0; p2<nptn; p2++) out_patlh[p2] = patlh[p2];
    if (out_lhcat) GCK(cudaMemcpy(out_lhcat, d_lhcat, (size_t)nmix*nptn*sizeof(double), cudaMemcpyDeviceToHost));  // G.8.1
    double lnL=0.0, kc=0.0;
    for (int p2=0; p2<nptn; p2++){ double term = ptn_freq[p2]*patlh[p2];
        double y=term-kc, t2=lnL+y; kc=(t2-lnL)-y; lnL=t2; }
    return lnL;
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

    GCK(cudaMemcpyToSymbol(g_Uinv, Uinv, sizeof(double)*ns*ns));
    GCK(cudaMemcpyToSymbol(g_UinvRowSum, UinvRowSum, sizeof(double)*ns));
    GCK(cudaMemcpyToSymbol(g_freq, freq, sizeof(double)*ns));
    GCK(cudaMemcpyToSymbol(g_catw, catProp, sizeof(double)*ncat));

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
        k1_node<<<GB,TB>>>(ns,nptn,ncat,/*isRoot=*/0,out,d_patlh,nchild,
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

// JOLT-specific persistent device buffers (separate from the lnL/derv pools; same alloc-once / reuse policy).
static DevBuf gbj_echild, gbj_partial, gbj_patlh, gbj_pdf, gbj_pddf,
              gbj_pretmp, gbj_tipeig, gbj_prepool, gbj_expfac, gbj_rnum, gbj_tip, gbj_baseinvar,
              gbj_ptnfreq, gbj_redpart,   // G.5.0: on-device ptn_freq + per-block reduction partials
              gbj_invlbase, gbj_redR;     // G.5.0 Part B: base-edge 1/L_p + per-category gradR partials

extern "C" double gpu_jolt_optimize(
    int nstates, int nptn, int ncat, int ntax, int nnodes, int root,
    const double* Uinv, const double* UinvRowSum, const double* U, const double* eval,
    const double* catProp, const unsigned char* tip, const double* ptn_freq,
    const int* node_nchild, const int* node_child, const int* node_leaf, const double* node_parentLen,
    double alpha0, int optAlpha, int maxiter,
    const double* base_invar, double pinv0, int optPinv, double pinvMin, double pinvMax,
    const double* catRate0, int freeRate,   // G.5.1: +R FreeRate — catRate0=rates[c] (else nullptr); freeRate=1 seeds rates directly (no alpha)
    int nFreeQ, const double* q0, jolt_qdecompose_fn qdecompose, void* qctx, double* out_q,   // G.6: DNA free-Q (eigensystem moves)
    double* out_brlen, double* out_alpha, double* out_pinv, int* out_iters)
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

    // alpha-independent eigen constants — upload once (the BASE-Q eigensystem). For free-Q (nFreeQ>0) qApply()
    // re-uploads these whenever an exchangeability changes; for fixed-Q this is the only upload.
    GCK(cudaMemcpyToSymbol(g_Uinv, Uinv, sizeof(double)*ns*ns));
    GCK(cudaMemcpyToSymbol(g_U,    U,    sizeof(double)*ns*ns));
    GCK(cudaMemcpyToSymbol(g_UinvRowSum, UinvRowSum, sizeof(double)*ns));

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
        cudaMemcpyToSymbol(g_Uinv, UinvB.data(), sizeof(double)*ns*ns);
        cudaMemcpyToSymbol(g_U,    UB.data(),    sizeof(double)*ns*ns);
        double rs[NS_MAX]; for(int i=0;i<ns;i++){ double s=0; for(int j=0;j<ns;j++) s+=UinvB[i*ns+j]; rs[i]=s; }
        cudaMemcpyToSymbol(g_UinvRowSum, rs, sizeof(double)*ns); };

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
    if (freeRate) nTile = 1;   // +R declines to CPU before the optimise loop (RGRADCHECK uses full-nptn buffers); no tiling
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
    auto setChunk=[&](int t){
        int p0=t*chunk0, p1=p0+chunk0; if(p1>nptn)p1=nptn; int cw=p1-p0;
        Pn=cw; pOff=p0; slotSz=(size_t)ncat*ns*cw; GB=(cw+TB-1)/TB;
        for(int a=0;a<ntax;a++) memcpy(&tipChunk[(size_t)a*cw], tip+(size_t)a*nptn+p0, (size_t)cw);
        cudaMemcpy(d_tip, tipChunk.data(), (size_t)ntax*cw, cudaMemcpyHostToDevice);
        cudaMemcpy(d_ptnfreq, ptn_freq+p0, (size_t)cw*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_baseinvar, biFull.data()+p0, (size_t)cw*sizeof(double), cudaMemcpyHostToDevice);
    };

    DEVB(gbj_redpart, (size_t)3*GBmax*sizeof(double));   // G.5.0: 3 channels x GBmax per-block partial sums
    double* d_redpart=(double*)gbj_redpart.p;
    std::vector<double> h_redpart((size_t)3*GBmax);
    DEVB(gbj_invlbase, (size_t)chunk0*sizeof(double)); double* d_invLbase=(double*)gbj_invlbase.p;  // G.5.0 Part B
    DEVB(gbj_redR,     (size_t)ncat*GBmax*sizeof(double)); double* d_redR=(double*)gbj_redR.p;
    std::vector<double> h_redR((size_t)ncat*GBmax);
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
    // G.5.1 +R FreeRate: seed meanR=rates[c], bprop=weights so applyPinv(0) is identity (catRate=rates, catProp_v=weights).
    if (freeRate) { for(int c=0;c<ncat;c++){ meanR[c]=catRate0[c]; bprop[c]=catProp[c]; } }
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
        for(int c=0;c<nnodes;c++){ if(c==root){ for(size_t z=0;z<ecStride;z++) h_echild[(size_t)c*ecStride+z]=0.0; continue; }
            for(int cat=0;cat<ncat;cat++){ double len=brlen[c]*catRate[cat]; double ex[NS_MAX]; for(int i=0;i<ns;i++) ex[i]=exp(evalP[i]*len);
                double* e=&h_echild[(size_t)c*ecStride+(size_t)cat*ns*ns]; for(int x=0;x<ns;x++) for(int i=0;i<ns;i++) e[x*ns+i]=UP[x*ns+i]*ex[i];
                for(int i=0;i<ns;i++) h_expfac[(size_t)c*ncat*ns+cat*ns+i]=ex[i]; } }
        cudaMemcpy(d_echild,h_echild.data(),(size_t)nnodes*ecStride*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(d_expfac,h_expfac.data(),(size_t)nnodes*ncat*ns*sizeof(double),cudaMemcpyHostToDevice); };
    auto postorderFill=[&](){
        for(int idx=0; idx<nInternal; idx++){ int u=postorder[idx]; if(u==root) continue;
            int nch; const double* ec[3]; const double* p[3]; const unsigned char* t[3]; childArgs(u,-1,nch,ec,p,t);
            k1_node<<<GB,TB>>>(ns,Pn,ncat,0,d_partial+(size_t)slot[u]*slotSz,d_patlh,nch,ec[0],p[0],t[0],ec[1],p[1],t[1],ec[2],p[2],t[2]); }
        cudaDeviceSynchronize(); };
    auto setVal=[&](double t){ std::vector<double> v0(ncat*ns),v1(ncat*ns),v2(ncat*ns);
        for(int c=0;c<ncat;c++){ double rc=catRate[c],pcw=catProp_v[c]; for(int x=0;x<ns;x++){ double re=rc*evalP[x],e=exp(evalP[x]*rc*t)*pcw;
            v0[c*ns+x]=e; v1[c*ns+x]=re*e; v2[c*ns+x]=re*re*e; } }
        cudaMemcpyToSymbol(g_val0,v0.data(),sizeof(double)*ncat*ns); cudaMemcpyToSymbol(g_val1,v1.data(),sizeof(double)*ncat*ns); cudaMemcpyToSymbol(g_val2,v2.data(),sizeof(double)*ncat*ns); };
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
            k1_node<<<GB,TB>>>(ns,nptn,ncat,0,d_pretmp,d_patlh,nch,ec[0],p[0],tp[0],ec[1],p[1],tp[1],ec[2],p[2],tp[2]); cudaDeviceSynchronize();
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
        return (double)NAN;   // G.5.1a: +R optimiser branch not yet wired -> decline to CPU after the (optional) check
    }

    long nGradSweeps=0,nLnLEval=0;
    // part8 #2 base-sweep skip: record the (brlen,alpha,pinv) the device echild/partial were last built for (by
    // evalLnL) so computeGradient can skip the redundant rebuild+postorder when its base already matches. Values are
    // COPIES of the same candidate vectors (no recompute) => exact == is reliable; any mismatch falls back to rebuild.
    std::vector<double> devB; double devA=1e300, devP=1e300; bool devValid=false;
    auto evalLnL=[&](const std::vector<double>& cand_b,double cand_a,double cand_pinv,const double* cand_q)->double{
        if(nFreeQ>0 && cand_q) qApply(cand_q);   // G.6: re-decompose+reupload the trial Q -> rebuildEchild() below uses the new evalP/UP
        if(ncat>1) applyAlpha(cand_a); applyPinv(cand_pinv); brlen=cand_b;
        rebuildEchild();   // G.7.1: chunk-INDEPENDENT (echild/expfac carry no nptn) — build once per eval, reused across all chunks
        double Lacc=0,Lk=0;   // G.7.1: Kahan accumulator of lnL over the pattern chunks (exact additivity, rel<=1e-12 vs one-shot)
        for(int t=0;t<nTile;t++){
            setChunk(t); postorderFill();
            int nch; const double* ec[3]; const double* p[3]; const unsigned char* tp[3]; childArgs(root,c0,nch,ec,p,tp);
            k1_node<<<GB,TB>>>(ns,Pn,ncat,0,d_pretmp,d_patlh,nch,ec[0],p[0],tp[0],ec[1],p[1],tp[1],ec[2],p[2],tp[2]); cudaDeviceSynchronize();
            const double* pl0=edgeNodePtr(c0); cudaDeviceSynchronize();   // part8 #3: fused — no theta materialisation
            setVal(brlen[c0]); kj_derv_fused<<<GB,TB>>>(ns,Pn,ncat,pl0,d_pretmp,cand_pinv,d_baseinvar,d_patlh,d_pdf,d_pddf,nullptr,nullptr); cudaDeviceSynchronize();
            double l,d,dd; reduceDerv(l,d,dd);
            double y=l-Lk, s=Lacc+y; Lk=(s-Lacc)-y; Lacc=s;   // Kahan add this chunk's lnL contribution
        }
        devB=cand_b; devA=cand_a; devP=cand_pinv; devValid=true;   // part8 #2: echild matches this base (full postorder present on device only when nTile==1)
        return Lacc; };

    std::vector<double> g_df(nedge,0.0),g_ddf(nedge,0.0),gradR(ncat,0.0);   // G.5.0 Part B: invL/rnumH now on-device
    auto computeGradient=[&](double& lnLout,double& galphaOut){
        applyPinv(curPinv);   // G.4.3b: align catRate=meanR/(1-curPinv) and catProp_v to the base pinv before the sweep
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
        double Lacc=0,Lk=0;
        for(int t=0;t<nTile;t++){
            setChunk(t);
            if(!postValid) postorderFill();
            cudaMemset(d_rnum,0,(size_t)ncat*Pn*sizeof(double));
            std::vector<int> freeSlots; for(int s=nPool-1;s>=0;s--) freeSlots.push_back(s);
            auto acq=[&](){int s=freeSlots.back();freeSlots.pop_back();return s;}; auto rls=[&](int s){freeSlots.push_back(s);};
            std::vector<double> dfC(nnodes,0.0),ddfC(nnodes,0.0); bool gotL=false; double lnLfirst=0;
            std::function<void(int,int)> proc=[&](int u,int su){
                for(int v:child[u]){
                    int sv=acq(); double* pre=d_prepool+(size_t)sv*slotSz;
                    if(u==root){ int nch; const double* ec[3]; const double* p[3]; const unsigned char* tp[3]; childArgs(root,v,nch,ec,p,tp);
                        k1_node<<<GB,TB>>>(ns,Pn,ncat,0,pre,d_patlh,nch,ec[0],p[0],tp[0],ec[1],p[1],tp[1],ec[2],p[2],tp[2]); }
                    else { const double* ec[2]={0,0}; const double* sp[2]={0,0}; const unsigned char* st[2]={0,0}; int nsb=0;
                        for(int w:child[u]){ if(w==v||nsb>=2) continue; sibArg(w,ec[nsb],sp[nsb],st[nsb]); nsb++; }
                        kj_pre<<<GB,TB>>>(ns,Pn,ncat,pre,d_prepool+(size_t)su*slotSz,d_expfac+(size_t)u*ncat*ns,nsb,ec[0],sp[0],st[0],ec[1],sp[1],st[1]); }
                    cudaDeviceSynchronize();
                    const double* plv=edgeNodePtr(v);   // part8 #3: fused theta+derv+ratenum, no d_theta round-trip
                    double bv=brlen[v]; std::vector<double> rs(ncat); for(int c=0;c<ncat;c++) rs[c]=bv/(catRate[c]*catProp_v[c]);
                    cudaMemcpyToSymbol(g_rscale,rs.data(),sizeof(double)*ncat); setVal(bv); cudaDeviceSynchronize();
                    kj_derv_fused<<<GB,TB>>>(ns,Pn,ncat,plv,pre,curPinv,d_baseinvar,d_patlh,d_pdf,d_pddf,d_rnum,nullptr); cudaDeviceSynchronize();
                    double l,d,dd; reduceDerv(l,d,dd); dfC[v]=d; ddfC[v]=dd;
                    if(!gotL){ lnLfirst=l;
                        // G.5.0 Part B: 1/L_p on-device from the base-edge patlh (was a host D2H + exp loop over nptn).
                        kj_invl<<<GB,TB>>>(Pn,d_patlh,d_invLbase); gotL=true; }
                    if(leaf[v]<0) proc(v,sv); rls(sv);
                } };
            proc(root,-1); cudaDeviceSynchronize();
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
        }
        for(int e=0;e<nedge;e++){ g_df[e]=accDf[e]; g_ddf[e]=accDdf[e]; }
        for(int c=0;c<ncat;c++) gradR[c]=catProp_v[c]*accR[c];
        double ga=0;
        // alpha gradient: ga = Σ_c (d catRate[c]/dα)·gradR[c]; catRate[c]=meanR[c]/f so the perturbed mean-1 rate
        // rp[c] must be scaled by 1/f too (else mixing scaled/unscaled rates -> wrong alpha grad on the +I path).
        if(ncat>1){ double f = optPinv ? (1.0-curPinv) : 1.0; double rp[64]; jolt_discreteGammaMean(curAlpha+1e-5,ncat,rp);
            for(int c=0;c<ncat;c++) ga+=((rp[c]/f-catRate[c])/1e-5)*gradR[c]; }
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
    double lnL=evalLnL(startB,curAlpha,curPinv,nullptr); nLnLEval++;
    double mu=1.0, tol=1e-7; int it=0,nRej=0; bool conv=false;
    double aPrev=0,gaPrev=0; bool haveSec=false;
    double pPrev=0,gpPrev=0;   // G.4.3b: pinv secant curvature (mirrors the alpha secant)
    for(it=1; it<=maxiter; it++){
        base=brlen; double baseA=curAlpha, baseP=curPinv; if(ncat>1) applyAlpha(baseA);
        double lg,ga; computeGradient(lg,ga);
        // G.4.3b pinv gradient by FORWARD FINITE DIFFERENCE (robust to the rate<->prop<->pinv coupling that the
        // 1/(1-pinv) rate rescaling introduces; an analytic form would need the rate-derivative term). One extra
        // postorder lnL eval (cheap vs computeGradient's full preorder). lg = lnL at the base point (from above).
        double gradPinv=0.0;
        if(optPinv){ double ep=1e-4, pp=baseP+ep, dep;
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
        double ddA=(haveSec && fabs(baseA-aPrev)>1e-9)?(ga-gaPrev)/(baseA-aPrev):-1e6;
        double ddP=(haveSec && fabs(baseP-pPrev)>1e-12)?(gradPinv-gpPrev)/(baseP-pPrev):-1e6;
        for(int k=0;k<nFreeQ;k++) ddQ[k]=(haveSec && fabs(qcur[k]-qPrev[k])>1e-12)?(gradQ[k]-gqPrev[k])/(qcur[k]-qPrev[k]):-1e6;
        aPrev=baseA; gaPrev=ga; pPrev=baseP; gpPrev=gradPinv;
        for(int k=0;k<nFreeQ;k++){ qPrev[k]=qcur[k]; gqPrev[k]=gradQ[k]; }
        haveSec=true;
        bool acc=false;
        for(int bt=0; bt<14; bt++){
            cand=base; for(int e=0;e<nedge;e++){ int v=edgeV[e]; double dn=fabs(g_ddf[e])+mu; double nb=base[v]+g_df[e]/dn; if(nb<1e-6)nb=1e-6; if(nb>20.0)nb=20.0; cand[v]=nb; }
            double ca=baseA; if(optAlpha && ncat>1){ double da=ga/(fabs(ddA)+mu); ca=baseA+da; if(ca<0.02)ca=0.02; if(ca>50.0)ca=50.0; }
            double cp=baseP; if(optPinv){ double dp=gradPinv/(fabs(ddP)+mu); cp=baseP+dp; if(cp<pinvMin)cp=pinvMin; if(cp>pinvMax)cp=pinvMax; }
            std::vector<double> cq(nFreeQ>0?nFreeQ:0);
            for(int k=0;k<nFreeQ;k++){ double dn=fabs(ddQ[k])+mu; double nq=qcur[k]+gradQ[k]/dn; if(nq<MINQ)nq=MINQ; if(nq>MAXQ)nq=MAXQ; cq[k]=nq; }
            double ln=evalLnL(cand,ca,cp, nFreeQ>0?cq.data():nullptr); nLnLEval++;
            if(ln>lnL+1e-9){ double dl=ln-lnL; brlen=cand; curAlpha=ca; curPinv=cp; if(nFreeQ>0) qcur=cq; lnL=ln; mu=fmax(mu*0.5,1e-9); acc=true; if(dl<tol)conv=true; break; }
            else { mu*=4.0; nRej++; } }
        if(!acc){ brlen=base; curAlpha=baseA; curPinv=baseP; if(nFreeQ>0) qApply(qcur.data()); break; }
        if(conv) break; }

    if (cudaGetLastError()!=cudaSuccess) return (double)NAN;   // any launch/sync error -> caller falls back to CPU
    for(int v=0;v<nnodes;v++) out_brlen[v]=brlen[v];
    if(out_alpha) *out_alpha=curAlpha;
    if(out_pinv)  *out_pinv = optPinv ? curPinv : pinv0;
    if(nFreeQ>0 && out_q) for(int k=0;k<nFreeQ;k++) out_q[k]=qcur[k];
    if(out_iters) *out_iters=it;
    return lnL;
}
