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

#define NS_MAX 20

// ---- device model constants (set per cross-check call; one tree, one model) ----
__constant__ double g_Uinv[NS_MAX*NS_MAX];
__constant__ double g_UinvRowSum[NS_MAX];
__constant__ double g_freq[NS_MAX];
__constant__ double g_catw[64];
// G.2.1a single-edge derivative coefficients (per cat,state at the central branch length t):
__constant__ double g_val0[64*NS_MAX];   // exp(eval[x]*rate_c*t) * prop_c
__constant__ double g_val1[64*NS_MAX];   // (rate_c*eval[x]) * val0
__constant__ double g_val2[64*NS_MAX];   // (rate_c*eval[x]) * val1

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
