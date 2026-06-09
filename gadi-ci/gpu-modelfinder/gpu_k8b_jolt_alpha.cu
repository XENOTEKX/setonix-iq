// gpu_k7b_freerate.cu — Phase G.4.0b (JOLT make-or-break): O(depth) preorder recycling + the FreeRate (+R)
// rate-parameter gradient OVERFLOW KILL-SWITCH. Extends the G.4.0 K7 preorder all-branch gradient with:
//  (A) Ji-2020 O(depth) pre-slot RECYCLING — the G.4.0 harness held one preorder slot per node (~nnodes*slotSz,
//      OOMs r8/r10 on a 32GB V100). Here a POOL of (treeHeight+2) slots is recycled by a single interleaved
//      preorder DFS: pre_v -> recycled slot, consumed immediately (theta_v = pre_v (.) pl_v -> df_v), recurse,
//      release. Live set = current root->node path = tree height << nnodes -> r8/r10 fit. The branch-grad gates
//      (edge-invariant lnL + df FD) re-run and MUST match G.4.0 -> recycling is numerically identical.
//  (B) FreeRate rate gradient dlnL/dr_k — the EXACT reduction that overflowed ~1e54 in CPU Mode-L (contrib =
//      cf*qp*exp(scale_log - _pattern_lh)). On the UNSCALED FP64 eigen path scale_log==0, so it is assembled as
//      (Sum_e b_e*qp_e[k])/L_ptn -> O(1). Kill-switch (4 gates): (B1) FINITE & bounded (no overflow); (B2) the
//      EXACT scaling identity  Sum_k r_k*dlnL/dr_k == Sum_e b_e*dlnL/db_e  ties +R grad to the validated branch
//      grad; (B3) FD-validate |G-ratio|<0.01 (the Mode-L FDCHECK that read 1e54), incl. the plan-named LG+R4.
// Build: nvcc -O3 -std=c++17 -arch=sm_70.  Run: ./gpu_k7b_freerate <aln.phy> <tree> [g4|r4|r8|r10|g1] [reps]
// (original G.1.2 K2 header below)
// gpu_k2_derv.cu — Phase G.1.2: custom CUDA single-edge branch-length derivative kernel (K2).
//
// The analog of IQ-TREE's computeLikelihoodDervSIMD: at a central edge (node,dad) of a FIXED tree it
// computes lnL, df = d lnL/dt, ddf = d2 lnL/dt2 w.r.t. the branch length t — the primitive driven by
// minimizeNewton inside optimizeAllBranches (= 75-85% of per-model wall). Standalone harness; reuses the
// G.1.1 eigen-space postorder kernel (k1_node) for the endpoint partials, then adds:
//   theta[c][x] = node_eig[c][x] * dad_eig[c][x]            (t-INDEPENDENT, cached once)
//   val0[c][x]  = exp(eval[x]*rate_c*t) * prop_c            (per t)
//   val1[c][x]  = (rate_c*eval[x]) * val0[c][x]             (d/dt)
//   val2[c][x]  = (rate_c*eval[x]) * val1[c][x]             (d2/dt2)
//   lh = sum_{c,x} val0*theta ; d1 = sum val1*theta ; d2 = sum val2*theta
//   per pattern: df += d1/lh ; ddf += d2/lh - (d1/lh)^2     (lnL += log|lh|)
//
// Eigen-space edge formulation needs NO explicit freq[x]: IQ-TREE's convention gives the identity
// freq_a*U[a][x] = Uinv[x][a] (U=diag(1/sqrt pi)W, Uinv=W^T diag(sqrt pi)), so the freq folds into the
// eigen-space partials. => K2's lnL(t0) MUST equal K1's oracle (a built-in cross-check).
//
// VALIDATION (the non-negotiable gate): FD-validate df/ddf against central differences of lnL(t) at a t
// OFF the optimum (where df != 0), swept-eps best of {1e-2,1e-3,1e-4}; PASS g4<3e-3, g1<1e-6. Then Newton
// from a perturbed t must converge back to the tree's optimized edge length (df -> 0).
//
// Build (gpuvolta job): module load cuda/12.5.1 gcc/12.2.0
//   nvcc -O3 -std=c++17 -arch=sm_70 gpu_k2_derv.cu -o gpu_k2_derv
// Run:  ./gpu_k2_derv <aln.phy> <tree> [model=g4|r8|r10|g1] [reps]
//
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <functional>
#include <fcntl.h>          // G.4.1: mmap the alignment (RAM-resident page cache) -> pinned host -> GPU
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
using namespace std;
using Clock = chrono::high_resolution_clock;
static double now_ms(Clock::time_point a, Clock::time_point b){ return chrono::duration<double, milli>(b-a).count(); }
#define NS 20
#define MAXCAT 16
#define CK(x) do{ cudaError_t _e=(x); if(_e!=cudaSuccess){ fprintf(stderr,"[CUDA-ERR] %s at %s:%d: %s\n",#x,__FILE__,__LINE__,cudaGetErrorString(_e)); exit(7);} }while(0)

// ===================== CPU scaffolding (lifted from gpu_derisk.cpp / gpu_k1_lnl.cu, BEAGLE-free) =====================
static const char* AA_ORDER = "ARNDCQEGHILKMFPSTWYV";
static void fill_LG(double R[20][20], double f[20]) {
    for (int i=0;i<20;i++) for(int j=0;j<20;j++) R[i][j]=0.0;
    auto S=[&](int i,int j,double v){ R[i][j]=v; R[j][i]=v; };
    S(1,0,0.425093);
    S(2,0,0.276818); S(2,1,0.751878);
    S(3,0,0.395144); S(3,1,0.123954); S(3,2,5.076149);
    S(4,0,2.489084); S(4,1,0.534551); S(4,2,0.528768); S(4,3,0.062556);
    S(5,0,0.969894); S(5,1,2.807908); S(5,2,1.695752); S(5,3,0.523386); S(5,4,0.084808);
    S(6,0,1.038545); S(6,1,0.363970); S(6,2,0.541712); S(6,3,5.243870); S(6,4,0.003499); S(6,5,4.128591);
    S(7,0,2.066040); S(7,1,0.390192); S(7,2,1.437645); S(7,3,0.844926); S(7,4,0.569265); S(7,5,0.267959); S(7,6,0.348847);
    S(8,0,0.358858); S(8,1,2.426601); S(8,2,4.509238); S(8,3,0.927114); S(8,4,0.640543); S(8,5,4.813505); S(8,6,0.423881); S(8,7,0.311484);
    S(9,0,0.149830); S(9,1,0.126991); S(9,2,0.191503); S(9,3,0.010690); S(9,4,0.320627); S(9,5,0.072854); S(9,6,0.044265); S(9,7,0.008705); S(9,8,0.108882);
    S(10,0,0.395337); S(10,1,0.301848); S(10,2,0.068427); S(10,3,0.015076); S(10,4,0.594007); S(10,5,0.582457); S(10,6,0.069673); S(10,7,0.044261); S(10,8,0.366317); S(10,9,4.145067);
    S(11,0,0.536518); S(11,1,6.326067); S(11,2,2.145078); S(11,3,0.282959); S(11,4,0.013266); S(11,5,3.234294); S(11,6,1.807177); S(11,7,0.296636); S(11,8,0.697264); S(11,9,0.159069); S(11,10,0.137500);
    S(12,0,1.124035); S(12,1,0.484133); S(12,2,0.371004); S(12,3,0.025548); S(12,4,0.893680); S(12,5,1.672569); S(12,6,0.173735); S(12,7,0.139538); S(12,8,0.442472); S(12,9,4.273607); S(12,10,6.312358); S(12,11,0.656604);
    S(13,0,0.253701); S(13,1,0.052722); S(13,2,0.089525); S(13,3,0.017416); S(13,4,1.105251); S(13,5,0.035855); S(13,6,0.018811); S(13,7,0.089586); S(13,8,0.682139); S(13,9,1.112727); S(13,10,2.592692); S(13,11,0.023918); S(13,12,1.798853);
    S(14,0,1.177651); S(14,1,0.332533); S(14,2,0.161787); S(14,3,0.394456); S(14,4,0.075382); S(14,5,0.624294); S(14,6,0.419409); S(14,7,0.196961); S(14,8,0.508851); S(14,9,0.078281); S(14,10,0.249060); S(14,11,0.390322); S(14,12,0.099849); S(14,13,0.094464);
    S(15,0,4.727182); S(15,1,0.858151); S(15,2,4.008358); S(15,3,1.240275); S(15,4,2.784478); S(15,5,1.223828); S(15,6,0.611973); S(15,7,1.739990); S(15,8,0.990012); S(15,9,0.064105); S(15,10,0.182287); S(15,11,0.748683); S(15,12,0.346960); S(15,13,0.361819); S(15,14,1.338132);
    S(16,0,2.139501); S(16,1,0.578987); S(16,2,2.000679); S(16,3,0.425860); S(16,4,1.143480); S(16,5,1.080136); S(16,6,0.604545); S(16,7,0.129836); S(16,8,0.584262); S(16,9,1.033739); S(16,10,0.302936); S(16,11,1.136863); S(16,12,2.020366); S(16,13,0.165001); S(16,14,0.571468); S(16,15,6.472279);
    S(17,0,0.180717); S(17,1,0.593607); S(17,2,0.045376); S(17,3,0.029890); S(17,4,0.670128); S(17,5,0.236199); S(17,6,0.077852); S(17,7,0.268491); S(17,8,0.597054); S(17,9,0.111660); S(17,10,0.619632); S(17,11,0.049906); S(17,12,0.696175); S(17,13,2.457121); S(17,14,0.095131); S(17,15,0.248862); S(17,16,0.140825);
    S(18,0,0.218959); S(18,1,0.314440); S(18,2,0.612025); S(18,3,0.135107); S(18,4,1.165532); S(18,5,0.257336); S(18,6,0.120037); S(18,7,0.054679); S(18,8,5.306834); S(18,9,0.232523); S(18,10,0.299648); S(18,11,0.131932); S(18,12,0.481306); S(18,13,7.803902); S(18,14,0.089613); S(18,15,0.400547); S(18,16,0.245841); S(18,17,3.151815);
    S(19,0,2.547870); S(19,1,0.170887); S(19,2,0.083688); S(19,3,0.037967); S(19,4,1.959291); S(19,5,0.210332); S(19,6,0.245034); S(19,7,0.076701); S(19,8,0.119013); S(19,9,10.649107); S(19,10,1.702745); S(19,11,0.185202); S(19,12,1.898718); S(19,13,0.654683); S(19,14,0.296501); S(19,15,0.098369); S(19,16,2.188158); S(19,17,0.189510); S(19,18,0.249313);
    double ff[20]={0.079066,0.055941,0.041977,0.053052,0.012937,0.040767,0.071586,0.057337,0.022355,0.062157,0.099081,0.064600,0.022951,0.042302,0.044040,0.061197,0.053287,0.012066,0.034155,0.069146};
    for (int i=0;i<20;i++) f[i]=ff[i];
}
static void jacobi_eig(double A[20][20], double eval[20], double evec[20][20]) {
    const int n=20;
    for (int i=0;i<n;i++){ for(int j=0;j<n;j++) evec[i][j]=(i==j)?1.0:0.0; }
    for (int sweep=0; sweep<100; sweep++) {
        double off=0; for(int p=0;p<n;p++) for(int q=p+1;q<n;q++) off+=A[p][q]*A[p][q];
        if (off < 1e-30) break;
        for (int p=0;p<n;p++) for (int q=p+1;q<n;q++) {
            if (fabs(A[p][q])<1e-300) continue;
            double theta=(A[q][q]-A[p][p])/(2.0*A[p][q]);
            double t=(theta>=0?1.0:-1.0)/(fabs(theta)+sqrt(theta*theta+1.0));
            double c=1.0/sqrt(t*t+1.0), s=t*c;
            for (int i=0;i<n;i++){ double aip=A[i][p], aiq=A[i][q]; A[i][p]=c*aip-s*aiq; A[i][q]=s*aip+c*aiq; }
            for (int i=0;i<n;i++){ double api=A[p][i], aqi=A[q][i]; A[p][i]=c*api-s*aqi; A[q][i]=s*api+c*aqi; }
            for (int i=0;i<n;i++){ double vip=evec[i][p], viq=evec[i][q]; evec[i][p]=c*vip-s*viq; evec[i][q]=s*vip+c*viq; }
        }
    }
    for (int i=0;i<n;i++) eval[i]=A[i][i];
}
static int aa_index(char c) { c=toupper(c); static map<char,int> m; if(m.empty()) for(int i=0;i<20;i++) m[AA_ORDER[i]]=i;
    auto it=m.find(c); return (it!=m.end())?it->second:-1; }
struct Node { vector<int> child; vector<double> blen; int leaf=-1; };
struct Tree { vector<Node> nodes; int root=-1; };
static Tree parse_newick(const string& s, map<string,int>& name2tip) {
    Tree T; size_t i=0;
    function<int(void)> parse=[&]()->int{ int id=T.nodes.size(); T.nodes.push_back(Node());
        if (s[i]=='('){ i++; while(true){ int c=parse(); double bl=0.0;
                if (s[i]==':'){ i++; size_t j=i; while(i<s.size()&&(isdigit(s[i])||s[i]=='.'||s[i]=='-'||s[i]=='e'||s[i]=='E'||s[i]=='+')) i++; bl=atof(s.substr(j,i-j).c_str()); }
                T.nodes[id].child.push_back(c); T.nodes[id].blen.push_back(bl);
                if (s[i]==','){ i++; continue; } if (s[i]==')'){ i++; break; } } }
        else { size_t j=i; while(i<s.size()&&s[i]!=':'&&s[i]!=','&&s[i]!=')'&&s[i]!=';') i++; string nm=s.substr(j,i-j);
            auto it=name2tip.find(nm); if(it==name2tip.end()){fprintf(stderr,"tip '%s' not in aln\n",nm.c_str());exit(5);} T.nodes[id].leaf=it->second; }
        return id; };
    T.root=parse(); return T;
}

// ===================== device constants + kernels =====================
__constant__ double c_Uinv[NS*NS];
__constant__ double c_U[NS*NS];          // G.4.0: evec (getEigenvectors), needed by k7_pre step 1
__constant__ double c_UinvRowSum[NS];
__constant__ double c_val0[MAXCAT*NS];   // exp(eval*rate*t)*prop
__constant__ double c_val1[MAXCAT*NS];   // (rate*eval)*val0
__constant__ double c_val2[MAXCAT*NS];   // (rate*eval)*val1
__constant__ double c_rscale[MAXCAT];    // G.4.0b: per-cat edge scale b_e/(r_k*w_k) for the +R rate-grad numerator

// per-child probability-space contribution (same as K1)
__device__ __forceinline__ void accum_child(double* prod, int c, int ptn, int nptn,
        const double* __restrict__ ec, const double* __restrict__ p, const unsigned char* __restrict__ t) {
    const double* ecc = ec + (size_t)c*NS*NS;
    if (p) { const double* pc = p + (size_t)(c*NS)*nptn + ptn;
        #pragma unroll
        for (int x=0;x<NS;x++){ double v=0.0;
            #pragma unroll
            for (int i=0;i<NS;i++) v += ecc[x*NS+i]*pc[(size_t)i*nptn]; prod[x]*=v; }
    } else { int s=t[ptn];
        #pragma unroll
        for (int x=0;x<NS;x++){ double v=0.0;
            #pragma unroll
            for (int i=0;i<NS;i++){ double Li=(s<NS)?c_Uinv[i*NS+s]:c_UinvRowSum[i]; v+=ecc[x*NS+i]*Li; } prod[x]*=v; }
    }
}
// eigen-space partial for one internal node (K1 kernel; isRoot path unused here)
__global__ void k1_node(int nptn, int ncat, double* __restrict__ out, int nchild,
        const double* ec0,const double* p0,const unsigned char* t0,
        const double* ec1,const double* p1,const unsigned char* t1,
        const double* ec2,const double* p2,const unsigned char* t2){
    int ptn=blockIdx.x*blockDim.x+threadIdx.x; if(ptn>=nptn) return;
    for (int c=0;c<ncat;c++){ double prod[NS];
        #pragma unroll
        for(int x=0;x<NS;x++) prod[x]=1.0;
        accum_child(prod,c,ptn,nptn,ec0,p0,t0);
        if(nchild>1) accum_child(prod,c,ptn,nptn,ec1,p1,t1);
        if(nchild>2) accum_child(prod,c,ptn,nptn,ec2,p2,t2);
        double* o=out+(size_t)(c*NS)*nptn+ptn;
        #pragma unroll
        for(int r=0;r<NS;r++){ double v=0.0;
            #pragma unroll
            for(int x=0;x<NS;x++) v+=c_Uinv[r*NS+x]*prod[x]; o[(size_t)r*nptn]=v; }
    }
}
// theta = node_eig elementwise* dad_eig  (t-independent)
__global__ void k2_theta(int nptn, int blockc, const double* __restrict__ node, const double* __restrict__ dad, double* __restrict__ theta){
    int ptn=blockIdx.x*blockDim.x+threadIdx.x; if(ptn>=nptn) return;
    for (int k=0;k<blockc;k++){ size_t o=(size_t)k*nptn+ptn; theta[o]=node[o]*dad[o]; }
}
// per-pattern lnL + df + ddf from theta and the t-dependent val0/val1/val2
__global__ void k2_derv(int nptn, int ncat, const double* __restrict__ theta,
        double* __restrict__ patlh, double* __restrict__ pdf, double* __restrict__ pddf){
    int ptn=blockIdx.x*blockDim.x+threadIdx.x; if(ptn>=nptn) return;
    double lh=0.0,d1=0.0,d2=0.0;
    for (int c=0;c<ncat;c++){
        #pragma unroll
        for (int x=0;x<NS;x++){ double th=theta[(size_t)(c*NS+x)*nptn+ptn]; int k=c*NS+x;
            lh+=c_val0[k]*th; d1+=c_val1[k]*th; d2+=c_val2[k]*th; }
    }
    lh=fabs(lh);                        // +ptn_invar = 0 for these models
    double inv=1.0/lh, r=d1*inv;
    patlh[ptn]=log(lh); pdf[ptn]=r; pddf[ptn]=d2*inv-r*r;
}

// G.4.0 — leaf-edge eig partial: pl_leaf[c*NS+r][ptn] = Uinv[r][s] (s=tip state; ambiguous -> UinvRowSum),
// rate-independent (same for every category). Lets a leaf edge use the same theta = pre (.) pl reduction.
__global__ void k_leafeig(int nptn,int ncat,const unsigned char* __restrict__ tipstate,double* __restrict__ out){
    int ptn=blockIdx.x*blockDim.x+threadIdx.x; if(ptn>=nptn) return; int s=tipstate[ptn];
    for(int c=0;c<ncat;c++){ double* o=out+(size_t)(c*NS)*nptn+ptn;
        for(int r=0;r<NS;r++) o[(size_t)r*nptn]=(s<NS)?c_Uinv[r*NS+s]:c_UinvRowSum[r]; } }

// G.4.0 — K7: top-down PREORDER partial (Ji-2020 / Mode-L L.0b.ii recursion, eigen-space).
//   pus[t]  = Sum_i Uinv[i][t] * pre_u[c][i]                       (parent preorder partial, projected)
//   fsib[t] = Prod_siblings ( Sum_i echild_sib[t][i]*pl_sib[i] )   (= Prod_sib pk_sib[t]; via accum_child)
//   pre_v[c][i] = exp(eval[i]*rate_c*b_v) * Sum_t Uinv[i][t]*pus[t]*fsib[t]
// pre_v = eigen-space "rest of tree" partial at v; with pl_v (postorder) -> theta_v = pre_v (.) pl_v, and the
// SAME val0/val1/val2 reduction (k2_derv, validated G.2.1a/b) gives lnL/df/ddf for edge (u->v).
__global__ void k7_pre(int nptn,int ncat,double* __restrict__ out_pre,
        const double* __restrict__ pre_u,const double* __restrict__ expfac_u,
        int nsib,const double* ec0,const double* sp0,const unsigned char* st0,
                const double* ec1,const double* sp1,const unsigned char* st1){
    // CONVENTION (matches the validated seeded central edge): d_pre[v] stores the "above" partial WITHOUT
    // v's own branch, as f(parent state) — the gradient's val0/val1(b_v) reapply b_v exactly once (no
    // double-count, and pre_v is independent of b_v so the FD perturbation is well-posed). The PARENT branch
    // b_u (expfac_u) is applied here in step 1 to propagate pre_u from the grandparent endpoint to u's state.
    int ptn=blockIdx.x*blockDim.x+threadIdx.x; if(ptn>=nptn) return;
    for(int c=0;c<ncat;c++){
        double fsib[NS];
        for(int t=0;t<NS;t++) fsib[t]=1.0;
        accum_child(fsib,c,ptn,nptn,ec0,sp0,st0);
        if(nsib>1) accum_child(fsib,c,ptn,nptn,ec1,sp1,st1);
        const double* puc=pre_u+(size_t)(c*NS)*nptn+ptn; const double* ef=expfac_u+(size_t)c*NS;
        double pus[NS];                                  // propagate pre_u through PARENT branch b_u, to state space
        for(int t=0;t<NS;t++){ double v=0.0;
            for(int i=0;i<NS;i++) v+=c_U[t*NS+i]*ef[i]*puc[(size_t)i*nptn]; pus[t]=v; }
        double* o=out_pre+(size_t)(c*NS)*nptn+ptn;
        for(int j=0;j<NS;j++){ double v=0.0;             // back to eigen; store WITHOUT own branch b_v
            for(int t=0;t<NS;t++) v+=c_Uinv[j*NS+t]*pus[t]*fsib[t]; o[(size_t)j*nptn]=v; } }
}

// G.4.0b — FreeRate rate-gradient numerator accumulator. For edge e (length b_e) with theta_e cached and
// c_val1 set at b_e, this adds b_e*qp_e[k] to rnum[k][ptn] for every category k, where
//   qp_e[k] = Sum_x lambda_x * exp(lambda_x r_k b_e) * theta_e[k,x]  and  Sum_x c_val1[k,x]*theta = r_k w_k qp_e[k].
// c_rscale[k] = b_e/(r_k w_k) folds the chain-rule factor, so rnum[k] accumulates Sum_e b_e*qp_e[k] across edges.
// The +R rate gradient is then dlnL/dr_k = w_k * Sum_ptn rnum[k][ptn]/L_ptn  (assembled on host, division O(1)).
__global__ void k_ratenum(int nptn,int ncat,const double* __restrict__ theta,double* __restrict__ rnum){
    int ptn=blockIdx.x*blockDim.x+threadIdx.x; if(ptn>=nptn) return;
    for(int k=0;k<ncat;k++){ double s=0.0;
        #pragma unroll
        for(int x=0;x<NS;x++) s+=c_val1[k*NS+x]*theta[(size_t)(k*NS+x)*nptn+ptn];
        rnum[(size_t)k*nptn+ptn]+=c_rscale[k]*s; }
}


// ===================== G.4.1b: mean-rate discrete-Gamma (Yang 1994; IQ-TREE's "MEAN of the portion") =====================
// IQ-TREE .iqtree states: "Relative rates are computed as MEAN of the portion of the Gamma distribution falling
// in the category." Gamma has mean 1 (shape=alpha, scale=1/alpha). K equal-probability bins; category rate
//   r_c = K * [ P(alpha+1, alpha*b_c) - P(alpha+1, alpha*b_{c-1}) ],   b_c = (c/K)-quantile, b_0=0, b_K=inf,
// where P is the regularized lower incomplete gamma. This MUST reproduce {0.1362,0.4756,0.9994,2.3887} at
// alpha=0.9963 (validated in-harness before optimising).
static double gammp_reg(double a,double x){     // regularized lower incomplete gamma P(a,x) (NR gser/gcf)
    if(x<=0.0) return 0.0;
    double gln=lgamma(a);
    if(x<a+1.0){ double ap=a,sum=1.0/a,del=sum; for(int n=1;n<=300;n++){ ap+=1.0; del*=x/ap; sum+=del; if(fabs(del)<fabs(sum)*1e-16) break; }
        return sum*exp(-x+a*log(x)-gln); }
    double b=x+1.0-a,c=1e300,d=1.0/b,h=d;        // continued fraction for Q, return 1-Q
    for(int i=1;i<=300;i++){ double an=-(double)i*((double)i-a); b+=2.0; d=an*d+b; if(fabs(d)<1e-300)d=1e-300; c=b+an/c; if(fabs(c)<1e-300)c=1e-300; d=1.0/d; double del=d*c; h*=del; if(fabs(del-1.0)<1e-16) break; }
    return 1.0-exp(-x+a*log(x)-gln)*h;
}
static double gammp_inv(double a,double p){      // inverse: x s.t. P(a,x)=p, by bracketed bisection
    if(p<=0.0) return 0.0; if(p>=1.0) return 1e300;
    double lo=0.0,hi=a+10.0*sqrt(a+1.0)+20.0; int guard=0; while(gammp_reg(a,hi)<p && guard++<200) hi*=2.0;
    for(int it=0;it<200;it++){ double mid=0.5*(lo+hi); if(gammp_reg(a,mid)<p) lo=mid; else hi=mid; if(hi-lo<1e-13*(mid+1e-13)) break; }
    return 0.5*(lo+hi);
}
static void discreteGammaMean(double alpha,int K,double* rates){
    if(K==1){ rates[0]=1.0; return; }
    double prev=0.0;                              // P(alpha+1, alpha*b_0=0)=0
    for(int c=0;c<K;c++){ double hi;
        if(c==K-1) hi=1.0;                        // P(alpha+1, inf)=1
        else { double bc=gammp_inv(alpha,(double)(c+1)/(double)K)/alpha; hi=gammp_reg(alpha+1.0, alpha*bc); }
        rates[c]=(double)K*(hi-prev); prev=hi; }
}
int main(int argc, char** argv){
    if (argc<3){ fprintf(stderr,"usage: %s <aln.phy> <tree> [model=g4] [maxiter]\n",argv[0]); return 1; }
    string alnpath=argv[1],treepath=argv[2]; string model=(argc>3)?argv[3]:"g4"; int maxiter=(argc>4)?atoi(argv[4]):400;
    if(model!="g4"){ fprintf(stderr,"G.4.1b joint-alpha gate uses g4 (LG+G4); other models later\n"); return 1; }

    // ---- alignment via MMAP (RAM-resident page cache) -> pinned host -> GPU ----
    int afd=open(alnpath.c_str(),O_RDONLY); if(afd<0){fprintf(stderr,"no aln\n");return 2;}
    struct stat ast; if(fstat(afd,&ast)!=0){fprintf(stderr,"fstat fail\n");return 2;} size_t aln_bytes=(size_t)ast.st_size;
    char* amap=(char*)mmap(nullptr,aln_bytes,PROT_READ,MAP_PRIVATE,afd,0); if(amap==MAP_FAILED){fprintf(stderr,"mmap failed\n");return 2;}
    madvise(amap,aln_bytes,MADV_WILLNEED);
    int ntax=0,nsite=0; const char* pcur=amap; const char* pend=amap+aln_bytes;
    { sscanf(pcur,"%d %d",&ntax,&nsite); while(pcur<pend&&*pcur!='\n')pcur++; if(pcur<pend)pcur++; }
    vector<string> seqs(ntax); map<string,int> name2tip;
    for(int t=0;t<ntax;t++){ const char* ls=pcur; while(pcur<pend&&*pcur!='\n')pcur++; string line(ls,(size_t)(pcur-ls)); if(pcur<pend)pcur++;
        istringstream is(line); string nm,sq; is>>nm>>sq; seqs[t]=sq; name2tip[nm]=t; }
    int nptn=nsite; printf("[aln] ntax=%d nsite=%d (mmap %.1f MB RAM-resident)\n",ntax,nptn,aln_bytes/1048576.0);

    string ts; { ifstream tf(treepath); stringstream ss; ss<<tf.rdbuf(); ts=ss.str(); }
    string tcl; for(char c:ts) if(!isspace((unsigned char)c)) tcl+=c;
    Tree T=parse_newick(tcl,name2tip); int nnodes=T.nodes.size();
    printf("[tree] nodes=%d root_children=%zu\n",nnodes,T.nodes[T.root].child.size());

    // ---- model: LG + Gamma(4); rates DERIVED from alpha via the mean-rate discretisation (JOINTLY optimised) ----
    double R[20][20],f[20]; fill_LG(R,f); int NCAT=4; vector<double> catRates(NCAT,1.0), catWeights(NCAT,0.25);
    double curAlpha=1.0;
    auto applyAlpha=[&](double a){ double r[16]; discreteGammaMean(a,NCAT,r); for(int c=0;c<NCAT;c++) catRates[c]=r[c]; };
    // VALIDATE the discretisation reproduces IQ-TREE's printed rates at the MLE alpha=0.9963
    { double r[16]; discreteGammaMean(0.9963,4,r); double ref[4]={0.1362,0.4756,0.9994,2.3887}; double w=0;
      for(int c=0;c<4;c++) w=fmax(w,fabs(r[c]-ref[c]));
      printf("[gamma] discretisation @alpha=0.9963: {%.4f,%.4f,%.4f,%.4f} vs IQ-TREE {0.1362,0.4756,0.9994,2.3887} maxdiff=%.2e %s\n",
        r[0],r[1],r[2],r[3],w,(w<5e-4?"PASS":"FAIL(check discretisation)")); }

    // ---- eigendecomposition (reversible convention; alpha-INDEPENDENT, done once) ----
    double Q[20][20]; for(int i=0;i<NS;i++){double row=0; for(int j=0;j<NS;j++){if(i!=j){Q[i][j]=R[i][j]*f[j]; row+=Q[i][j];}} Q[i][i]=-row;}
    double mu0=0; for(int i=0;i<NS;i++) mu0+=f[i]*(-Q[i][i]); for(int i=0;i<NS;i++) for(int j=0;j<NS;j++) Q[i][j]/=mu0;
    double sq[20],Bm[20][20]; for(int i=0;i<NS;i++) sq[i]=sqrt(f[i]);
    for(int i=0;i<NS;i++) for(int j=0;j<NS;j++) Bm[i][j]=sq[i]*Q[i][j]/sq[j];
    for(int i=0;i<NS;i++) for(int j=i+1;j<NS;j++){double m=0.5*(Bm[i][j]+Bm[j][i]); Bm[i][j]=Bm[j][i]=m;}
    double evl[20],V[20][20]; jacobi_eig(Bm,evl,V);
    vector<double> U(NS*NS),Uinv(NS*NS),evals(NS);
    for(int i=0;i<NS;i++){ evals[i]=evl[i]; for(int j=0;j<NS;j++){ U[i*NS+j]=V[i][j]/sq[i]; Uinv[i*NS+j]=V[j][i]*sq[j]; } }
    vector<double> UinvRowSum(NS,0.0); for(int i=0;i<NS;i++){double s=0; for(int j=0;j<NS;j++) s+=Uinv[i*NS+j]; UinvRowSum[i]=s;}

    vector<double> mleLen(nnodes,0.0);
    for(int u=0;u<nnodes;u++) for(size_t k=0;k<T.nodes[u].child.size();k++) mleLen[T.nodes[u].child[k]]=T.nodes[u].blen[k];
    vector<int> postorder; vector<int> slot(nnodes,-1);
    function<void(int)> dfs=[&](int u){ for(int c:T.nodes[u].child) dfs(c); if(T.nodes[u].leaf<0){ slot[u]=postorder.size(); postorder.push_back(u);} };
    dfs(T.root); int nInternal=postorder.size();
    int c0=-1; for(int c:T.nodes[T.root].child) if(T.nodes[c].leaf<0){ c0=c; break; } if(c0<0){fprintf(stderr,"no internal root child\n");return 6;}
    vector<int> edgeV; for(int u=0;u<nnodes;u++) for(int v:T.nodes[u].child) edgeV.push_back(v); int nedge=edgeV.size();
    int treeH=0; function<void(int,int)> depthDfs=[&](int u,int d){ if(d>treeH)treeH=d; for(int c:T.nodes[u].child) depthDfs(c,d+1); };
    depthDfs(T.root,0); int nPool=treeH+2;
    printf("[edges] %d branches; central=(root=%d,c0=%d); height=%d pool=%d ; joint theta=(197 brlen, alpha)\n",nedge,T.root,c0,treeH,nPool);

    size_t ecStride=(size_t)NCAT*NS*NS, slotSz=(size_t)NCAT*NS*nptn;
    unsigned char* h_tip=nullptr; CK(cudaHostAlloc((void**)&h_tip,(size_t)ntax*nptn,cudaHostAllocDefault));
    for(int u=0;u<nnodes;u++){ if(T.nodes[u].leaf<0) continue; int lf=T.nodes[u].leaf; const string&s=seqs[lf];
        for(int p=0;p<nptn;p++){ int a=aa_index(s[p]); h_tip[(size_t)lf*nptn+p]=(unsigned char)((a<0)?NS:a); } }
    unsigned char* d_tip; CK(cudaMalloc(&d_tip,(size_t)ntax*nptn)); CK(cudaMemcpy(d_tip,h_tip,(size_t)ntax*nptn,cudaMemcpyHostToDevice));
    munmap(amap,aln_bytes); close(afd);

    CK(cudaMemcpyToSymbol(c_Uinv,Uinv.data(),sizeof(double)*NS*NS));
    CK(cudaMemcpyToSymbol(c_U,U.data(),sizeof(double)*NS*NS));
    CK(cudaMemcpyToSymbol(c_UinvRowSum,UinvRowSum.data(),sizeof(double)*NS));
    double *d_echild,*d_partial,*d_theta,*d_patlh,*d_pdf,*d_pddf,*d_pretmp,*d_tipeig,*d_expfac,*d_prepool,*d_rnum;
    CK(cudaMalloc(&d_echild,(size_t)nnodes*ecStride*sizeof(double)));
    CK(cudaMalloc(&d_partial,(size_t)nInternal*slotSz*sizeof(double)));
    CK(cudaMalloc(&d_theta,slotSz*sizeof(double)));
    CK(cudaMalloc(&d_patlh,(size_t)nptn*sizeof(double))); CK(cudaMalloc(&d_pdf,(size_t)nptn*sizeof(double))); CK(cudaMalloc(&d_pddf,(size_t)nptn*sizeof(double)));
    CK(cudaMalloc(&d_pretmp,slotSz*sizeof(double))); CK(cudaMalloc(&d_tipeig,slotSz*sizeof(double)));
    CK(cudaMalloc(&d_prepool,(size_t)nPool*slotSz*sizeof(double)));
    CK(cudaMalloc(&d_expfac,(size_t)nnodes*NCAT*NS*sizeof(double)));
    CK(cudaMalloc(&d_rnum,(size_t)NCAT*nptn*sizeof(double)));      // G.4.1b: per-category rate-grad numerator
    double* h_echild=nullptr; CK(cudaHostAlloc((void**)&h_echild,(size_t)nnodes*ecStride*sizeof(double),cudaHostAllocDefault));
    double* h_expfac=nullptr; CK(cudaHostAlloc((void**)&h_expfac,(size_t)nnodes*NCAT*NS*sizeof(double),cudaHostAllocDefault));

    int TB=256, GB=(nptn+TB-1)/TB;
    vector<double> brlen(nnodes,0.0), patlh(nptn),pdf(nptn),pddf(nptn);

    auto childArgs=[&](int u,int excl,int& nch,const double** ec,const double** p,const unsigned char** t){
        nch=0; for(int k=0;k<3;k++){ec[k]=p[k]=nullptr;t[k]=nullptr;}
        for(int c:T.nodes[u].child){ if(c==excl) continue; ec[nch]=d_echild+(size_t)c*ecStride;
            if(T.nodes[c].leaf>=0) t[nch]=d_tip+(size_t)T.nodes[c].leaf*nptn; else p[nch]=d_partial+(size_t)slot[c]*slotSz; nch++; } };
    auto sibArg=[&](int w,const double*& ec,const double*& sp,const unsigned char*& st){
        ec=d_echild+(size_t)w*ecStride; sp=nullptr; st=nullptr;
        if(T.nodes[w].leaf>=0) st=d_tip+(size_t)T.nodes[w].leaf*nptn; else sp=d_partial+(size_t)slot[w]*slotSz; };
    auto rebuildEchild=[&](){
        for(int c=0;c<nnodes;c++){ if(c==T.root){ for(size_t z=0;z<ecStride;z++) h_echild[(size_t)c*ecStride+z]=0.0; continue; }
            for(int cat=0;cat<NCAT;cat++){ double len=brlen[c]*catRates[cat]; double ex[NS]; for(int i=0;i<NS;i++) ex[i]=exp(evals[i]*len);
                double* e=&h_echild[(size_t)c*ecStride+(size_t)cat*NS*NS]; for(int x=0;x<NS;x++) for(int i=0;i<NS;i++) e[x*NS+i]=U[x*NS+i]*ex[i];
                for(int i=0;i<NS;i++) h_expfac[(size_t)c*NCAT*NS+cat*NS+i]=ex[i]; } }
        CK(cudaMemcpy(d_echild,h_echild,(size_t)nnodes*ecStride*sizeof(double),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_expfac,h_expfac,(size_t)nnodes*NCAT*NS*sizeof(double),cudaMemcpyHostToDevice)); };
    auto postorderFill=[&](){
        for(int idx=0; idx<nInternal; idx++){ int u=postorder[idx]; if(u==T.root) continue;
            int nch; const double* ec[3]; const double* p[3]; const unsigned char* t[3]; childArgs(u,-1,nch,ec,p,t);
            k1_node<<<GB,TB>>>(nptn,NCAT,d_partial+(size_t)slot[u]*slotSz,nch,ec[0],p[0],t[0],ec[1],p[1],t[1],ec[2],p[2],t[2]); }
        CK(cudaDeviceSynchronize()); CK(cudaGetLastError()); };
    auto setVal=[&](double t){ vector<double> v0(NCAT*NS),v1(NCAT*NS),v2(NCAT*NS);
        for(int c=0;c<NCAT;c++){ double rc=catRates[c],pcw=catWeights[c]; for(int x=0;x<NS;x++){ double re=rc*evals[x],e=exp(evals[x]*rc*t)*pcw;
            v0[c*NS+x]=e; v1[c*NS+x]=re*e; v2[c*NS+x]=re*re*e; } }
        CK(cudaMemcpyToSymbol(c_val0,v0.data(),sizeof(double)*NCAT*NS)); CK(cudaMemcpyToSymbol(c_val1,v1.data(),sizeof(double)*NCAT*NS)); CK(cudaMemcpyToSymbol(c_val2,v2.data(),sizeof(double)*NCAT*NS)); };
    auto reduceDerv=[&](double& lnL,double& df,double& ddf){
        CK(cudaMemcpy(patlh.data(),d_patlh,nptn*sizeof(double),cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(pdf.data(),d_pdf,nptn*sizeof(double),cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(pddf.data(),d_pddf,nptn*sizeof(double),cudaMemcpyDeviceToHost));
        double L=0,kc=0; for(int p=0;p<nptn;p++){ double y=patlh[p]-kc,tt=L+y; kc=(tt-L)-y; L=tt; }
        double D=0,DD=0; for(int p=0;p<nptn;p++){ D+=pdf[p]; DD+=pddf[p]; } lnL=L; df=D; ddf=DD; };
    auto edgeThetaInto=[&](int v,const double* pre){ const double* pl;
        if(T.nodes[v].leaf<0) pl=d_partial+(size_t)slot[v]*slotSz;
        else { k_leafeig<<<GB,TB>>>(nptn,NCAT,d_tip+(size_t)T.nodes[v].leaf*nptn,d_tipeig); pl=d_tipeig; }
        k2_theta<<<GB,TB>>>(nptn,NCAT*NS,pl,pre,d_theta); };

    long nGradSweeps=0, nLnLEval=0;
    auto evalLnL=[&](const vector<double>& cand_b,double cand_a)->double{ applyAlpha(cand_a); brlen=cand_b; rebuildEchild(); postorderFill();
        int nch; const double* ec[3]; const double* p[3]; const unsigned char* tp[3]; childArgs(T.root,c0,nch,ec,p,tp);
        k1_node<<<GB,TB>>>(nptn,NCAT,d_pretmp,nch,ec[0],p[0],tp[0],ec[1],p[1],tp[1],ec[2],p[2],tp[2]); CK(cudaDeviceSynchronize());
        edgeThetaInto(c0,d_pretmp); CK(cudaDeviceSynchronize());
        setVal(brlen[c0]); k2_derv<<<GB,TB>>>(nptn,NCAT,d_theta,d_patlh,d_pdf,d_pddf); CK(cudaDeviceSynchronize());
        double l,d,dd; reduceDerv(l,d,dd); return l; };

    // joint gradient: branch df/ddf (preorder) + per-category rate-grad gradR[c] + alpha gradient.
    vector<double> g_df(nedge,0.0), g_ddf(nedge,0.0), gradR(NCAT,0.0), invL(nptn,0.0), rnumH((size_t)NCAT*nptn);
    auto computeGradient=[&](double& lnLout,double& galphaOut){
        rebuildEchild(); postorderFill(); nGradSweeps++; CK(cudaMemset(d_rnum,0,(size_t)NCAT*nptn*sizeof(double)));
        vector<int> freeSlots; for(int s=nPool-1;s>=0;s--) freeSlots.push_back(s);
        auto acq=[&](){int s=freeSlots.back();freeSlots.pop_back();return s;}; auto rls=[&](int s){freeSlots.push_back(s);};
        vector<double> dfC(nnodes,0.0), ddfC(nnodes,0.0); bool gotL=false; double lnLfirst=0;
        function<void(int,int)> proc=[&](int u,int su){
            for(int v:T.nodes[u].child){
                int sv=acq(); double* pre=d_prepool+(size_t)sv*slotSz;
                if(u==T.root){ int nch; const double* ec[3]; const double* p[3]; const unsigned char* tp[3]; childArgs(T.root,v,nch,ec,p,tp);
                    k1_node<<<GB,TB>>>(nptn,NCAT,pre,nch,ec[0],p[0],tp[0],ec[1],p[1],tp[1],ec[2],p[2],tp[2]); }
                else { const double* ec[2]={0,0}; const double* sp[2]={0,0}; const unsigned char* st[2]={0,0}; int ns=0;
                    for(int w:T.nodes[u].child){ if(w==v||ns>=2) continue; sibArg(w,ec[ns],sp[ns],st[ns]); ns++; }
                    k7_pre<<<GB,TB>>>(nptn,NCAT,pre,d_prepool+(size_t)su*slotSz,d_expfac+(size_t)u*NCAT*NS,ns,ec[0],sp[0],st[0],ec[1],sp[1],st[1]); }
                CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
                edgeThetaInto(v,pre); CK(cudaDeviceSynchronize());
                double bv=brlen[v]; setVal(bv); k2_derv<<<GB,TB>>>(nptn,NCAT,d_theta,d_patlh,d_pdf,d_pddf); CK(cudaDeviceSynchronize());
                vector<double> rs(NCAT); for(int c=0;c<NCAT;c++) rs[c]=bv/(catRates[c]*catWeights[c]);   // rate-grad numerator (G.4.0b k_ratenum)
                CK(cudaMemcpyToSymbol(c_rscale,rs.data(),sizeof(double)*NCAT)); k_ratenum<<<GB,TB>>>(nptn,NCAT,d_theta,d_rnum); CK(cudaDeviceSynchronize());
                double l,d,dd; reduceDerv(l,d,dd); dfC[v]=d; ddfC[v]=dd;
                if(!gotL){ lnLfirst=l; for(int p=0;p<nptn;p++) invL[p]=exp(-patlh[p]); gotL=true; }   // edge-invariant L_ptn
                if(T.nodes[v].leaf<0) proc(v,sv); rls(sv);
            } };
        proc(T.root,-1); CK(cudaDeviceSynchronize());
        for(int e=0;e<nedge;e++){ g_df[e]=dfC[edgeV[e]]; g_ddf[e]=ddfC[edgeV[e]]; }
        // per-category rate gradient gradR[c] = w_c * sum_ptn rnum[c][ptn]/L_ptn   (the validated G.4.0b assembly)
        CK(cudaMemcpy(rnumH.data(),d_rnum,(size_t)NCAT*nptn*sizeof(double),cudaMemcpyDeviceToHost));
        for(int c=0;c<NCAT;c++){ long double acc=0; for(int p=0;p<nptn;p++) acc+=rnumH[(size_t)c*nptn+p]*invL[p]; gradR[c]=catWeights[c]*(double)acc; }
        // alpha gradient = sum_c (dr_c/dalpha) * gradR[c]  (dr_c/dalpha by FD of the discretisation)
        double rp[16]; discreteGammaMean(curAlpha+1e-5,NCAT,rp); double ga=0; for(int c=0;c<NCAT;c++) ga+=((rp[c]-catRates[c])/1e-5)*gradR[c];
        lnLout=lnLfirst; galphaOut=ga; };

    double O=-7541976.8529;       // the FULL CPU MLE (LG+G4, alpha=0.9963; .iqtree) — joint (b,alpha) target

    // ===================== alpha-gradient FD check (validates the NEW alpha-grad assembly) =====================
    applyAlpha(0.9963); brlen=mleLen; curAlpha=0.9963; double lnL_chk,ga_an; computeGradient(lnL_chk,ga_an);
    double gnorm_mle=0; for(int e=0;e<nedge;e++) gnorm_mle+=g_df[e]*g_df[e]; gnorm_mle=sqrt(gnorm_mle);
    double best=1e30,bfd=0; for(double e:{1e-3,1e-4,1e-5}){ double lp=evalLnL(mleLen,0.9963+e), lm=evalLnL(mleLen,0.9963-e); double fd=(lp-lm)/(2*e);
        double r=fabs(fd)>1e-3?fabs((ga_an-fd)/fd):fabs(ga_an-fd); if(r<best){best=r;bfd=fd;} }
    printf("\n========== G.4.1b PRE-CHECK ==========\n");
    printf("lnL(.treefile, alpha=0.9963)=%.6f vs full MLE %.6f rel=%.3e ; ||g_b||=%.3e\n",lnL_chk,O,fabs((lnL_chk-O)/O),gnorm_mle);
    printf("alpha-gradient: analytic dlnL/dalpha=%.4e, FD=%.4e, rel=%.3e -> %s (gate 0.01; validates the joint alpha-grad)\n",
        ga_an,bfd,best,(best<0.01?"PASS":"FAIL"));

    // ===================== joint LM diagonal-Newton over (197 branches + alpha); WARM then COLD =====================
    struct OptRes{ double lnL; double alpha; int iters; long sweeps; long evals; int rejects; double gnorm; };
    vector<double> cand(nnodes,0.0);
    auto optimize=[&](vector<double> sb,double sa,const char* tag)->OptRes{
        long sw0=nGradSweeps, ev0=nLnLEval; brlen=sb; curAlpha=sa; double lnL=evalLnL(sb,sa); nLnLEval++;
        double mu=1.0, tol=1e-7; int it=0,nRej=0; bool conv=false; double gnorm=0,startL=lnL; double aPrev=0,gaPrev=0; bool haveSec=false;
        printf("\n---- optimise [%s]: start lnL=%.6f alpha=%.4f (rel %.3e) ----\n",tag,lnL,sa,fabs((lnL-O)/O));
        for(it=1; it<=maxiter; it++){
            vector<double> base=brlen; double baseA=curAlpha; applyAlpha(baseA);
            double lg,ga; computeGradient(lg,ga);
            gnorm=0; for(int e=0;e<nedge;e++) gnorm+=g_df[e]*g_df[e]; gnorm=sqrt(gnorm);
            double ddA = (haveSec && fabs(baseA-aPrev)>1e-9) ? (ga-gaPrev)/(baseA-aPrev) : -1e6;  // secant alpha-curvature
            aPrev=baseA; gaPrev=ga; haveSec=true;
            bool acc=false;
            for(int bt=0; bt<14; bt++){
                cand=base; for(int e=0;e<nedge;e++){ int v=edgeV[e]; double dn=fabs(g_ddf[e])+mu; double nb=base[v]+g_df[e]/dn; if(nb<1e-6)nb=1e-6; if(nb>20.0)nb=20.0; cand[v]=nb; }
                double da=ga/(fabs(ddA)+mu); double ca=baseA+da; if(ca<0.02)ca=0.02; if(ca>50.0)ca=50.0;
                double ln=evalLnL(cand,ca); nLnLEval++;
                if(ln>lnL+1e-9){ double dl=ln-lnL; brlen=cand; curAlpha=ca; lnL=ln; mu=fmax(mu*0.5,1e-9); acc=true; if(dl<tol)conv=true; break; }
                else { mu*=4.0; nRej++; } }
            if(!acc){ brlen=base; curAlpha=baseA; printf("  [%s it %d] no uphill step (mu=%.2e) -> stop\n",tag,it,mu); break; }
            if(it<=6||it%25==0||conv) printf("  [%s it %3d] lnL=%.6f rel=%.3e alpha=%.4f ||g||=%.2e mu=%.2e rej=%d\n",
                tag,it,lnL,fabs((lnL-O)/O),curAlpha,gnorm,mu,nRej);
            if(conv) break; }
        printf("  [%s] DONE: %d iters, %.6f->%.6f (alpha %.4f->%.4f), %ld grad + %ld evals, %d rejects\n",
            tag,it,startL,lnL,sa,curAlpha,nGradSweeps-sw0,nLnLEval-ev0,nRej);
        OptRes r; r.lnL=lnL; r.alpha=curAlpha; r.iters=it; r.sweeps=nGradSweeps-sw0; r.evals=nLnLEval-ev0; r.rejects=nRej; r.gnorm=gnorm; return r; };

    OptRes warm=optimize(mleLen,0.9963,"WARM .treefile a=0.9963");
    vector<double> coldVec(nnodes,0.1);
    OptRes cold=optimize(coldVec,3.0,"COLD b=0.1 a=3.0");

    double relCW=fabs((cold.lnL-warm.lnL)/warm.lnL), relCO=fabs((cold.lnL-O)/O);
    printf("\n========== G.4.1b VERDICT [g4 joint (branches + alpha)] ==========\n");
    printf("(1) COLD reaches the SAME optimum as WARM: cold lnL=%.6f warm lnL=%.6f rel=%.3e -> %s (gate 1e-9)\n",
        cold.lnL,warm.lnL,relCW,(relCW<=1e-9?"PASS":"FAIL"));
    printf("(2) COLD reaches the full CPU MLE %.4f: cold lnL=%.6f rel=%.3e -> %s ; alpha %.4f->%.4f (CPU MLE 0.9963)\n",
        O,cold.lnL,relCO,(relCO<=1e-9?"PASS":"FAIL"),3.0,cold.alpha);
    printf("(3) HEADLINE: COLD start (b=0.1, alpha=3.0 — both far) optimised JOINTLY over %d branches + alpha in\n",nedge);
    printf("    %d joint iterations (%ld grad-sweeps + %ld lnL-evals); alpha folded into the joint step, NO Brent line search.\n",
        cold.iters,cold.sweeps,cold.evals);
    bool PASS=(relCW<=1e-9)&&(relCO<=1e-9);
    printf("VERDICT [g4]: %s — full +G joint MLE (branches+alpha) from a cold start, no alpha-Brent\n",PASS?"PASS":"CHECK");
    printf("==================================================================\n");

    cudaFreeHost(h_tip); cudaFreeHost(h_echild); cudaFreeHost(h_expfac);
    cudaFree(d_echild);cudaFree(d_partial);cudaFree(d_theta);cudaFree(d_patlh);cudaFree(d_pdf);cudaFree(d_pddf);
    cudaFree(d_tip);cudaFree(d_pretmp);cudaFree(d_tipeig);cudaFree(d_expfac);cudaFree(d_prepool);cudaFree(d_rnum);
    return 0;
}
