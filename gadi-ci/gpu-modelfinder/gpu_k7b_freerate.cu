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

int main(int argc, char** argv){
    if (argc<3){ fprintf(stderr,"usage: %s <aln.phy> <tree> [model=g4|r8|r10|g1] [reps]\n",argv[0]); return 1; }
    string alnpath=argv[1],treepath=argv[2]; string model=(argc>3)?argv[3]:"g4"; int reps=(argc>4)?atoi(argv[4]):20;

    // ---- alignment ----
    ifstream af(alnpath); if(!af){fprintf(stderr,"no aln\n");return 2;}
    int ntax=0,nsite=0; { string h; getline(af,h); sscanf(h.c_str(),"%d %d",&ntax,&nsite); }
    vector<string> seqs(ntax); map<string,int> name2tip;
    for (int t=0;t<ntax;t++){ string line; getline(af,line); istringstream ls(line); string nm,sq; ls>>nm>>sq; seqs[t]=sq; name2tip[nm]=t; }
    int nptn=nsite; printf("[aln] ntax=%d nsite=%d\n",ntax,nptn);

    // ---- tree ----
    string ts; { ifstream tf(treepath); stringstream ss; ss<<tf.rdbuf(); ts=ss.str(); }
    string tc; for(char c:ts) if(!isspace((unsigned char)c)) tc+=c;
    Tree T=parse_newick(tc,name2tip); int nnodes=T.nodes.size();
    printf("[tree] nodes=%d root_children=%zu\n",nnodes,T.nodes[T.root].child.size());

    // ---- model ----
    double R[20][20],f[20]; fill_LG(R,f); int NCAT=4; vector<double> catRates,catWeights;
    if (model=="g1"){ NCAT=1; catRates={1.0}; catWeights={1.0}; }
    else if (model=="g4"){ NCAT=4; catRates={0.1362,0.4756,0.9994,2.3887}; catWeights={0.25,0.25,0.25,0.25}; }
    else if (model=="r4"){ NCAT=4; double br[4]={0.10,0.40,1.00,3.00},bw[4]={0.35,0.30,0.25,0.10};   // G.4.0b: LG+R4 (plan-named +R gate)
        catRates.assign(br,br+4); catWeights.assign(bw,bw+4); double sw=0; for(double w:catWeights)sw+=w; for(double&w:catWeights)w/=sw;
        double mean=0; for(int k=0;k<NCAT;k++)mean+=catWeights[k]*catRates[k]; for(int k=0;k<NCAT;k++)catRates[k]/=mean; }
    else if (model=="r8"){ NCAT=8; double br[8]={0.06,0.18,0.36,0.60,0.92,1.40,2.20,4.00},bw[8]={0.07,0.11,0.14,0.17,0.17,0.14,0.11,0.09};
        catRates.assign(br,br+8); catWeights.assign(bw,bw+8); double sw=0; for(double w:catWeights)sw+=w; for(double&w:catWeights)w/=sw;
        double mean=0; for(int k=0;k<NCAT;k++)mean+=catWeights[k]*catRates[k]; for(int k=0;k<NCAT;k++)catRates[k]/=mean; }
    else if (model=="r10"){ NCAT=10; double br[10]={0.05,0.15,0.30,0.50,0.75,1.00,1.40,2.00,3.00,5.00},bw[10]={0.05,0.08,0.10,0.12,0.15,0.15,0.12,0.10,0.08,0.05};
        catRates.assign(br,br+10); catWeights.assign(bw,bw+10); double sw=0; for(double w:catWeights)sw+=w; for(double&w:catWeights)w/=sw;
        double mean=0; for(int k=0;k<NCAT;k++)mean+=catWeights[k]*catRates[k]; for(int k=0;k<NCAT;k++)catRates[k]/=mean; }
    else { fprintf(stderr,"unknown model\n"); return 1; }
    printf("[model] %s NCAT=%d\n",model.c_str(),NCAT);

    // ---- eigendecomposition (IQ-TREE reversible convention) ----
    double Q[20][20]; for(int i=0;i<NS;i++){double row=0; for(int j=0;j<NS;j++){if(i!=j){Q[i][j]=R[i][j]*f[j]; row+=Q[i][j];}} Q[i][i]=-row;}
    double mu=0; for(int i=0;i<NS;i++) mu+=f[i]*(-Q[i][i]); for(int i=0;i<NS;i++) for(int j=0;j<NS;j++) Q[i][j]/=mu;
    double sq[20],B[20][20]; for(int i=0;i<NS;i++) sq[i]=sqrt(f[i]);
    for(int i=0;i<NS;i++) for(int j=0;j<NS;j++) B[i][j]=sq[i]*Q[i][j]/sq[j];
    for(int i=0;i<NS;i++) for(int j=i+1;j<NS;j++){double m=0.5*(B[i][j]+B[j][i]); B[i][j]=B[j][i]=m;}
    double evl[20],V[20][20]; jacobi_eig(B,evl,V);
    vector<double> U(NS*NS),Uinv(NS*NS),evals(NS);
    for(int i=0;i<NS;i++){ evals[i]=evl[i]; for(int j=0;j<NS;j++){ U[i*NS+j]=V[i][j]/sq[i]; Uinv[i*NS+j]=V[j][i]*sq[j]; } }
    vector<double> UinvRowSum(NS,0.0); for(int i=0;i<NS;i++){double s=0; for(int j=0;j<NS;j++) s+=Uinv[i*NS+j]; UinvRowSum[i]=s;}

    // ---- postorder internal nodes + slots; child edge lengths ----
    vector<double> childLen(nnodes,0.0);
    for(int u=0;u<nnodes;u++) for(size_t k=0;k<T.nodes[u].child.size();k++) childLen[T.nodes[u].child[k]]=T.nodes[u].blen[k];
    vector<int> postorder; vector<int> slot(nnodes,-1);
    function<void(int)> dfs=[&](int u){ for(int c:T.nodes[u].child) dfs(c); if(T.nodes[u].leaf<0){ slot[u]=postorder.size(); postorder.push_back(u);} };
    dfs(T.root); int nInternal=postorder.size();

    // central edge = (root, c0); pick first INTERNAL child of root so node_eig is a full subtree partial
    int c0=-1; for(int c:T.nodes[T.root].child) if(T.nodes[c].leaf<0){ c0=c; break; }
    if (c0<0){ fprintf(stderr,"root has no internal child — pick another central edge\n"); return 6; }
    double t0=childLen[c0];
    printf("[edge] central=(root=%d, c0=%d) t0=%.6f ; root other children = %zu\n", T.root,c0,t0,T.nodes[T.root].child.size()-1);

    // ---- echild[child][cat][x][i] = U[x][i]*exp(eval[i]*rate_c*len_child) ----
    size_t ecStride=(size_t)NCAT*NS*NS; vector<double> echild((size_t)nnodes*ecStride,0.0);
    for(int c=0;c<nnodes;c++){ if(c==T.root) continue; for(int cat=0;cat<NCAT;cat++){ double len=childLen[c]*catRates[cat];
        double ex[NS]; for(int i=0;i<NS;i++) ex[i]=exp(evals[i]*len);
        double* e=&echild[(size_t)c*ecStride+(size_t)cat*NS*NS];
        for(int x=0;x<NS;x++) for(int i=0;i<NS;i++) e[x*NS+i]=U[x*NS+i]*ex[i]; } }

    // ---- compact tip states ----
    vector<unsigned char> tip((size_t)ntax*nptn);
    for(int u=0;u<nnodes;u++){ if(T.nodes[u].leaf<0) continue; int lf=T.nodes[u].leaf; const string&s=seqs[lf];
        for(int p=0;p<nptn;p++){ int a=aa_index(s[p]); tip[(size_t)lf*nptn+p]=(unsigned char)((a<0)?NS:a); } }

    // ---- upload ----
    CK(cudaMemcpyToSymbol(c_Uinv,Uinv.data(),sizeof(double)*NS*NS));
    CK(cudaMemcpyToSymbol(c_U,U.data(),sizeof(double)*NS*NS));        // G.4.0: evec for k7_pre step 1
    CK(cudaMemcpyToSymbol(c_UinvRowSum,UinvRowSum.data(),sizeof(double)*NS));
    double *d_echild,*d_partial,*d_dad,*d_theta,*d_patlh,*d_pdf,*d_pddf; unsigned char* d_tip;
    CK(cudaMalloc(&d_echild,echild.size()*sizeof(double))); CK(cudaMemcpy(d_echild,echild.data(),echild.size()*sizeof(double),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_tip,tip.size())); CK(cudaMemcpy(d_tip,tip.data(),tip.size(),cudaMemcpyHostToDevice));
    size_t slotSz=(size_t)NCAT*NS*nptn;
    CK(cudaMalloc(&d_partial,(size_t)nInternal*slotSz*sizeof(double)));
    CK(cudaMalloc(&d_dad,slotSz*sizeof(double)));
    CK(cudaMalloc(&d_theta,slotSz*sizeof(double)));
    CK(cudaMalloc(&d_patlh,(size_t)nptn*sizeof(double)));
    CK(cudaMalloc(&d_pdf,(size_t)nptn*sizeof(double)));
    CK(cudaMalloc(&d_pddf,(size_t)nptn*sizeof(double)));
    printf("[mem] partials %.2f GB\n",(double)nInternal*slotSz*8/1073741824.0);

    int TB=256, GB=(nptn+TB-1)/TB;
    auto childArgs=[&](int u,int excl,int& nch,const double** ec,const double** p,const unsigned char** t){
        nch=0; for(int k=0;k<3;k++){ec[k]=p[k]=nullptr;t[k]=nullptr;}
        for(int c:T.nodes[u].child){ if(c==excl) continue; ec[nch]=d_echild+(size_t)c*ecStride;
            if(T.nodes[c].leaf>=0) t[nch]=d_tip+(size_t)T.nodes[c].leaf*nptn; else p[nch]=d_partial+(size_t)slot[c]*slotSz; nch++; }
    };

    // ===================== G.4.0b — O(depth) recycling + FreeRate rate-grad kill-switch =====================
    bool isFreeRate = (model[0]=='r');
    // (A) tree height for the O(depth) pre-slot pool (live set = root->node path << nnodes)
    int treeH=0; function<void(int,int)> depthDfs=[&](int u,int d){ if(d>treeH)treeH=d; for(int c:T.nodes[u].child) depthDfs(c,d+1); };
    depthDfs(T.root,0);
    int nPool=treeH+2;
    double *d_prepool,*d_pretmp,*d_tipeig,*d_expfac,*d_rnum,*d_invL;
    CK(cudaMalloc(&d_prepool,(size_t)nPool*slotSz*sizeof(double)));   // RECYCLED preorder slots (was nnodes/node)
    CK(cudaMalloc(&d_pretmp, slotSz*sizeof(double)));                 // scratch pre for the central-edge L_ptn / FD recompute
    CK(cudaMalloc(&d_tipeig, slotSz*sizeof(double)));
    CK(cudaMalloc(&d_rnum,(size_t)NCAT*nptn*sizeof(double))); CK(cudaMemset(d_rnum,0,(size_t)NCAT*nptn*sizeof(double)));
    CK(cudaMalloc(&d_invL,(size_t)nptn*sizeof(double)));
    vector<double> expfac((size_t)nnodes*NCAT*NS,0.0);               // exp(eval[i]*rate_c*childLen[node]) (parent-branch factor)
    for(int u=0;u<nnodes;u++){ if(u==T.root) continue; for(int c=0;c<NCAT;c++) for(int i=0;i<NS;i++)
        expfac[(size_t)u*NCAT*NS+c*NS+i]=exp(evals[i]*catRates[c]*childLen[u]); }
    CK(cudaMalloc(&d_expfac,expfac.size()*sizeof(double)));
    CK(cudaMemcpy(d_expfac,expfac.data(),expfac.size()*sizeof(double),cudaMemcpyHostToDevice));
    printf("[mem] partials %.2f GB + pre-pool %d slots(height=%d) %.2f GB   (naive 1/node would be %.2f GB)\n",
        (double)nInternal*slotSz*8/1073741824.0, nPool, treeH, (double)nPool*slotSz*8/1073741824.0,
        (double)nnodes*slotSz*8/1073741824.0);

    // ---- postorder: ALL non-root internal partials (computed ONCE; kept resident — only the PREORDER recycles) ----
    for(int idx=0; idx<nInternal; idx++){ int u=postorder[idx]; if(u==T.root) continue;
        int nch; const double* ec[3]; const double* p[3]; const unsigned char* t[3]; childArgs(u,-1,nch,ec,p,t);
        k1_node<<<GB,TB>>>(nptn,NCAT,d_partial+(size_t)slot[u]*slotSz,nch,ec[0],p[0],t[0],ec[1],p[1],t[1],ec[2],p[2],t[2]); }
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());

    auto sibArg=[&](int w,const double*& ec,const double*& sp,const unsigned char*& st){
        ec=d_echild+(size_t)w*ecStride; sp=nullptr; st=nullptr;
        if(T.nodes[w].leaf>=0) st=d_tip+(size_t)T.nodes[w].leaf*nptn; else sp=d_partial+(size_t)slot[w]*slotSz; };

    // ---- val builder + per-edge evaluator (theta cached per edge) ----
    vector<double> patlh(nptn),pdf(nptn),pddf(nptn);
    auto setValR=[&](const vector<double>& rates,double t){ vector<double> v0(NCAT*NS),v1(NCAT*NS),v2(NCAT*NS);
        for(int c=0;c<NCAT;c++){ double rc=rates[c],pc=catWeights[c]; for(int x=0;x<NS;x++){ double re=rc*evals[x],e=exp(evals[x]*rc*t)*pc;
            v0[c*NS+x]=e; v1[c*NS+x]=re*e; v2[c*NS+x]=re*re*e; } }
        CK(cudaMemcpyToSymbol(c_val0,v0.data(),sizeof(double)*NCAT*NS));
        CK(cudaMemcpyToSymbol(c_val1,v1.data(),sizeof(double)*NCAT*NS));
        CK(cudaMemcpyToSymbol(c_val2,v2.data(),sizeof(double)*NCAT*NS)); };
    auto setVal=[&](double t){ setValR(catRates,t); };
    auto evalAt=[&](double t,double& lnL,double& df,double& ddf){ setVal(t);
        k2_derv<<<GB,TB>>>(nptn,NCAT,d_theta,d_patlh,d_pdf,d_pddf); CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        CK(cudaMemcpy(patlh.data(),d_patlh,nptn*sizeof(double),cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(pdf.data(),d_pdf,nptn*sizeof(double),cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(pddf.data(),d_pddf,nptn*sizeof(double),cudaMemcpyDeviceToHost));
        double L=0,kc=0; for(int p=0;p<nptn;p++){ double y=patlh[p]-kc,tt=L+y; kc=(tt-L)-y; L=tt; }
        double D=0,DD=0; for(int p=0;p<nptn;p++){ D+=pdf[p]; DD+=pddf[p]; } lnL=L; df=D; ddf=DD; };
    auto edgeThetaInto=[&](int v,const double* pre){ const double* pl;        // theta_e = pre (.) pl_v
        if(T.nodes[v].leaf<0) pl=d_partial+(size_t)slot[v]*slotSz;
        else { k_leafeig<<<GB,TB>>>(nptn,NCAT,d_tip+(size_t)T.nodes[v].leaf*nptn,d_tipeig); pl=d_tipeig; }
        k2_theta<<<GB,TB>>>(nptn,NCAT*NS,pl,pre,d_theta); };

    // ---- validation accumulators (edge-invariance + df FD + branch grad + Sum b_e*df_e) ----
    map<string,double> oracle={{"g4",-7541976.9391},{"r8",-7556251.9185},{"r10",-7554280.5776},{"g1",-7974816.4323}};
    double O=oracle.count(model)?oracle[model]:0.0;
    double worstInv=0,worstOra=0,worstDf=0,gradNorm=0,sumBdf=0; int nInt=0,nLeaf=0;
    double dfGate=(model=="g1")?1e-6:3e-3;
    double refLnL=0; bool haveRef=false;
    int cedge=-1; double cdf=0;
    vector<double> grad; vector<int> gv;

    // O(depth) pre-slot pool: a stack of free slot indices; acquire on entering a node, release after its subtree.
    vector<int> freeSlots; for(int s=nPool-1;s>=0;s--) freeSlots.push_back(s);
    int liveNow=0, peakLive=0;
    auto acquire=[&]()->int{ int s=freeSlots.back(); freeSlots.pop_back(); if(++liveNow>peakLive)peakLive=liveNow; return s; };
    auto release=[&](int s){ freeSlots.push_back(s); --liveNow; };

    // consume pre_v (already in slot `pre`): branch df (lnL-inv + FD) + FreeRate rate-grad numerator
    auto doEdge=[&](int u,int v,const double* pre){
        double bv=childLen[v];
        edgeThetaInto(v,pre); CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        double le,dfe,ddfe; evalAt(bv,le,dfe,ddfe);                  // val now @bv, theta = this edge
        if(isFreeRate){                                             // (B) accumulate Sum_e b_e*qp_e[k] (val1 @bv reused)
            vector<double> rs(NCAT); for(int k=0;k<NCAT;k++) rs[k]=bv/(catRates[k]*catWeights[k]);
            CK(cudaMemcpyToSymbol(c_rscale,rs.data(),sizeof(double)*NCAT));
            k_ratenum<<<GB,TB>>>(nptn,NCAT,d_theta,d_rnum); CK(cudaDeviceSynchronize()); CK(cudaGetLastError()); }
        if(!haveRef){ refLnL=le; haveRef=true; }                    // internal edge-invariance (works w/o an oracle, e.g. r4)
        double rl=fabs((le-refLnL)/refLnL); if(rl>worstInv) worstInv=rl;
        if(O){ double ro=fabs((le-O)/O); if(ro>worstOra) worstOra=ro; }
        double tfd=bv+0.2, lc,dft,d3; evalAt(tfd,lc,dft,d3);         // FD off-optimum (df~0 at bv is ill-conditioned)
        double rdf=1e30; for(double e:{1e-2,1e-3,1e-4}){ double lp,lm; evalAt(tfd+e,lp,d3,d3); evalAt(tfd-e,lm,d3,d3);
            double dfFD=(lp-lm)/(2*e); double r=fabs(dfFD)>1e-3?fabs((dft-dfFD)/dfFD):fabs(dft-dfFD); if(r<rdf) rdf=r; }
        if(rdf>worstDf) worstDf=rdf;
        grad.push_back(dfe); gv.push_back(v); gradNorm+=dfe*dfe; sumBdf+=bv*dfe;
        if(u==T.root && v==c0){ cedge=v; cdf=dfe; }
        if(T.nodes[v].leaf<0) nInt++; else nLeaf++;
    };
    // interleaved preorder DFS with slot recycling: seed root children via k1_node, descend via k7_pre.
    function<void(int,int)> process=[&](int u,int su){
        for(int v:T.nodes[u].child){
            int sv=acquire(); double* pre=d_prepool+(size_t)sv*slotSz;
            if(u==T.root){ int nch; const double* ec[3]; const double* p[3]; const unsigned char* tp[3]; childArgs(T.root,v,nch,ec,p,tp);
                k1_node<<<GB,TB>>>(nptn,NCAT,pre,nch,ec[0],p[0],tp[0],ec[1],p[1],tp[1],ec[2],p[2],tp[2]); }
            else { const double* ec[2]={nullptr,nullptr}; const double* sp[2]={nullptr,nullptr}; const unsigned char* st[2]={nullptr,nullptr}; int ns=0;
                for(int w:T.nodes[u].child){ if(w==v||ns>=2) continue; sibArg(w,ec[ns],sp[ns],st[ns]); ns++; }
                k7_pre<<<GB,TB>>>(nptn,NCAT,pre,d_prepool+(size_t)su*slotSz,d_expfac+(size_t)u*NCAT*NS,ns,ec[0],sp[0],st[0],ec[1],sp[1],st[1]); }
            CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
            doEdge(u,v,pre);                                        // consume pre_v BEFORE recursing (theta/rnum reused)
            if(T.nodes[v].leaf<0) process(v,sv);                    // pre_v stays live for grandchildren
            release(sv);
        }
    };
    process(T.root,-1); CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    gradNorm=sqrt(gradNorm);

    printf("\n========== G.4.0b (A) O(depth)-recycled all-branch gradient [%s] ==========\n",model.c_str());
    printf("edges: %d internal + %d leaf = %d ; pre-pool peak=%d / %d slots (tree height=%d) -> O(depth) confirmed\n",
        nInt,nLeaf,nInt+nLeaf,peakLive,nPool,treeH);
    printf("(1) lnL EDGE-INVARIANCE (self, ref=%.4f): worst rel=%.3e -> %s  (gate 1e-9; VALIDATES pre_v under recycling)\n",
        refLnL,worstInv,(worstInv<=1e-9?"PASS":"FAIL"));
    if(O) printf("    + vs oracle %.4f: worst rel=%.3e -> %s\n",O,worstOra,(worstOra<=1e-9?"PASS":"FAIL"));
    printf("(2) all-branch df FD-validation: worst rel=%.3e -> %s  (gate %.0e)\n",
        worstDf,(worstDf<dfGate?"PASS":"FAIL"),dfGate);
    if(cedge>=0) printf("(3) central edge (root->c0=%d): df=%.6e  (== G.1.2/G.2.1a-validated K2 path)\n",cedge,cdf);
    printf("|grad_b|_2=%.6e over %zu branches; Sum_e b_e*df_e=%.6e\n",gradNorm,grad.size(),sumBdf);

    // ===================== (B) FreeRate rate-parameter gradient OVERFLOW KILL-SWITCH =====================
    if(isFreeRate){
        // L_ptn (edge-invariant) at the central edge from the intact BASE postorder (process() never touched d_partial/d_echild)
        { int nch; const double* ec[3]; const double* p[3]; const unsigned char* tp[3]; childArgs(T.root,c0,nch,ec,p,tp);
          k1_node<<<GB,TB>>>(nptn,NCAT,d_pretmp,nch,ec[0],p[0],tp[0],ec[1],p[1],tp[1],ec[2],p[2],tp[2]); CK(cudaDeviceSynchronize());
          k2_theta<<<GB,TB>>>(nptn,NCAT*NS,d_partial+(size_t)slot[c0]*slotSz,d_pretmp,d_theta);
          setVal(t0); k2_derv<<<GB,TB>>>(nptn,NCAT,d_theta,d_patlh,d_pdf,d_pddf); CK(cudaDeviceSynchronize());
          CK(cudaMemcpy(patlh.data(),d_patlh,nptn*sizeof(double),cudaMemcpyDeviceToHost)); }
        vector<double> invL(nptn); double maxInvL=0; for(int p=0;p<nptn;p++){ invL[p]=exp(-patlh[p]); if(invL[p]>maxInvL)maxInvL=invL[p]; }
        // assemble dlnL/dr_k = w_k * Sum_ptn rnum[k][ptn]/L_ptn  (the qp~1e-34 * 1/L_p~1e34 -> O(1) overflow test)
        vector<double> rnum((size_t)NCAT*nptn); CK(cudaMemcpy(rnum.data(),d_rnum,(size_t)NCAT*nptn*sizeof(double),cudaMemcpyDeviceToHost));
        vector<double> gradR(NCAT,0.0); double maxAbs=0,sumRkGr=0,maxRnum=0,maxRatio=0; bool allFinite=true;
        for(int k=0;k<NCAT;k++){ long double acc=0;
            for(int p=0;p<nptn;p++){ double rn=rnum[(size_t)k*nptn+p], ratio=rn*invL[p];
                if(!std::isfinite(ratio)) allFinite=false;
                if(fabs(ratio)>maxRatio)maxRatio=fabs(ratio); if(fabs(rn)>maxRnum)maxRnum=fabs(rn); acc+=ratio; }
            gradR[k]=catWeights[k]*(double)acc; if(!std::isfinite(gradR[k]))allFinite=false;
            if(fabs(gradR[k])>maxAbs)maxAbs=fabs(gradR[k]); sumRkGr+=catRates[k]*gradR[k]; }

        printf("\n========== G.4.0b (B) FreeRate rate-gradient OVERFLOW KILL-SWITCH [%s] ==========\n",model.c_str());
        printf("per-pattern terms: max|1/L_ptn|=%.3e  max|rnum|=%.3e  max|rnum/L_ptn|=%.3e   (Mode-L overflowed at ~1e54)\n",
            maxInvL,maxRnum,maxRatio);
        printf("dlnL/dr_k:"); for(int k=0;k<NCAT;k++) printf(" r%d(%.3f)=%.4e",k,catRates[k],gradR[k]); printf("\n");
        bool b1 = allFinite && maxAbs<1e8;
        printf("(B1) FINITE & bounded (NO OVERFLOW): max|dlnL/dr_k|=%.4e -> %s  (kill-switch: finite & <1e8)\n",maxAbs,b1?"PASS":"FAIL");
        double rel2 = fabs(sumBdf)>0 ? fabs((sumRkGr-sumBdf)/sumBdf) : fabs(sumRkGr-sumBdf);
        bool b2 = rel2<1e-6;
        printf("(B2) scaling identity  Sum_k r_k*gr_k=%.6e == Sum_e b_e*gb_e=%.6e  rel=%.3e -> %s  (exact tie to validated branch grad; gate 1e-6)\n",
            sumRkGr,sumBdf,rel2,b2?"PASS":"FAIL");
        // (B3) FD-validate dlnL/dr_k by full-lnL recompute (perturb one rate; rebuild echild+postorder; central-edge lnL)
        auto fullLnL=[&](const vector<double>& rates)->double{
            vector<double> ec((size_t)nnodes*ecStride,0.0);
            for(int c=0;c<nnodes;c++){ if(c==T.root) continue; for(int cat=0;cat<NCAT;cat++){ double len=childLen[c]*rates[cat];
                double ex[NS]; for(int i=0;i<NS;i++) ex[i]=exp(evals[i]*len); double* e=&ec[(size_t)c*ecStride+(size_t)cat*NS*NS];
                for(int x=0;x<NS;x++) for(int i=0;i<NS;i++) e[x*NS+i]=U[x*NS+i]*ex[i]; } }
            CK(cudaMemcpy(d_echild,ec.data(),ec.size()*sizeof(double),cudaMemcpyHostToDevice));
            for(int idx=0;idx<nInternal;idx++){ int u=postorder[idx]; if(u==T.root)continue;
                int nch; const double* e2[3]; const double* p2[3]; const unsigned char* t2[3]; childArgs(u,-1,nch,e2,p2,t2);
                k1_node<<<GB,TB>>>(nptn,NCAT,d_partial+(size_t)slot[u]*slotSz,nch,e2[0],p2[0],t2[0],e2[1],p2[1],t2[1],e2[2],p2[2],t2[2]); }
            CK(cudaDeviceSynchronize());
            int nch; const double* e2[3]; const double* p2[3]; const unsigned char* t2[3]; childArgs(T.root,c0,nch,e2,p2,t2);
            k1_node<<<GB,TB>>>(nptn,NCAT,d_pretmp,nch,e2[0],p2[0],t2[0],e2[1],p2[1],t2[1],e2[2],p2[2],t2[2]); CK(cudaDeviceSynchronize());
            k2_theta<<<GB,TB>>>(nptn,NCAT*NS,d_partial+(size_t)slot[c0]*slotSz,d_pretmp,d_theta);
            setValR(rates,t0); k2_derv<<<GB,TB>>>(nptn,NCAT,d_theta,d_patlh,d_pdf,d_pddf); CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(patlh.data(),d_patlh,nptn*sizeof(double),cudaMemcpyDeviceToHost));
            double L=0,kc=0; for(int p=0;p<nptn;p++){ double y=patlh[p]-kc,tt=L+y; kc=(tt-L)-y; L=tt; } return L; };
        double worstR=0;
        printf("(B3) FD-validate dlnL/dr_k (full-lnL recompute, swept eps):\n");
        for(int k=0;k<NCAT;k++){ double best=1e30,bestfd=0;
            for(double e:{1e-3,1e-4,1e-5}){ vector<double> rp=catRates,rm=catRates; rp[k]+=e; rm[k]-=e;
                double lp=fullLnL(rp), lm=fullLnL(rm); double fd=(lp-lm)/(2*e);
                double r=fabs(fd)>1e-3?fabs((gradR[k]-fd)/fd):fabs(gradR[k]-fd); if(r<best){best=r;bestfd=fd;} }
            if(best>worstR)worstR=best;
            printf("   r%d=%.4f: analytic=%.4e FD=%.4e rel=%.3e %s\n",k,catRates[k],gradR[k],bestfd,best,(best<0.01?"ok":"OFF")); }
        bool b3 = worstR<0.01;
        printf("(B3) worst rate-grad FD rel=%.3e -> %s  (gate |G-ratio|<0.01, the Mode-L FDCHECK that read 1e54)\n",worstR,b3?"PASS":"FAIL");
        printf("KILL-SWITCH VERDICT [%s]: %s\n", model.c_str(),
            (b1&&b2&&b3) ? "PASS — unscaled GPU eigen path does NOT overflow; +R gradient FINITE + IDENTITY-TIED + FD-VALID"
                         : "FAIL — investigate (overflow / identity / FD)");
    }
    printf("============================================================\n");

    cudaFree(d_echild);cudaFree(d_partial);cudaFree(d_dad);cudaFree(d_theta);cudaFree(d_patlh);cudaFree(d_pdf);cudaFree(d_pddf);cudaFree(d_tip);
    cudaFree(d_prepool);cudaFree(d_pretmp);cudaFree(d_tipeig);cudaFree(d_expfac);cudaFree(d_rnum);cudaFree(d_invL);
    return 0;
}
