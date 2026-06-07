// gpu_k3_graph.cu — Phase G.1.3: CUDA-Graph capture of the postorder lnL sweep + on-device echild rebuild.
//
// Extends the validated K1 (gpu_k1_lnl.cu) postorder log-likelihood. Two new capabilities, both required to
// take the GPU off the host's critical path so an optimiser can replay the whole sweep without re-feeding it:
//   (1) build_echild: rebuild echild = U*exp(eval*rate*t) ON DEVICE from a device-resident branch-length
//       buffer d_brlen (K1 built echild on the host once). This is what lets a branch-length change flow
//       through the captured graph with no host work.
//   (2) CUDA graph: capture [brlen H2D node -> build_echild -> blocksum memset -> 98x k1_node postorder ->
//       reduce_patlh -> reduce_final -> lnL D2H node] on a NON-default stream, instantiate once, and REPLAY
//       per branch-length change with a single cudaGraphLaunch (no re-capture, no SetParams) — the kernel
//       node params (pointers/grids) never change; only d_brlen contents do, read from global memory.
//
// Design verified by a 3-auditor adversarial sweep (CUDA-graph semantics / FP64 numerics / gate intent).
// Load-bearing fixes folded in:
//  * build_echild MUST mirror the HOST operation grouping (gpu_k1_lnl.cu:263-266): len = brlen*catRate FIRST
//    (single rounding), then exp(eval*len), then U*ex. The bare 3-factor product exp(eval*brlen*catRate) =
//    (eval*brlen)*catRate bit-differs from the host (FP64 mul non-associative) and fails the patlh bit-identity.
//  * NO synchronizing CUDA API inside the capture region (no cudaMemcpyToSymbol / cudaMemcpy / cudaMemGetInfo /
//    cudaDeviceSynchronize / cudaMalloc): only kernel launches + cudaMemcpyAsync/cudaMemsetAsync ON capStream.
//    Model constants (c_Uinv/c_UinvRowSum/c_freq/c_catw) are set ONCE before capture.
//  * Capture mode = cudaStreamCaptureModeGlobal (strictest; surfaces any stray sync call) — NOT ThreadLocal.
//  * Reduction = block-local shared-mem PAIRWISE -> d_blocksum[nblocks] -> single-block PAIRWISE final.
//    NOT atomicAdd (non-deterministic) and NOT a long sequential accumulator (worst-case rel ~2e-11 would fail
//    the rel<1e-6 oracle gate). d_blocksum is zeroed by a CAPTURED cudaMemsetAsync node.
//  * Replay race fix (PATTERN A): the d_brlen H2D is the FIRST captured node, copying from a PINNED host
//    staging buffer; each replay rewrites the pinned buffer then launches — strictly ordered before build_echild.
//  * Optimiser bracket bounded to [1e-6, 10] + isfinite(lnL) guard (UNSCALED FP64: exp(-745)=0 -> log=-inf trap).
//
// SCOPE HONESTY: K3 captures a FULL-tree sweep (rebuild ALL echild + all 98 nodes every replay) — the honest
// worst-case baseline, NOT the production optimizeAllBranches dirty-path / K2-cached-theta per-edge hot loop
// (that is K2 (G.1.2) per-edge + a G.2 refinement). "materially faster" is reported as a CURVE vs pattern
// count: AA-100K is compute-bound (~38ms g4) => wall ~parity; the win is launch/API collapse (98->1) and the
// device-resident-brlen capability, demonstrated at small (launch-bound) pattern counts.
//
// Build (gpuvolta job): module load cuda/12.5.1 gcc/12.2.0
//   nvcc -O3 -std=c++17 -arch=sm_70 -lineinfo gpu_k3_graph.cu -o gpu_k3_graph     (NO fast-math, FP64)
// Run:  ./gpu_k3_graph <aln.phy> <tree> [model=g4|r8|r10|g1] [reps] [ptncap=0(all)] [multiBranch=0]
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

// ===================== CPU scaffolding (lifted verbatim from gpu_k1_lnl.cu, BEAGLE-free) =====================
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

// ===================== device constant model data (set ONCE, never inside capture) =====================
__constant__ double c_Uinv[NS*NS];      // U^-1 (inv_evec)
__constant__ double c_UinvRowSum[NS];   // row sums of U^-1 (ambiguous/unknown tip)
__constant__ double c_freq[NS];         // state frequencies (root reduction)
__constant__ double c_catw[MAXCAT];     // category weights (prop_c)

// per-child probability-space contribution (verbatim from K1)
__device__ __forceinline__ void accum_child(double* prod, int c, int ptn, int nptn,
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
            for (int i=0;i<NS;i++){ double Li = (s<NS)? c_Uinv[i*NS+s] : c_UinvRowSum[i]; v += ecc[x*NS+i]*Li; }
            prod[x]*=v; }
    }
}

// one internal node (or root); one thread per pattern. VERBATIM from gpu_k1_lnl.cu (isRoot writes patlh).
__global__ void k1_node(int nptn, int ncat, int isRoot, double* __restrict__ out, double* __restrict__ patlh,
        int nchild,
        const double* ec0, const double* p0, const unsigned char* t0,
        const double* ec1, const double* p1, const unsigned char* t1,
        const double* ec2, const double* p2, const unsigned char* t2) {
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    double lh = 0.0;
    for (int c=0;c<ncat;c++){
        double prod[NS];
        #pragma unroll
        for (int x=0;x<NS;x++) prod[x]=1.0;
        accum_child(prod,c,ptn,nptn,ec0,p0,t0);
        if (nchild>1) accum_child(prod,c,ptn,nptn,ec1,p1,t1);
        if (nchild>2) accum_child(prod,c,ptn,nptn,ec2,p2,t2);
        if (isRoot){
            double s=0.0;
            #pragma unroll
            for (int x=0;x<NS;x++) s += c_freq[x]*prod[x];
            lh += c_catw[c]*s;
        } else {
            double* o = out + (size_t)(c*NS)*nptn + ptn;
            #pragma unroll
            for (int r=0;r<NS;r++){ double v=0.0;
                #pragma unroll
                for (int x=0;x<NS;x++) v += c_Uinv[r*NS+x]*prod[x];
                o[(size_t)r*nptn]=v; }
        }
    }
    if (isRoot) patlh[ptn] = log(fabs(lh));
}

// NEW (G.1.3): rebuild echild on device from d_brlen. One block per child node; NS*NS threads (thread = x*NS+i).
// CRITICAL: mirror the host grouping (gpu_k1_lnl.cu:263-266) EXACTLY — len = brlen*rate FIRST (single
// rounding), then exp(eval*len), then U*ex. Do NOT write exp(eval*brlen*rate) (that groups as (eval*brlen)*rate
// and bit-differs from the host => fails the patlh bit-identity gate).
__global__ void build_echild(const double* __restrict__ d_brlen, const double* __restrict__ d_eval,
        const double* __restrict__ d_U, const double* __restrict__ d_catRates,
        int ncat, int nnodes, int rootId, double* __restrict__ d_echild) {
    int c = blockIdx.x;
    if (c >= nnodes || c == rootId) return;          // root has no parent edge — skip, exactly like the host
    int tid = threadIdx.x; if (tid >= NS*NS) return;
    int x = tid / NS, i = tid % NS;
    double bl  = d_brlen[c];
    double Uxi = d_U[x*NS + i];
    double ev  = d_eval[i];
    size_t base = (size_t)c * (size_t)ncat * NS * NS;
    for (int cat=0; cat<ncat; cat++){
        double len = bl * d_catRates[cat];           // == host len = childLen[c]*catRates[cat]
        double ex  = exp(ev * len);                  // == host ex[i] = exp(evals[i]*len)
        d_echild[base + (size_t)cat*NS*NS + x*NS + i] = Uxi * ex;   // == host e[x*NS+i] = U[x*NS+i]*ex[i]
    }
}

// NEW (G.1.3): deterministic block-local pairwise reduction of patlh -> one partial per block.
__global__ void reduce_patlh(const double* __restrict__ patlh, int nptn, double* __restrict__ blocksum) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i   = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = (i < nptn) ? patlh[i] : 0.0;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1){
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) blocksum[blockIdx.x] = sdata[0];
}
// final pairwise reduction over the (~391) block partials -> scalar lnL. Single block.
__global__ void reduce_final(const double* __restrict__ blocksum, int n, double* __restrict__ out) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    double v = 0.0;
    for (int i = tid; i < n; i += blockDim.x) v += blocksum[i];   // each thread sums <=ceil(n/blockDim) partials
    sdata[tid] = v;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1){
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[0] = sdata[0];
}

// golden-section maximiser of f on [a,b] to bracket tol (derivative-free; for the full-sweep lnL).
static double golden_max(const function<double(double)>& f, double a, double b, double tol, int maxit){
    const double gr = (sqrt(5.0)-1.0)/2.0;          // 0.6180339887...
    double c = b - gr*(b-a), d = a + gr*(b-a);
    double fc = f(c), fd = f(d);
    for (int it=0; it<maxit && (b-a)>tol; it++){
        if (fc > fd){ b=d; d=c; fd=fc; c=b-gr*(b-a); fc=f(c); }
        else        { a=c; c=d; fc=fd; d=a+gr*(b-a); fd=f(d); }
    }
    return 0.5*(a+b);
}

int main(int argc, char** argv){
    if (argc<3){ fprintf(stderr,"usage: %s <aln.phy> <tree> [model=g4|r8|r10|g1] [reps] [ptncap=0] [multiBranch=0]\n",argv[0]); return 1; }
    string alnpath=argv[1], treepath=argv[2];
    string model=(argc>3)?argv[3]:"g4";
    int reps   = (argc>4)?atoi(argv[4]):20;
    int ptncap = (argc>5)?atoi(argv[5]):0;          // 0 = all sites; >0 = subsample first N (TIMING-ONLY)
    int doMulti= (argc>6)?atoi(argv[6]):0;          // 1 = run V6 multi-branch convergence sweep

    // ---- alignment (sequential PHYLIP) ----
    ifstream af(alnpath); if(!af){ fprintf(stderr,"no aln %s\n",alnpath.c_str()); return 2; }
    int ntax=0,nsite=0; { string h; getline(af,h); sscanf(h.c_str(),"%d %d",&ntax,&nsite); }
    vector<string> seqs(ntax); map<string,int> name2tip;
    for (int t=0;t<ntax;t++){ string line; if(!getline(af,line)){fprintf(stderr,"aln short\n");return 2;}
        istringstream ls(line); string nm,sq; ls>>nm>>sq; seqs[t]=sq; name2tip[nm]=t; }
    int nptn = nsite;
    bool timingOnly = false;
    if (ptncap>0 && ptncap<nsite){ nptn=ptncap; timingOnly=true; }   // subsample for the launch-bound curve point
    printf("[aln] ntax=%d nsite=%d nptn=%d%s\n", ntax, nsite, nptn, timingOnly?"  (TIMING-ONLY subsample)":"");

    // ---- tree ----
    string ts; { ifstream tf(treepath); stringstream ss; ss<<tf.rdbuf(); ts=ss.str(); }
    string tc; for(char c:ts) if(!isspace((unsigned char)c)) tc+=c;
    Tree T=parse_newick(tc,name2tip);
    int nnodes=T.nodes.size();
    printf("[tree] nodes=%d root_children=%zu\n", nnodes, T.nodes[T.root].child.size());

    // ---- model: freqs + category rates/weights (verbatim from K1) ----
    double R[20][20], f[20]; fill_LG(R,f);
    int NCAT=4; vector<double> catRates, catWeights;
    if (model=="g1"){ NCAT=1; catRates={1.0}; catWeights={1.0}; }
    else if (model=="g4"){ NCAT=4; catRates={0.1362,0.4756,0.9994,2.3887}; catWeights={0.25,0.25,0.25,0.25}; }
    else if (model=="r8"){ NCAT=8; double br[8]={0.06,0.18,0.36,0.60,0.92,1.40,2.20,4.00},bw[8]={0.07,0.11,0.14,0.17,0.17,0.14,0.11,0.09};
        catRates.assign(br,br+8); catWeights.assign(bw,bw+8); double sw=0; for(double w:catWeights)sw+=w; for(double&w:catWeights)w/=sw;
        double mean=0; for(int k=0;k<NCAT;k++)mean+=catWeights[k]*catRates[k]; for(int k=0;k<NCAT;k++)catRates[k]/=mean; }
    else if (model=="r10"){ NCAT=10; double br[10]={0.05,0.15,0.30,0.50,0.75,1.00,1.40,2.00,3.00,5.00},bw[10]={0.05,0.08,0.10,0.12,0.15,0.15,0.12,0.10,0.08,0.05};
        catRates.assign(br,br+10); catWeights.assign(bw,bw+10); double sw=0; for(double w:catWeights)sw+=w; for(double&w:catWeights)w/=sw;
        double mean=0; for(int k=0;k<NCAT;k++)mean+=catWeights[k]*catRates[k]; for(int k=0;k<NCAT;k++)catRates[k]/=mean; }
    else { fprintf(stderr,"unknown model '%s'\n",model.c_str()); return 1; }
    printf("[model] %s NCAT=%d\n", model.c_str(), NCAT);

    // ---- LG reversible eigendecomposition (U/Lambda/U^-1, IQ-TREE convention) — verbatim from K1 ----
    double Q[20][20];
    for (int i=0;i<NS;i++){ double row=0; for(int j=0;j<NS;j++){ if(i!=j){ Q[i][j]=R[i][j]*f[j]; row+=Q[i][j]; } } Q[i][i]=-row; }
    double mu=0; for(int i=0;i<NS;i++) mu+=f[i]*(-Q[i][i]);
    for (int i=0;i<NS;i++) for(int j=0;j<NS;j++) Q[i][j]/=mu;
    double sq[20],B[20][20]; for(int i=0;i<NS;i++) sq[i]=sqrt(f[i]);
    for (int i=0;i<NS;i++) for(int j=0;j<NS;j++) B[i][j]=sq[i]*Q[i][j]/sq[j];
    for (int i=0;i<NS;i++) for(int j=i+1;j<NS;j++){ double m=0.5*(B[i][j]+B[j][i]); B[i][j]=B[j][i]=m; }
    double evl[20],V[20][20]; jacobi_eig(B,evl,V);
    vector<double> U(NS*NS), Uinv(NS*NS), evals(NS);
    for (int i=0;i<NS;i++){ evals[i]=evl[i]; for(int j=0;j<NS;j++){ U[i*NS+j]=V[i][j]/sq[i]; Uinv[i*NS+j]=V[j][i]*sq[j]; } }
    vector<double> UinvRowSum(NS,0.0); for(int i=0;i<NS;i++){ double s=0; for(int j=0;j<NS;j++) s+=Uinv[i*NS+j]; UinvRowSum[i]=s; }

    // ---- postorder internal nodes + slots; per-child edge lengths ----
    vector<double> childLen(nnodes,0.0);
    for (int u=0;u<nnodes;u++) for(size_t k=0;k<T.nodes[u].child.size();k++) childLen[T.nodes[u].child[k]]=T.nodes[u].blen[k];
    vector<int> postorder; vector<int> slot(nnodes,-1);
    function<void(int)> dfs=[&](int u){ for(int c:T.nodes[u].child) dfs(c); if(T.nodes[u].leaf<0){ slot[u]=postorder.size(); postorder.push_back(u);} };
    dfs(T.root);
    int nInternal=postorder.size();
    printf("[post] internal nodes=%d (root=%d slot=%d)\n", nInternal, T.root, slot[T.root]);

    // central edge = (root, first INTERNAL child of root) for the single-branch demo (matches K2)
    int c0=-1; for(int c:T.nodes[T.root].child) if(T.nodes[c].leaf<0){ c0=c; break; }
    if (c0<0){ fprintf(stderr,"root has no internal child\n"); return 6; }
    double tc0 = childLen[c0];

    // ---- host echild builder (reference + naive path) — mirrors gpu_k1_lnl.cu:259-268 ----
    size_t ecStride = (size_t)NCAT*NS*NS;
    vector<double> echild((size_t)nnodes*ecStride, 0.0);
    auto hostBuildEchild=[&](const vector<double>& brl, vector<double>& ec){
        for (int c=0;c<nnodes;c++){ if(c==T.root) continue;
            for (int cat=0;cat<NCAT;cat++){ double len=brl[c]*catRates[cat];
                double ex[NS]; for(int i=0;i<NS;i++) ex[i]=exp(evals[i]*len);
                double* e=&ec[(size_t)c*ecStride + (size_t)cat*NS*NS];
                for (int x=0;x<NS;x++) for(int i=0;i<NS;i++) e[x*NS+i]=U[x*NS+i]*ex[i]; } }
    };

    // ---- compact tip states[leaf][ptn] ----
    vector<unsigned char> tip((size_t)ntax*nptn);
    for (int u=0;u<nnodes;u++){ if(T.nodes[u].leaf<0) continue; int lf=T.nodes[u].leaf; const string& s=seqs[lf];
        for (int p=0;p<nptn;p++){ int a=aa_index(s[p]); tip[(size_t)lf*nptn+p]=(unsigned char)((a<0)?NS:a); } }

    // ---- device upload (all OUTSIDE any capture region) ----
    CK(cudaMemcpyToSymbol(c_Uinv, Uinv.data(), sizeof(double)*NS*NS));
    CK(cudaMemcpyToSymbol(c_UinvRowSum, UinvRowSum.data(), sizeof(double)*NS));
    CK(cudaMemcpyToSymbol(c_freq, f, sizeof(double)*NS));
    CK(cudaMemcpyToSymbol(c_catw, catWeights.data(), sizeof(double)*NCAT));
    double *d_echild=nullptr,*d_partial=nullptr,*d_patlh=nullptr; unsigned char *d_tip=nullptr;
    double *d_U=nullptr,*d_eval=nullptr,*d_catRates=nullptr,*d_brlen=nullptr,*d_blocksum=nullptr,*d_lnL=nullptr;
    CK(cudaMalloc(&d_echild, echild.size()*sizeof(double)));
    CK(cudaMalloc(&d_tip, tip.size()));
    CK(cudaMemcpy(d_tip, tip.data(), tip.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_U, sizeof(double)*NS*NS));        CK(cudaMemcpy(d_U, U.data(), sizeof(double)*NS*NS, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_eval, sizeof(double)*NS));        CK(cudaMemcpy(d_eval, evals.data(), sizeof(double)*NS, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_catRates, sizeof(double)*NCAT));  CK(cudaMemcpy(d_catRates, catRates.data(), sizeof(double)*NCAT, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_brlen, sizeof(double)*nnodes));
    size_t slotSz=(size_t)NCAT*NS*nptn;
    CK(cudaMalloc(&d_partial, (size_t)nInternal*slotSz*sizeof(double)));
    CK(cudaMalloc(&d_patlh, (size_t)nptn*sizeof(double)));
    int TB=256, GB=(nptn+TB-1)/TB;                    // GB = #blocks for the sweep AND #block-partials for reduce
    CK(cudaMalloc(&d_blocksum, (size_t)GB*sizeof(double)));
    CK(cudaMalloc(&d_lnL, sizeof(double)));
    double *h_brlen_pinned=nullptr, *h_lnL_pinned=nullptr;
    CK(cudaHostAlloc((void**)&h_brlen_pinned, sizeof(double)*nnodes, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**)&h_lnL_pinned, sizeof(double), cudaHostAllocDefault));
    printf("[mem] partials arena = %.2f GB (%d slots x %zu doubles)\n",
           (double)nInternal*slotSz*8/1073741824.0, nInternal, slotSz);

    // ---- per-internal-node child descriptors (host); pointers fixed across replays ----
    struct Desc { int isRoot,nchild; const double* ec[3]; const double* p[3]; const unsigned char* t[3]; double* out; };
    vector<Desc> desc(nInternal);
    for (int idx=0; idx<nInternal; idx++){ int u=postorder[idx]; Desc& D=desc[idx];
        D.isRoot=(u==T.root)?1:0; D.nchild=T.nodes[u].child.size(); if(D.nchild>3){fprintf(stderr,"node deg %d >3\n",D.nchild);return 6;}
        D.out = D.isRoot? nullptr : (d_partial + (size_t)slot[u]*slotSz);
        for (int k=0;k<3;k++){ D.ec[k]=nullptr; D.p[k]=nullptr; D.t[k]=nullptr; }
        for (int k=0;k<D.nchild;k++){ int c=T.nodes[u].child[k];
            D.ec[k]=d_echild + (size_t)c*ecStride;
            if (T.nodes[c].leaf>=0) D.t[k]=d_tip + (size_t)T.nodes[c].leaf*nptn;
            else                    D.p[k]=d_partial + (size_t)slot[c]*slotSz;
        }
    }
    // launch the 98-node postorder sweep on stream `st` (default 0 = naive path; capStream during capture)
    auto launchSweep=[&](cudaStream_t st){
        for (int idx=0; idx<nInternal; idx++){ Desc& D=desc[idx];
            k1_node<<<GB,TB,0,st>>>(nptn,NCAT,D.isRoot,D.out,d_patlh,D.nchild,
                D.ec[0],D.p[0],D.t[0], D.ec[1],D.p[1],D.t[1], D.ec[2],D.p[2],D.t[2]);
        }
    };

    printf("\n========== G.1.3 K3 — CUDA-graph postorder sweep + on-device echild ==========\n");

    // ================= V0: standalone build_echild bit-reproduction gate (BEFORE any capture) =================
    auto echildGate=[&](const vector<double>& brl, const char* tag)->bool{
        for (int c=0;c<nnodes;c++) h_brlen_pinned[c]=brl[c];
        CK(cudaMemcpy(d_brlen, h_brlen_pinned, sizeof(double)*nnodes, cudaMemcpyHostToDevice));
        build_echild<<<nnodes, NS*NS>>>(d_brlen,d_eval,d_U,d_catRates,NCAT,nnodes,T.root,d_echild);
        CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        vector<double> ecDev(echild.size());
        CK(cudaMemcpy(ecDev.data(), d_echild, echild.size()*sizeof(double), cudaMemcpyDeviceToHost));
        hostBuildEchild(brl, echild);
        double maxrel=0; size_t ndiff=0;
        for (size_t k=0;k<echild.size();k++){ double a=ecDev[k], b=echild[k];
            if (a!=b) ndiff++;
            double d = (fabs(b)>0)? fabs((a-b)/b) : fabs(a-b);
            if (d>maxrel) maxrel=d; }
        bool pass = (maxrel<1e-12);
        printf("V0 build_echild vs host (%s): bit-identical=%s ndiff=%zu/%zu maxrel=%.3e -> %s\n",
               tag, ndiff==0?"YES":"no", ndiff, echild.size(), maxrel, pass?"PASS":"FAIL");
        return pass;
    };
    { vector<double> bp=childLen; echildGate(childLen, "tree brlen");
      for (double& x:bp) x*=1.5; echildGate(bp, "perturbed x1.5"); }

    // ================= build + instantiate the CUDA graph (PATTERN A; Global capture mode) =================
    cudaStream_t capStream; CK(cudaStreamCreateWithFlags(&capStream, cudaStreamNonBlocking));
    for (int c=0;c<nnodes;c++) h_brlen_pinned[c]=childLen[c];   // initial pinned source = tree lengths
    cudaGraph_t graph=nullptr; cudaGraphExec_t graphExec=nullptr; bool useGraph=true;

    cudaError_t capErr = cudaStreamBeginCapture(capStream, cudaStreamCaptureModeGlobal);
    if (capErr!=cudaSuccess){ fprintf(stderr,"[G.1.3] BeginCapture failed: %s\n",cudaGetErrorString(capErr)); useGraph=false; cudaGetLastError(); }
    if (useGraph){
        // Node 1: brlen H2D from pinned buffer (PATTERN A — strictly ordered before build_echild)
        cudaMemcpyAsync(d_brlen, h_brlen_pinned, sizeof(double)*nnodes, cudaMemcpyHostToDevice, capStream);
        // Node 2: rebuild echild on device from d_brlen
        build_echild<<<nnodes, NS*NS, 0, capStream>>>(d_brlen,d_eval,d_U,d_catRates,NCAT,nnodes,T.root,d_echild);
        // Node 3: zero the reduction scratch (captured memset — NOT an external memset)
        cudaMemsetAsync(d_blocksum, 0, sizeof(double)*GB, capStream);
        // Nodes 4..101: postorder partial sweep
        launchSweep(capStream);
        // Nodes 102/103: deterministic two-level reduction -> d_lnL
        reduce_patlh<<<GB,TB,TB*sizeof(double),capStream>>>(d_patlh,nptn,d_blocksum);
        reduce_final<<<1,TB,TB*sizeof(double),capStream>>>(d_blocksum,GB,d_lnL);
        // Node 104: lnL D2H -> pinned
        cudaMemcpyAsync(h_lnL_pinned, d_lnL, sizeof(double), cudaMemcpyDeviceToHost, capStream);
        cudaError_t endErr = cudaStreamEndCapture(capStream, &graph);
        if (endErr!=cudaSuccess || graph==nullptr){
            fprintf(stderr,"[G.1.3] EndCapture failed: %s -> fallback\n",cudaGetErrorString(endErr));
            cudaGetLastError(); if(graph) cudaGraphDestroy(graph); graph=nullptr; useGraph=false;
        }
    }
    if (useGraph){
        cudaError_t instErr = cudaGraphInstantiate(&graphExec, graph, 0);   // 3-arg flags overload (CUDA 12.x)
        if (instErr!=cudaSuccess || graphExec==nullptr){
            fprintf(stderr,"[G.1.3] Instantiate failed: %s -> fallback\n",cudaGetErrorString(instErr));
            cudaGetLastError(); useGraph=false;
        }
    }
    printf("[graph] capture+instantiate: %s\n", useGraph?"OK (replay path active)":"FAILED (naive fallback active)");

    // ---- evaluators ----
    vector<double> patlh(nptn);
    auto hostKahan=[&](const vector<double>& v)->double{ double L=0,kc=0; for(double x:v){ double y=x-kc,t=L+y; kc=(t-L)-y; L=t; } return L; };
    // graph replay (PATTERN A): write pinned -> launch -> sync -> read scalar
    auto graphLnL=[&](const vector<double>& brl)->double{
        for (int c=0;c<nnodes;c++) h_brlen_pinned[c]=brl[c];
        CK(cudaGraphLaunch(graphExec, capStream));
        CK(cudaStreamSynchronize(capStream));
        double L=*h_lnL_pinned;
        if(!isfinite(L)) fprintf(stderr,"[WARN] non-finite lnL in graph replay (brlen out of bracket?)\n");
        return L;
    };
    // naive HOST-echild path (host builds echild, host Kahan sum) — proves the legacy path unchanged
    auto naiveHostLnL=[&](const vector<double>& brl)->double{
        hostBuildEchild(brl, echild);
        CK(cudaMemcpy(d_echild, echild.data(), echild.size()*sizeof(double), cudaMemcpyHostToDevice));
        launchSweep(0); CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        CK(cudaMemcpy(patlh.data(), d_patlh, nptn*sizeof(double), cudaMemcpyDeviceToHost));
        return hostKahan(patlh);
    };
    // naive DEVICE-echild path (same ops as the graph, but individual default-stream launches) — timing baseline
    auto naiveDeviceLnL=[&](const vector<double>& brl)->double{
        for (int c=0;c<nnodes;c++) h_brlen_pinned[c]=brl[c];
        CK(cudaMemcpy(d_brlen, h_brlen_pinned, sizeof(double)*nnodes, cudaMemcpyHostToDevice));
        build_echild<<<nnodes,NS*NS>>>(d_brlen,d_eval,d_U,d_catRates,NCAT,nnodes,T.root,d_echild);
        CK(cudaMemset(d_blocksum,0,sizeof(double)*GB));
        launchSweep(0);
        reduce_patlh<<<GB,TB,TB*sizeof(double)>>>(d_patlh,nptn,d_blocksum);
        reduce_final<<<1,TB,TB*sizeof(double)>>>(d_blocksum,GB,d_lnL);
        CK(cudaMemcpy(h_lnL_pinned, d_lnL, sizeof(double), cudaMemcpyDeviceToHost));
        CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        return *h_lnL_pinned;
    };

    map<string,double> oracle={{"g4",-7541976.9391},{"r8",-7556251.9185},{"r10",-7554280.5776},{"g1",-7974816.4323}};

    if (useGraph){
        // ===================== TIMING-ONLY subsample path (launch-bound curve point) =====================
        if (timingOnly){
            // V3 determinism still cheap + meaningful at any nptn
            double a=graphLnL(childLen), b=graphLnL(childLen);
            printf("V3 determinism (subsampled): lnL twice = %.6f / %.6f  bit-identical=%s\n", a,b, (a==b)?"YES":"no");
            graphLnL(childLen); naiveDeviceLnL(childLen);  // warmup both
            double bg=1e30; for(int r=0;r<reps;r++){ auto t0=Clock::now(); graphLnL(childLen); bg=min(bg,now_ms(t0,Clock::now())); }
            double bn=1e30; for(int r=0;r<reps;r++){ auto t0=Clock::now(); naiveDeviceLnL(childLen); bn=min(bn,now_ms(t0,Clock::now())); }
            printf("TIMING-CURVE nptn=%d : graph %.4f ms  naive %.4f ms  speedup %.2fx  (API 1 vs ~%d launches)\n",
                   nptn, bg, bn, bn/bg, nInternal+6);
            printf("================================================================================\n");
            cudaGraphExecDestroy(graphExec); cudaGraphDestroy(graph); cudaStreamDestroy(capStream);
            cudaFreeHost(h_brlen_pinned); cudaFreeHost(h_lnL_pinned);
            cudaFree(d_echild);cudaFree(d_partial);cudaFree(d_patlh);cudaFree(d_tip);
            cudaFree(d_U);cudaFree(d_eval);cudaFree(d_catRates);cudaFree(d_brlen);cudaFree(d_blocksum);cudaFree(d_lnL);
            return 0;
        }

        // ===================== V1: correctness vs oracle (graph & naive-host) =====================
        double lnL_graph = graphLnL(childLen);
        double lnL_nhost = naiveHostLnL(childLen);
        printf("\nV1 correctness @ tree brlen:\n");
        printf("   graph lnL      = %.4f\n", lnL_graph);
        printf("   naive-host lnL = %.4f  (host echild + host Kahan; legacy K1 path)\n", lnL_nhost);
        if (oracle.count(model)){ double o=oracle[model];
            double rg=fabs((lnL_graph-o)/o), rn=fabs((lnL_nhost-o)/o);
            printf("   oracle(G.0)    = %.4f   graph rel=%.3e (%s)  naive-host rel=%.3e (%s)\n",
                   o, rg, rg<1e-6?"PASS":"FAIL", rn, rn<1e-6?"PASS":"FAIL");
            printf("   graph vs naive-host: |dlnL|=%.3e rel=%.3e (reduction-order: device tree vs host Kahan)\n",
                   fabs(lnL_graph-lnL_nhost), fabs((lnL_graph-lnL_nhost)/o));
        }

        // ===================== V2: bit-identical patlh (pre-reduction), graph vs naive-device =====================
        graphLnL(childLen);                                  // graph populates d_patlh (reduce only reads it)
        vector<double> patlh_graph(nptn); CK(cudaMemcpy(patlh_graph.data(), d_patlh, nptn*sizeof(double), cudaMemcpyDeviceToHost));
        naiveDeviceLnL(childLen);                            // naive (device echild) populates d_patlh
        vector<double> patlh_naive(nptn); CK(cudaMemcpy(patlh_naive.data(), d_patlh, nptn*sizeof(double), cudaMemcpyDeviceToHost));
        size_t pdiff=0; double pmaxrel=0;
        for (int p=0;p<nptn;p++){ if(patlh_graph[p]!=patlh_naive[p]) pdiff++;
            double b=patlh_naive[p], d=(fabs(b)>0)?fabs((patlh_graph[p]-b)/b):fabs(patlh_graph[p]-b); if(d>pmaxrel)pmaxrel=d; }
        printf("V2 patlh bit-identity (graph vs naive-device, both build_echild): ndiff=%zu/%d maxrel=%.3e -> %s\n",
               pdiff, nptn, pmaxrel, pdiff==0?"PASS (compute path bit-identical)":"CHECK");

        // ===================== V3: determinism (two replays, same brlen) =====================
        double r1=graphLnL(childLen), r2=graphLnL(childLen);
        printf("V3 determinism: lnL twice = %.6f / %.6f  bit-identical=%s -> %s\n", r1,r2, (r1==r2)?"YES":"no", (r1==r2)?"PASS":"FAIL");

        // ===================== V4: perturbation (one edge), graph vs naive-host =====================
        { vector<double> bp=childLen; bp[c0]=tc0*1.5;
          double lg=graphLnL(bp), ln=naiveHostLnL(bp);
          printf("V4 perturb edge c0=%d (t %.4f->%.4f): graph %.4f  naive-host %.4f  |dlnL|=%.3e -> %s\n",
                 c0,tc0,bp[c0], lg, ln, fabs(lg-ln), (fabs(lg-ln)<1e-4)?"PASS":"FAIL"); }

        // ===================== V5: single-branch optimisation via graph replay vs naive =====================
        { vector<double> bg=childLen, bn=childLen;
          double tg=golden_max([&](double t){ bg[c0]=t; return graphLnL(bg); }, 1e-6, 10.0, 1e-6, 50);
          double tn=golden_max([&](double t){ bn[c0]=t; return naiveHostLnL(bn); }, 1e-6, 10.0, 1e-6, 50);
          bg[c0]=tg; bn[c0]=tn;
          printf("V5 single-branch opt (edge c0): graph t*=%.6f  naive t*=%.6f  |dt|=%.3e  (tree t0=%.6f) -> %s\n",
                 tg,tn,fabs(tg-tn),tc0, (fabs(tg-tn)<1e-4)?"PASS":"FAIL"); }

        // ===================== V6: multi-branch convergence (optimizeAllBranches-shaped; g4 only) =====================
        if (doMulti){
            printf("\nV6 multi-branch sweep (perturb ALL x1.3, one Gauss-Seidel pass golden-section per branch):\n");
            // every branch = every non-root node's parent edge (indexed by the child node id); incl. tip branches
            vector<int> edges; for(int c=0;c<nnodes;c++) if(c!=T.root) edges.push_back(c);
            vector<double> vg(nnodes), vn(nnodes);
            for (int c=0;c<nnodes;c++){ vg[c]=childLen[c]*1.3; vn[c]=childLen[c]*1.3; }
            auto sweep=[&](vector<double>& v, const function<double(const vector<double>&)>& ev){
                for (int e : edges){ double opt=golden_max([&](double t){ v[e]=t; return ev(v); }, 1e-6, 10.0, 1e-3, 24); v[e]=opt; }
                return ev(v);
            };
            double Lg = sweep(vg, [&](const vector<double>& v){ return graphLnL(v); });
            double Ln = sweep(vn, [&](const vector<double>& v){ return naiveHostLnL(v); });
            double maxdt=0; for(int e:edges){ double d=fabs(vg[e]-vn[e]); if(d>maxdt)maxdt=d; }
            double lnL_check = naiveHostLnL(vg);   // re-evaluate the graph-converged vector on the naive path
            printf("   edges swept=%zu  graph-final lnL=%.4f  naive-final lnL=%.4f  |dlnL|=%.3e\n",
                   edges.size(), Lg, Ln, fabs(Lg-Ln));
            printf("   converged-vector max |dt| (graph vs naive) = %.3e ; naive re-eval of graph vector = %.4f (|d|=%.3e)\n",
                   maxdt, lnL_check, fabs(lnL_check-Lg));
            bool pass = (fabs(Lg-Ln)<1e-4) && (maxdt<1e-2) && isfinite(Lg) && isfinite(Ln);
            printf("   -> %s (graph-driven sweep converges to the same branch lengths + lnL as the naive sweep)\n", pass?"PASS":"CHECK");
        }

        // ===================== TIMING @ full nptn (compute-bound; report as a curve via the run script) =====================
        graphLnL(childLen); naiveDeviceLnL(childLen);        // warmup (excludes one-time graph upload/JIT)
        double bg=1e30; for(int r=0;r<reps;r++){ auto t0=Clock::now(); graphLnL(childLen); bg=min(bg,now_ms(t0,Clock::now())); }
        double bn=1e30; for(int r=0;r<reps;r++){ auto t0=Clock::now(); naiveDeviceLnL(childLen); bn=min(bn,now_ms(t0,Clock::now())); }
        printf("\nTIMING-CURVE nptn=%d : graph %.4f ms  naive %.4f ms  speedup %.2fx  (host API per eval: 1 graphLaunch vs ~%d launches)\n",
               nptn, bg, bn, bn/bg, nInternal+6);

        cudaGraphExecDestroy(graphExec); cudaGraphDestroy(graph);
    } else {
        // ===================== FALLBACK (no graph): naive-host path only, still validate vs oracle =====================
        double lnL_nhost = naiveHostLnL(childLen);
        printf("\n[FALLBACK] graph unavailable; naive-host lnL @ tree brlen = %.4f\n", lnL_nhost);
        if (oracle.count(model)){ double o=oracle[model], rn=fabs((lnL_nhost-o)/o);
            printf("   oracle(G.0)=%.4f rel=%.3e -> %s\n", o, rn, rn<1e-6?"PASS":"FAIL"); }
    }

    size_t gpufree=0,gputot=0; cudaMemGetInfo(&gpufree,&gputot);
    printf("VRAM used ~ %.2f GB / %.1f GB\n", (gputot-gpufree)/1073741824.0, gputot/1073741824.0);
    printf("================================================================================\n");

    cudaStreamDestroy(capStream);
    cudaFreeHost(h_brlen_pinned); cudaFreeHost(h_lnL_pinned);
    cudaFree(d_echild);cudaFree(d_partial);cudaFree(d_patlh);cudaFree(d_tip);
    cudaFree(d_U);cudaFree(d_eval);cudaFree(d_catRates);cudaFree(d_brlen);cudaFree(d_blocksum);cudaFree(d_lnL);
    return 0;
}
