// gpu_k4_fused.cu — G.1.3 perf pass: same-depth kernel FUSION of the postorder lnL sweep.
//
// G.1.3 (gpu_k3_graph.cu) showed the 98 per-node k1_node launches give wall-clock PARITY under CUDA-graph
// replay: the chain is bound by GPU-side per-kernel scheduling latency (~85us/launch), which a graph does
// NOT remove (it only collapses host submission, already overlapped). The fix the graph cannot deliver:
// FUSE the launches. Internal nodes grouped by tree HEIGHT (longest path to a leaf) are mutually independent
// — every node at height h has all children at height < h (computed in earlier launches) — so an entire
// height level can run in ONE kernel launch. This collapses ~98 launches into ~tree-height launches and lets
// same-level nodes run concurrently (more blocks in flight). Numerics are UNCHANGED (identical k1_node body,
// only the dispatch is batched), so the result must be BIT-IDENTICAL to the K1/K3 per-node sweep and the G.0
// oracle. Best config = fused levels + graph replay (the architecture G.2 will integrate).
//
// k4_level: one launch per height level. 2D grid <<<dim3(GB, nodesInLevel), TB>>> — blockIdx.y selects the
// node (reads its LevelDesc: out slot + per-child ec/p/t pointers), blockIdx.x*blockDim.x+threadIdx.x is the
// pattern (coalesced, pattern-innermost layout preserved). The root (writes patlh) stays a single k1_node
// launch after the last non-root level.
//
// Build (gpuvolta job): module load cuda/12.5.1 gcc/12.2.0
//   nvcc -O3 -std=c++17 -arch=sm_70 -lineinfo gpu_k4_fused.cu -o gpu_k4_fused          (NO fast-math, FP64)
// Run:  ./gpu_k4_fused <aln.phy> <tree> [model=g4|r8|r10|g1] [reps] [ptncap=0(all)]
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

// ===================== CPU scaffolding (verbatim from gpu_k1_lnl.cu / gpu_k3_graph.cu, BEAGLE-free) =====================
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

// ===================== device constant model data (set once, never inside capture) =====================
__constant__ double c_Uinv[NS*NS];
__constant__ double c_UinvRowSum[NS];
__constant__ double c_freq[NS];
__constant__ double c_catw[MAXCAT];

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

// per-node device descriptor (for the fused k4_level kernel; pointers fixed across replays)
struct LevelDesc { double* out; int nchild; const double* ec[3]; const double* p[3]; const unsigned char* t[3]; };

// FUSED kernel: one launch processes ALL nodes in one height level. blockIdx.y = node within level.
__global__ void k4_level(const LevelDesc* __restrict__ D, int nptn, int ncat) {
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    LevelDesc d = D[blockIdx.y];                         // broadcast: all threads in block read the same node
    for (int c=0;c<ncat;c++){
        double prod[NS];
        #pragma unroll
        for (int x=0;x<NS;x++) prod[x]=1.0;
        accum_child(prod,c,ptn,nptn,d.ec[0],d.p[0],d.t[0]);
        if (d.nchild>1) accum_child(prod,c,ptn,nptn,d.ec[1],d.p[1],d.t[1]);
        if (d.nchild>2) accum_child(prod,c,ptn,nptn,d.ec[2],d.p[2],d.t[2]);
        double* o = d.out + (size_t)(c*NS)*nptn + ptn;
        #pragma unroll
        for (int r=0;r<NS;r++){ double v=0.0;
            #pragma unroll
            for (int x=0;x<NS;x++) v += c_Uinv[r*NS+x]*prod[x];
            o[(size_t)r*nptn]=v; }
    }
}

// per-node kernel (the K1/K3 baseline + the root node, which writes patlh) — VERBATIM from K3.
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

// on-device echild rebuild (VERBATIM from K3: host-grouping-exact — len=brlen*rate FIRST, then exp(eval*len))
__global__ void build_echild(const double* __restrict__ d_brlen, const double* __restrict__ d_eval,
        const double* __restrict__ d_U, const double* __restrict__ d_catRates,
        int ncat, int nnodes, int rootId, double* __restrict__ d_echild) {
    int c = blockIdx.x;
    if (c >= nnodes || c == rootId) return;
    int tid = threadIdx.x; if (tid >= NS*NS) return;
    int x = tid / NS, i = tid % NS;
    double bl  = d_brlen[c];
    double Uxi = d_U[x*NS + i];
    double ev  = d_eval[i];
    size_t base = (size_t)c * (size_t)ncat * NS * NS;
    for (int cat=0; cat<ncat; cat++){
        double len = bl * d_catRates[cat];
        double ex  = exp(ev * len);
        d_echild[base + (size_t)cat*NS*NS + x*NS + i] = Uxi * ex;
    }
}

// deterministic block-local pairwise reductions (VERBATIM from K3)
__global__ void reduce_patlh(const double* __restrict__ patlh, int nptn, double* __restrict__ blocksum) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i   = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = (i < nptn) ? patlh[i] : 0.0;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1){ if (tid < s) sdata[tid] += sdata[tid + s]; __syncthreads(); }
    if (tid == 0) blocksum[blockIdx.x] = sdata[0];
}
__global__ void reduce_final(const double* __restrict__ blocksum, int n, double* __restrict__ out) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    double v = 0.0;
    for (int i = tid; i < n; i += blockDim.x) v += blocksum[i];
    sdata[tid] = v;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1){ if (tid < s) sdata[tid] += sdata[tid + s]; __syncthreads(); }
    if (tid == 0) out[0] = sdata[0];
}

int main(int argc, char** argv){
    if (argc<3){ fprintf(stderr,"usage: %s <aln.phy> <tree> [model=g4|r8|r10|g1] [reps] [ptncap=0]\n",argv[0]); return 1; }
    string alnpath=argv[1], treepath=argv[2];
    string model=(argc>3)?argv[3]:"g4";
    int reps   = (argc>4)?atoi(argv[4]):50;
    int ptncap = (argc>5)?atoi(argv[5]):0;

    // ---- alignment ----
    ifstream af(alnpath); if(!af){ fprintf(stderr,"no aln %s\n",alnpath.c_str()); return 2; }
    int ntax=0,nsite=0; { string h; getline(af,h); sscanf(h.c_str(),"%d %d",&ntax,&nsite); }
    vector<string> seqs(ntax); map<string,int> name2tip;
    for (int t=0;t<ntax;t++){ string line; if(!getline(af,line)){fprintf(stderr,"aln short\n");return 2;}
        istringstream ls(line); string nm,sq; ls>>nm>>sq; seqs[t]=sq; name2tip[nm]=t; }
    int nptn = nsite; bool timingOnly=false;
    if (ptncap>0 && ptncap<nsite){ nptn=ptncap; timingOnly=true; }
    printf("[aln] ntax=%d nsite=%d nptn=%d%s\n", ntax, nsite, nptn, timingOnly?"  (TIMING-ONLY subsample)":"");

    // ---- tree ----
    string ts; { ifstream tf(treepath); stringstream ss; ss<<tf.rdbuf(); ts=ss.str(); }
    string tc; for(char c:ts) if(!isspace((unsigned char)c)) tc+=c;
    Tree T=parse_newick(tc,name2tip);
    int nnodes=T.nodes.size();
    printf("[tree] nodes=%d root_children=%zu\n", nnodes, T.nodes[T.root].child.size());

    // ---- model ----
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

    // ---- eigendecomposition (verbatim) ----
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

    // ---- postorder internal nodes + slots; child edge lengths ----
    vector<double> childLen(nnodes,0.0);
    for (int u=0;u<nnodes;u++) for(size_t k=0;k<T.nodes[u].child.size();k++) childLen[T.nodes[u].child[k]]=T.nodes[u].blen[k];
    vector<int> postorder; vector<int> slot(nnodes,-1);
    function<void(int)> dfs=[&](int u){ for(int c:T.nodes[u].child) dfs(c); if(T.nodes[u].leaf<0){ slot[u]=postorder.size(); postorder.push_back(u);} };
    dfs(T.root);
    int nInternal=postorder.size();

    // ---- tree HEIGHT per node (longest path to a leaf); leaves=0, internal=1+max(child) ----
    vector<int> height(nnodes,0);
    function<int(int)> computeH=[&](int u)->int{ if(T.nodes[u].leaf>=0){ height[u]=0; return 0; }
        int h=0; for(int c:T.nodes[u].child) h=max(h, computeH(c)); height[u]=h+1; return h+1; };
    computeH(T.root);
    int maxh=height[T.root];
    // group NON-ROOT internal nodes by height (root is the unique height-maxh node -> launched separately)
    vector<vector<int>> levelNodes(maxh+1);
    for (int idx=0; idx<nInternal; idx++){ int u=postorder[idx]; if(u==T.root) continue; levelNodes[height[u]].push_back(u); }
    int nLaunchFused=1;  // the root launch
    { string hist; for(int h=1;h<maxh;h++){ if(levelNodes[h].empty()) continue; nLaunchFused++;
        hist += " L"+to_string(h)+"="+to_string(levelNodes[h].size()); }
      printf("[fuse] tree height=%d ; non-root internal=%d in %d levels;%s -> fused launches=%d (+reduce) vs per-node=%d\n",
             maxh, nInternal-1, nLaunchFused-1, hist.c_str(), nLaunchFused, nInternal); }

    // ---- host echild builder + tip states ----
    size_t ecStride = (size_t)NCAT*NS*NS;
    vector<double> echild((size_t)nnodes*ecStride, 0.0);
    auto hostBuildEchild=[&](const vector<double>& brl, vector<double>& ec){
        for (int c=0;c<nnodes;c++){ if(c==T.root) continue;
            for (int cat=0;cat<NCAT;cat++){ double len=brl[c]*catRates[cat];
                double ex[NS]; for(int i=0;i<NS;i++) ex[i]=exp(evals[i]*len);
                double* e=&ec[(size_t)c*ecStride + (size_t)cat*NS*NS];
                for (int x=0;x<NS;x++) for(int i=0;i<NS;i++) e[x*NS+i]=U[x*NS+i]*ex[i]; } }
    };
    vector<unsigned char> tip((size_t)ntax*nptn);
    for (int u=0;u<nnodes;u++){ if(T.nodes[u].leaf<0) continue; int lf=T.nodes[u].leaf; const string& s=seqs[lf];
        for (int p=0;p<nptn;p++){ int a=aa_index(s[p]); tip[(size_t)lf*nptn+p]=(unsigned char)((a<0)?NS:a); } }

    // ---- device upload ----
    CK(cudaMemcpyToSymbol(c_Uinv, Uinv.data(), sizeof(double)*NS*NS));
    CK(cudaMemcpyToSymbol(c_UinvRowSum, UinvRowSum.data(), sizeof(double)*NS));
    CK(cudaMemcpyToSymbol(c_freq, f, sizeof(double)*NS));
    CK(cudaMemcpyToSymbol(c_catw, catWeights.data(), sizeof(double)*NCAT));
    double *d_echild,*d_partial,*d_patlh,*d_U,*d_eval,*d_catRates,*d_brlen,*d_blocksum,*d_lnL; unsigned char *d_tip;
    CK(cudaMalloc(&d_echild, echild.size()*sizeof(double)));
    CK(cudaMalloc(&d_tip, tip.size())); CK(cudaMemcpy(d_tip, tip.data(), tip.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_U, sizeof(double)*NS*NS));       CK(cudaMemcpy(d_U, U.data(), sizeof(double)*NS*NS, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_eval, sizeof(double)*NS));       CK(cudaMemcpy(d_eval, evals.data(), sizeof(double)*NS, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_catRates, sizeof(double)*NCAT)); CK(cudaMemcpy(d_catRates, catRates.data(), sizeof(double)*NCAT, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_brlen, sizeof(double)*nnodes));
    size_t slotSz=(size_t)NCAT*NS*nptn;
    CK(cudaMalloc(&d_partial, (size_t)nInternal*slotSz*sizeof(double)));
    CK(cudaMalloc(&d_patlh, (size_t)nptn*sizeof(double)));
    int TB=256, GB=(nptn+TB-1)/TB;
    CK(cudaMalloc(&d_blocksum, (size_t)GB*sizeof(double)));
    CK(cudaMalloc(&d_lnL, sizeof(double)));
    double *h_brlen_pinned, *h_lnL_pinned;
    CK(cudaHostAlloc((void**)&h_brlen_pinned, sizeof(double)*nnodes, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**)&h_lnL_pinned, sizeof(double), cudaHostAllocDefault));
    printf("[mem] partials arena = %.2f GB\n", (double)nInternal*slotSz*8/1073741824.0);

    // ---- helper: child pointers for a node ----
    auto fillChildPtrs=[&](int u, int& nch, const double** ec, const double** p, const unsigned char** t){
        nch=T.nodes[u].child.size(); for(int k=0;k<3;k++){ec[k]=nullptr;p[k]=nullptr;t[k]=nullptr;}
        for (int k=0;k<nch;k++){ int c=T.nodes[u].child[k];
            ec[k]=d_echild + (size_t)c*ecStride;
            if (T.nodes[c].leaf>=0) t[k]=d_tip + (size_t)T.nodes[c].leaf*nptn;
            else                    p[k]=d_partial + (size_t)slot[c]*slotSz; }
    };

    // ---- per-node descriptors (for the per-node baseline sweep) ----
    struct Desc { int isRoot,nchild; const double* ec[3]; const double* p[3]; const unsigned char* t[3]; double* out; };
    vector<Desc> desc(nInternal);
    for (int idx=0; idx<nInternal; idx++){ int u=postorder[idx]; Desc& D=desc[idx];
        D.isRoot=(u==T.root)?1:0;
        int nch; const double* ec[3]; const double* p[3]; const unsigned char* t[3]; fillChildPtrs(u,nch,ec,p,t);
        D.nchild=nch; if(nch>3){fprintf(stderr,"deg>3\n");return 6;}
        D.out=D.isRoot?nullptr:(d_partial+(size_t)slot[u]*slotSz);
        for(int k=0;k<3;k++){ D.ec[k]=ec[k]; D.p[k]=p[k]; D.t[k]=t[k]; }
    }
    // ---- fused level descriptors (device array, ordered by level) + per-level offsets/counts ----
    vector<LevelDesc> hLevelDesc; vector<int> levelOff(maxh+1,0), levelCnt(maxh+1,0);
    for (int h=1; h<maxh; h++){ levelOff[h]=hLevelDesc.size();
        for (int u : levelNodes[h]){ LevelDesc d; int nch; const double* ec[3]; const double* p[3]; const unsigned char* t[3];
            fillChildPtrs(u,nch,ec,p,t); d.out=d_partial+(size_t)slot[u]*slotSz; d.nchild=nch;
            for(int k=0;k<3;k++){ d.ec[k]=ec[k]; d.p[k]=p[k]; d.t[k]=t[k]; } hLevelDesc.push_back(d); }
        levelCnt[h]=hLevelDesc.size()-levelOff[h]; }
    LevelDesc* d_levelDesc=nullptr;
    if (!hLevelDesc.empty()){ CK(cudaMalloc(&d_levelDesc, hLevelDesc.size()*sizeof(LevelDesc)));
        CK(cudaMemcpy(d_levelDesc, hLevelDesc.data(), hLevelDesc.size()*sizeof(LevelDesc), cudaMemcpyHostToDevice)); }
    // root child pointers (the final per-node launch that writes patlh)
    int rNch; const double* rEc[3]; const double* rP[3]; const unsigned char* rT[3]; fillChildPtrs(T.root,rNch,rEc,rP,rT);

    // ---- sweeps ----
    auto sweepPerNode=[&](cudaStream_t st){
        for (int idx=0; idx<nInternal; idx++){ Desc& D=desc[idx];
            k1_node<<<GB,TB,0,st>>>(nptn,NCAT,D.isRoot,D.out,d_patlh,D.nchild,
                D.ec[0],D.p[0],D.t[0], D.ec[1],D.p[1],D.t[1], D.ec[2],D.p[2],D.t[2]); }
    };
    auto sweepFused=[&](cudaStream_t st){
        for (int h=1; h<maxh; h++){ int cnt=levelCnt[h]; if(cnt==0) continue;
            k4_level<<<dim3(GB,cnt),TB,0,st>>>(d_levelDesc+levelOff[h], nptn, NCAT); }
        k1_node<<<GB,TB,0,st>>>(nptn,NCAT,1,nullptr,d_patlh,rNch, rEc[0],rP[0],rT[0], rEc[1],rP[1],rT[1], rEc[2],rP[2],rT[2]);
    };

    printf("\n========== G.1.3 perf pass — same-depth kernel fusion (K4) ==========\n");
    map<string,double> oracle={{"g4",-7541976.9391},{"r8",-7556251.9185},{"r10",-7554280.5776},{"g1",-7974816.4323}};
    vector<double> patlh(nptn);
    auto hostKahan=[&](const vector<double>& v)->double{ double L=0,kc=0; for(double x:v){ double y=x-kc,t=L+y; kc=(t-L)-y; L=t; } return L; };
    auto sweepLnL=[&](const function<void(cudaStream_t)>& sweep)->double{
        hostBuildEchild(childLen, echild);
        CK(cudaMemcpy(d_echild, echild.data(), echild.size()*sizeof(double), cudaMemcpyHostToDevice));
        sweep(0); CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        CK(cudaMemcpy(patlh.data(), d_patlh, nptn*sizeof(double), cudaMemcpyDeviceToHost));
        return hostKahan(patlh);
    };

    // ---- V1: fused == per-node == oracle, and patlh bit-identical ----
    double lnL_pn = sweepLnL([&](cudaStream_t s){ sweepPerNode(s); });
    vector<double> patlh_pn = patlh;
    double lnL_fz = sweepLnL([&](cudaStream_t s){ sweepFused(s); });
    vector<double> patlh_fz = patlh;
    size_t pdiff=0; for(int p=0;p<nptn;p++) if(patlh_pn[p]!=patlh_fz[p]) pdiff++;
    printf("V1 per-node lnL=%.4f  fused lnL=%.4f  |dlnL|=%.3e ; patlh bit-identical: ndiff=%zu/%d\n",
           lnL_pn, lnL_fz, fabs(lnL_pn-lnL_fz), pdiff, nptn);
    if (!timingOnly && oracle.count(model)){ double o=oracle[model];
        printf("   oracle(G.0)=%.4f  fused rel=%.3e -> %s ; bit-identical-to-per-node: %s\n",
               o, fabs((lnL_fz-o)/o), (fabs((lnL_fz-o)/o)<1e-6)?"PASS":"FAIL", (pdiff==0)?"PASS":"CHECK"); }

    // ---- build the FUSED graph (Pattern A; Global mode) ----
    cudaStream_t capStream; CK(cudaStreamCreateWithFlags(&capStream, cudaStreamNonBlocking));
    for (int c=0;c<nnodes;c++) h_brlen_pinned[c]=childLen[c];
    cudaGraph_t graph=nullptr; cudaGraphExec_t graphExec=nullptr; bool useGraph=true;
    cudaError_t capErr = cudaStreamBeginCapture(capStream, cudaStreamCaptureModeGlobal);
    if (capErr!=cudaSuccess){ fprintf(stderr,"[K4] BeginCapture failed: %s\n",cudaGetErrorString(capErr)); useGraph=false; cudaGetLastError(); }
    if (useGraph){
        cudaMemcpyAsync(d_brlen, h_brlen_pinned, sizeof(double)*nnodes, cudaMemcpyHostToDevice, capStream);
        build_echild<<<nnodes, NS*NS, 0, capStream>>>(d_brlen,d_eval,d_U,d_catRates,NCAT,nnodes,T.root,d_echild);
        cudaMemsetAsync(d_blocksum, 0, sizeof(double)*GB, capStream);
        sweepFused(capStream);
        reduce_patlh<<<GB,TB,TB*sizeof(double),capStream>>>(d_patlh,nptn,d_blocksum);
        reduce_final<<<1,TB,TB*sizeof(double),capStream>>>(d_blocksum,GB,d_lnL);
        cudaMemcpyAsync(h_lnL_pinned, d_lnL, sizeof(double), cudaMemcpyDeviceToHost, capStream);
        cudaError_t endErr = cudaStreamEndCapture(capStream, &graph);
        if (endErr!=cudaSuccess || graph==nullptr){ fprintf(stderr,"[K4] EndCapture failed: %s\n",cudaGetErrorString(endErr));
            cudaGetLastError(); if(graph) cudaGraphDestroy(graph); graph=nullptr; useGraph=false; }
    }
    if (useGraph){ cudaError_t ie=cudaGraphInstantiate(&graphExec, graph, 0);
        if (ie!=cudaSuccess || graphExec==nullptr){ fprintf(stderr,"[K4] Instantiate failed: %s\n",cudaGetErrorString(ie)); cudaGetLastError(); useGraph=false; } }
    printf("[graph] fused capture+instantiate: %s\n", useGraph?"OK":"FAILED (naive fallback)");

    auto graphLnL=[&]()->double{ CK(cudaGraphLaunch(graphExec, capStream)); CK(cudaStreamSynchronize(capStream)); return *h_lnL_pinned; };
    if (useGraph && !timingOnly){
        double lg=graphLnL();
        printf("V2 fused-graph lnL=%.4f  vs oracle rel=%.3e -> %s ; vs per-node |dlnL|=%.3e\n",
               lg, oracle.count(model)?fabs((lg-oracle[model])/oracle[model]):0.0,
               (oracle.count(model)&&fabs((lg-oracle[model])/oracle[model])<1e-6)?"PASS":"n/a", fabs(lg-lnL_pn));
        double r1=graphLnL(), r2=graphLnL();
        printf("V3 fused-graph determinism: %.6f / %.6f -> %s\n", r1,r2, (r1==r2)?"PASS":"FAIL");
    }

    // ---- TIMING: per-node-naive vs fused-naive vs fused-graph ----
    // warmups
    sweepPerNode(0); CK(cudaDeviceSynchronize());
    sweepFused(0);   CK(cudaDeviceSynchronize());
    if (useGraph){ graphLnL(); }
    double bpn=1e30,bfz=1e30,bgz=1e30;
    for(int r=0;r<reps;r++){ auto t0=Clock::now(); sweepPerNode(0); CK(cudaDeviceSynchronize()); bpn=min(bpn,now_ms(t0,Clock::now())); }
    for(int r=0;r<reps;r++){ auto t0=Clock::now(); sweepFused(0);   CK(cudaDeviceSynchronize()); bfz=min(bfz,now_ms(t0,Clock::now())); }
    if (useGraph) for(int r=0;r<reps;r++){ auto t0=Clock::now(); graphLnL(); bgz=min(bgz,now_ms(t0,Clock::now())); }
    printf("TIMING nptn=%d : per-node %.4f ms (%d launches)  |  fused %.4f ms (%d launches, %.2fx)  |  fused-graph %.4f ms (1 launch, %.2fx)\n",
           nptn, bpn, nInternal, bfz, nLaunchFused, bpn/bfz, useGraph?bgz:0.0, useGraph?bpn/bgz:0.0);

    size_t gpufree=0,gputot=0; cudaMemGetInfo(&gpufree,&gputot);
    printf("VRAM used ~ %.2f GB / %.1f GB\n", (gputot-gpufree)/1073741824.0, gputot/1073741824.0);
    printf("=====================================================================\n");

    if (useGraph){ cudaGraphExecDestroy(graphExec); cudaGraphDestroy(graph); }
    cudaStreamDestroy(capStream);
    if (d_levelDesc) cudaFree(d_levelDesc);
    cudaFreeHost(h_brlen_pinned); cudaFreeHost(h_lnL_pinned);
    cudaFree(d_echild);cudaFree(d_partial);cudaFree(d_patlh);cudaFree(d_tip);
    cudaFree(d_U);cudaFree(d_eval);cudaFree(d_catRates);cudaFree(d_brlen);cudaFree(d_blocksum);cudaFree(d_lnL);
    return 0;
}
