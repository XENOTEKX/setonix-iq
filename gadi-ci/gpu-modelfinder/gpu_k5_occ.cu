// gpu_k5_occ.cu — G.1.3 intra-kernel perf lever: OCCUPANCY sweep of the K1 postorder lnL kernel.
//
// Nsight (job 170195112) found k1_node is LATENCY-bound at 25% occupancy, capped by REGISTER PRESSURE:
// 128 regs/thread -> Block Limit(Registers)=2 -> 25% theoretical occupancy; Compute ~35%, Memory ~48%, DRAM
// ~16-40% (none saturated) -> too few warps to hide latency. NOT bandwidth- or compute-bound; NOT helped by
// shared-mem echild staging (echild reads are broadcast, L1 hit ~70-82%). The lever is register reduction to
// raise occupancy. This harness A/B's __launch_bounds__(maxThreads,minBlocks) caps (compiler spills the
// excess to L1-cached local): minBlocks {2,3,4,5,6} at 256 threads => reg caps {128,85,64,51,42} => occupancy
// targets {25,37.5,50,62.5,75}% — find the sweet spot where occupancy gain beats spill cost. The kernel BODY
// is byte-identical to k1_node, so every config MUST give the bit-identical lnL = the G.0 oracle.
//
// Build (gpuvolta job): module load cuda/12.5.1 gcc/12.2.0
//   nvcc -O3 -std=c++17 -arch=sm_70 -lineinfo gpu_k5_occ.cu -o gpu_k5_occ          (NO fast-math, FP64)
// Run:  ./gpu_k5_occ <aln.phy> <tree> [model=g4|r8|r10|g1|all] [reps]
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

// ===================== CPU scaffolding (verbatim from gpu_k1_lnl.cu) =====================
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

// ===================== device constants + kernel body =====================
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

// shared body (byte-identical to k1_node) — register count set by the kernel's __launch_bounds__.
__device__ __forceinline__ void k1_body(int nptn, int ncat, int isRoot, double* __restrict__ out,
        double* __restrict__ patlh, int nchild,
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
// baseline (compiler picks regs -> 128 -> 25% occ)
__global__ void k1_base(int nptn,int ncat,int isRoot,double* out,double* patlh,int nchild,
    const double* ec0,const double* p0,const unsigned char* t0, const double* ec1,const double* p1,const unsigned char* t1,
    const double* ec2,const double* p2,const unsigned char* t2){
    k1_body(nptn,ncat,isRoot,out,patlh,nchild,ec0,p0,t0,ec1,p1,t1,ec2,p2,t2);
}
// occupancy-capped variants: __launch_bounds__(MAXT,MINB) -> reg cap = 65536/(MAXT*MINB)
template<int MAXT,int MINB> __global__ void __launch_bounds__(MAXT,MINB)
k1_lb(int nptn,int ncat,int isRoot,double* out,double* patlh,int nchild,
    const double* ec0,const double* p0,const unsigned char* t0, const double* ec1,const double* p1,const unsigned char* t1,
    const double* ec2,const double* p2,const unsigned char* t2){
    k1_body(nptn,ncat,isRoot,out,patlh,nchild,ec0,p0,t0,ec1,p1,t1,ec2,p2,t2);
}

struct Desc { int isRoot,nchild; const double* ec[3]; const double* p[3]; const unsigned char* t[3]; double* out; };

// timing context shared by all configs (built once; only the kernel/block differ per config)
struct Ctx { const vector<Desc>* desc; int nInternal,nptn,NCAT; double* d_patlh; vector<double>* patlh; double oracle; int reps; };

static void report(const char* tag, double L, const Ctx& c, double best){
    printf("  %-12s : lnL=%.4f  rel=%.2e  sweep %.4f ms\n", tag, L, fabs((L-c.oracle)/c.oracle), best);
}
static double kahan(const vector<double>& v){ double L=0,kc=0; for(double x:v){double y=x-kc,t=L+y;kc=(t-L)-y;L=t;} return L; }

static void timeBase(const Ctx& c){
    int BS=256, GB=(c.nptn+BS-1)/BS;
    auto sweep=[&](){ for(int i=0;i<c.nInternal;i++){ const Desc&D=(*c.desc)[i];
        k1_base<<<GB,BS>>>(c.nptn,c.NCAT,D.isRoot,D.out,c.d_patlh,D.nchild,
            D.ec[0],D.p[0],D.t[0],D.ec[1],D.p[1],D.t[1],D.ec[2],D.p[2],D.t[2]); } };
    sweep(); CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    CK(cudaMemcpy(c.patlh->data(),c.d_patlh,c.nptn*sizeof(double),cudaMemcpyDeviceToHost));
    double L=kahan(*c.patlh);
    double best=1e30; for(int r=0;r<c.reps;r++){auto t0=Clock::now(); sweep(); CK(cudaDeviceSynchronize()); best=min(best,now_ms(t0,Clock::now()));}
    report("base(256)", L, c, best);
}
template<int MAXT,int MINB>
static void timeLB(const char* tag, const Ctx& c){
    int BS=MAXT, GB=(c.nptn+BS-1)/BS;
    auto sweep=[&](){ for(int i=0;i<c.nInternal;i++){ const Desc&D=(*c.desc)[i];
        k1_lb<MAXT,MINB><<<GB,BS>>>(c.nptn,c.NCAT,D.isRoot,D.out,c.d_patlh,D.nchild,
            D.ec[0],D.p[0],D.t[0],D.ec[1],D.p[1],D.t[1],D.ec[2],D.p[2],D.t[2]); } };
    sweep(); CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    CK(cudaMemcpy(c.patlh->data(),c.d_patlh,c.nptn*sizeof(double),cudaMemcpyDeviceToHost));
    double L=kahan(*c.patlh);
    double best=1e30; for(int r=0;r<c.reps;r++){auto t0=Clock::now(); sweep(); CK(cudaDeviceSynchronize()); best=min(best,now_ms(t0,Clock::now()));}
    report(tag, L, c, best);
}

int main(int argc, char** argv){
    if (argc<3){ fprintf(stderr,"usage: %s <aln.phy> <tree> [model=g4|r8|r10|g1|all] [reps]\n",argv[0]); return 1; }
    string alnpath=argv[1], treepath=argv[2];
    string modarg=(argc>3)?argv[3]:"all";
    int reps=(argc>4)?atoi(argv[4]):30;

    ifstream af(alnpath); if(!af){ fprintf(stderr,"no aln\n"); return 2; }
    int ntax=0,nsite=0; { string h; getline(af,h); sscanf(h.c_str(),"%d %d",&ntax,&nsite); }
    vector<string> seqs(ntax); map<string,int> name2tip;
    for (int t=0;t<ntax;t++){ string line; getline(af,line); istringstream ls(line); string nm,sq; ls>>nm>>sq; seqs[t]=sq; name2tip[nm]=t; }
    int nptn=nsite; printf("[aln] ntax=%d nsite=%d\n", ntax, nptn);

    string ts; { ifstream tf(treepath); stringstream ss; ss<<tf.rdbuf(); ts=ss.str(); }
    string tc; for(char ch:ts) if(!isspace((unsigned char)ch)) tc+=ch;
    Tree T=parse_newick(tc,name2tip); int nnodes=T.nodes.size();

    vector<double> childLen(nnodes,0.0);
    for (int u=0;u<nnodes;u++) for(size_t k=0;k<T.nodes[u].child.size();k++) childLen[T.nodes[u].child[k]]=T.nodes[u].blen[k];
    vector<int> postorder, slot(nnodes,-1);
    function<void(int)> dfs=[&](int u){ for(int ch:T.nodes[u].child) dfs(ch); if(T.nodes[u].leaf<0){ slot[u]=postorder.size(); postorder.push_back(u);} };
    dfs(T.root); int nInternal=postorder.size();

    // tip states
    vector<unsigned char> tip((size_t)ntax*nptn);
    for (int u=0;u<nnodes;u++){ if(T.nodes[u].leaf<0) continue; int lf=T.nodes[u].leaf; const string& s=seqs[lf];
        for (int p=0;p<nptn;p++){ int a=aa_index(s[p]); tip[(size_t)lf*nptn+p]=(unsigned char)((a<0)?NS:a); } }
    unsigned char* d_tip; CK(cudaMalloc(&d_tip,tip.size())); CK(cudaMemcpy(d_tip,tip.data(),tip.size(),cudaMemcpyHostToDevice));

    vector<double> R2; (void)R2;
    map<string,double> oracle={{"g4",-7541976.9391},{"r8",-7556251.9185},{"r10",-7554280.5776},{"g1",-7974816.4323}};
    vector<string> models; if(modarg=="all") models={"g4","r8","r10","g1"}; else models={modarg};

    // device buffers sized for the largest NCAT we will run
    int maxNCAT=1; for(auto&m:models){ if(m=="g4")maxNCAT=max(maxNCAT,4); else if(m=="r8")maxNCAT=max(maxNCAT,8); else if(m=="r10")maxNCAT=max(maxNCAT,10); }
    size_t maxSlot=(size_t)maxNCAT*NS*nptn;
    double *d_echild=nullptr,*d_partial=nullptr,*d_patlh=nullptr;
    size_t maxEc=(size_t)nnodes*maxNCAT*NS*NS;
    CK(cudaMalloc(&d_echild, maxEc*sizeof(double)));
    CK(cudaMalloc(&d_partial,(size_t)nInternal*maxSlot*sizeof(double)));
    CK(cudaMalloc(&d_patlh,(size_t)nptn*sizeof(double)));
    vector<double> patlh(nptn);

    for (const string& model : models){
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
        else { fprintf(stderr,"unknown model %s\n",model.c_str()); continue; }

        double Q[20][20];
        for (int i=0;i<NS;i++){ double row=0; for(int j=0;j<NS;j++){ if(i!=j){ Q[i][j]=R[i][j]*f[j]; row+=Q[i][j]; } } Q[i][i]=-row; }
        double mu=0; for(int i=0;i<NS;i++) mu+=f[i]*(-Q[i][i]); for(int i=0;i<NS;i++) for(int j=0;j<NS;j++) Q[i][j]/=mu;
        double sq[20],B[20][20]; for(int i=0;i<NS;i++) sq[i]=sqrt(f[i]);
        for (int i=0;i<NS;i++) for(int j=0;j<NS;j++) B[i][j]=sq[i]*Q[i][j]/sq[j];
        for (int i=0;i<NS;i++) for(int j=i+1;j<NS;j++){ double m=0.5*(B[i][j]+B[j][i]); B[i][j]=B[j][i]=m; }
        double evl[20],V[20][20]; jacobi_eig(B,evl,V);
        vector<double> U(NS*NS),Uinv(NS*NS),evals(NS);
        for (int i=0;i<NS;i++){ evals[i]=evl[i]; for(int j=0;j<NS;j++){ U[i*NS+j]=V[i][j]/sq[i]; Uinv[i*NS+j]=V[j][i]*sq[j]; } }
        vector<double> UinvRowSum(NS,0.0); for(int i=0;i<NS;i++){ double s=0; for(int j=0;j<NS;j++) s+=Uinv[i*NS+j]; UinvRowSum[i]=s; }

        size_t ecStride=(size_t)NCAT*NS*NS;
        vector<double> echild((size_t)nnodes*ecStride,0.0);
        for (int c=0;c<nnodes;c++){ if(c==T.root) continue; for(int cat=0;cat<NCAT;cat++){ double len=childLen[c]*catRates[cat];
            double ex[NS]; for(int i=0;i<NS;i++) ex[i]=exp(evals[i]*len);
            double* e=&echild[(size_t)c*ecStride+(size_t)cat*NS*NS];
            for (int x=0;x<NS;x++) for(int i=0;i<NS;i++) e[x*NS+i]=U[x*NS+i]*ex[i]; } }

        CK(cudaMemcpyToSymbol(c_Uinv,Uinv.data(),sizeof(double)*NS*NS));
        CK(cudaMemcpyToSymbol(c_UinvRowSum,UinvRowSum.data(),sizeof(double)*NS));
        CK(cudaMemcpyToSymbol(c_freq,f,sizeof(double)*NS));
        CK(cudaMemcpyToSymbol(c_catw,catWeights.data(),sizeof(double)*NCAT));
        CK(cudaMemcpy(d_echild,echild.data(),echild.size()*sizeof(double),cudaMemcpyHostToDevice));

        size_t slotSz=(size_t)NCAT*NS*nptn;
        vector<Desc> desc(nInternal);
        for (int idx=0; idx<nInternal; idx++){ int u=postorder[idx]; Desc& D=desc[idx];
            D.isRoot=(u==T.root)?1:0; D.nchild=T.nodes[u].child.size();
            D.out=D.isRoot?nullptr:(d_partial+(size_t)slot[u]*slotSz);
            for(int k=0;k<3;k++){ D.ec[k]=nullptr; D.p[k]=nullptr; D.t[k]=nullptr; }
            for(int k=0;k<D.nchild;k++){ int c=T.nodes[u].child[k]; D.ec[k]=d_echild+(size_t)c*ecStride;
                if(T.nodes[c].leaf>=0) D.t[k]=d_tip+(size_t)T.nodes[c].leaf*nptn; else D.p[k]=d_partial+(size_t)slot[c]*slotSz; } }

        Ctx c{&desc,nInternal,nptn,NCAT,d_patlh,&patlh,oracle.count(model)?oracle[model]:0.0,reps};
        printf("\n======== model=%s NCAT=%d (reg cap = 65536/(threads*minBlocks); occ = minBlocks*threads/2048) ========\n", model.c_str(), NCAT);
        timeBase(c);                       // 128 regs, ~25% occ
        timeLB<256,3>("LB256/3", c);       // <=85 regs, 37.5%
        timeLB<256,4>("LB256/4", c);       // <=64 regs, 50%
        timeLB<256,5>("LB256/5", c);       // <=51 regs, 62.5%
        timeLB<256,6>("LB256/6", c);       // <=42 regs, 75%
        timeLB<128,4>("LB128/4", c);       // 128 regs, 25% (smaller block)
        timeLB<128,6>("LB128/6", c);       // <=85 regs, 37.5%
        timeLB<128,8>("LB128/8", c);       // <=64 regs, 50%
    }

    size_t gpufree=0,gputot=0; cudaMemGetInfo(&gpufree,&gputot);
    printf("\nVRAM used ~ %.2f GB / %.1f GB\n", (gputot-gpufree)/1073741824.0, gputot/1073741824.0);
    cudaFree(d_echild);cudaFree(d_partial);cudaFree(d_patlh);cudaFree(d_tip);
    return 0;
}
