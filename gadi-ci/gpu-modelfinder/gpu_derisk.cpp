// gpu_derisk.cpp — Phase G.0 de-risk: standalone BEAGLE harness for IQ-TREE-parity AA likelihood.
//
// Computes the LG+G4 log-likelihood of a FIXED tree + AA alignment via libhmsbeagle, on either the
// CPU or GPU (CUDA) plugin, and reports lnL (parity vs the IQ-TREE CPU reference) + timing.
// Phase G.0 of setonix-iq/research/Modelfinder/gpu-modelfinder-design.md.
//
// Parity target (gate 169643959, AA-100K, LG+G4): lnL = -7541976.853.
// LG matrix + freqs embedded from IQ-TREE src pll/models.c (exact, for eigendecomposition parity).
// Gamma: 4 cats, MEAN method, alpha=0.9963 -> rates {0.1362,0.4756,0.9994,2.3887}, weights 0.25.
//
// Build:  module load beagle-lib/4.0.1 cuda/12.x intel-compiler-llvm/2025.3.2
//         icpx -O2 -std=c++17 gpu_derisk.cpp -o gpu_derisk -lhmsbeagle
// Run:    ./gpu_derisk <alignment.phy> <tree.treefile> <cpu|gpu> [reps]
//
#include <libhmsbeagle/beagle.h>
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
static int g_beagle_errors = 0;
#define BC(x) do{ int _rc=(x); if(_rc<0){ g_beagle_errors++; fprintf(stderr,"[BEAGLE-ERR] rc=%d at line %d: %s\n",_rc,__LINE__,#x); fflush(stderr);} }while(0)
static double now_ms(Clock::time_point a, Clock::time_point b){
    return chrono::duration<double, milli>(b-a).count();
}

// ---- LG model (standard AA order A R N D C Q E G H I L K M F P S T W Y V), from pll/models.c ----
static const char* AA_ORDER = "ARNDCQEGHILKMFPSTWYV";
// lower-triangular exchangeabilities daa[i*20+j], j<i
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

// ---- symmetric Jacobi eigensolver (20x20, exact enough for parity) ----
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

// AA char -> state index in LG order; -1 == fully ambiguous (gap/X/?)
static int aa_index(char c) {
    c = toupper(c);
    static map<char,int> m;
    if (m.empty()) for (int i=0;i<20;i++) m[AA_ORDER[i]]=i;
    auto it=m.find(c);
    if (it!=m.end()) return it->second;
    return -1; // B,Z,J,X,-,?,* -> treat as fully ambiguous (sufficient for de-risk parity at 1e-3)
}

// ---- minimal Newick parser ----
struct Node { vector<int> child; vector<double> blen; int leaf=-1; }; // leaf = tip index or -1
struct Tree { vector<Node> nodes; int root=-1; vector<string> tipname; };
static Tree parse_newick(const string& s, map<string,int>& name2tip) {
    Tree T; size_t i=0;
    function<int(void)> parse = [&]() -> int {
        int id=T.nodes.size(); T.nodes.push_back(Node());
        if (s[i]=='(') {
            i++; // consume (
            while (true) {
                int c=parse();
                double bl=0.0;
                if (s[i]==':'){ i++; size_t j=i; while(i<s.size() && (isdigit(s[i])||s[i]=='.'||s[i]=='-'||s[i]=='e'||s[i]=='E'||s[i]=='+')) i++; bl=atof(s.substr(j,i-j).c_str()); }
                T.nodes[id].child.push_back(c); T.nodes[id].blen.push_back(bl);
                if (s[i]==',') { i++; continue; }
                if (s[i]==')') { i++; break; }
            }
        } else {
            size_t j=i; while(i<s.size() && s[i]!=':' && s[i]!=',' && s[i]!=')' && s[i]!=';') i++;
            string nm=s.substr(j,i-j);
            auto it=name2tip.find(nm);
            if (it==name2tip.end()){ fprintf(stderr,"tip '%s' not in alignment\n",nm.c_str()); exit(5); }
            T.nodes[id].leaf=it->second;
        }
        return id;
    };
    T.root=parse();
    return T;
}

int main(int argc, char** argv) {
    if (argc<4){ fprintf(stderr,"usage: %s <aln.phy> <tree> <cpu|gpu> [model=g4|fig4|r10] [reps]\n",argv[0]); return 1; }
    string alnpath=argv[1], treepath=argv[2], dev=argv[3];
    string model = (argc>4)? argv[4] : "g4";
    int reps = (argc>5)? atoi(argv[5]) : 20;
    bool useScale = (argc>6)? (atoi(argv[6])!=0) : true;  // manual log-scaling on by default
    bool dograd   = (argc>7)? (atoi(argv[7])!=0) : false; // compute branch-length gradient (pre-order pass)
    const int NS=20;
    // NCAT + category rates/weights + frequencies are model-dependent; set after the alignment parse.
    int NCAT = 4;

    // ---- parse alignment (sequential PHYLIP) ----
    ifstream af(alnpath); if(!af){ fprintf(stderr,"no aln\n"); return 2; }
    int ntax=0, nsite=0; { string h; getline(af,h); sscanf(h.c_str(),"%d %d",&ntax,&nsite); }
    vector<string> names(ntax), seqs(ntax);
    map<string,int> name2tip;
    for (int t=0;t<ntax;t++){
        string line; if(!getline(af,line)){ fprintf(stderr,"aln short\n"); return 2; }
        istringstream ls(line); string nm, sq; ls>>nm>>sq;
        names[t]=nm; seqs[t]=sq; name2tip[nm]=t;
        if ((int)sq.size()!=nsite){ fprintf(stderr,"taxon %s len %zu != %d\n",nm.c_str(),sq.size(),nsite); return 2; }
    }
    int nptn = nsite; // use all sites, weight 1 (total lnL == pattern-compressed lnL)
    printf("[aln] ntax=%d nsite=%d (using all sites as patterns)\n", ntax, nptn);

    // ---- empirical AA frequencies (for +F models); ambiguous chars skipped ----
    double fEmp[20]; { long cnt[20]={0}; long tot=0;
        for (int t=0;t<ntax;t++) for(char c: seqs[t]){ int a=aa_index(c); if(a>=0){ cnt[a]++; tot++; } }
        for (int k=0;k<20;k++) fEmp[k] = tot? (double)cnt[k]/(double)tot : 0.05;
    }

    // ---- parse tree ----
    string ts; { ifstream tf(treepath); stringstream ss; ss<<tf.rdbuf(); ts=ss.str(); }
    // strip whitespace
    string tc; for(char c: ts) if(!isspace((unsigned char)c)) tc+=c;
    Tree T = parse_newick(tc, name2tip);
    // Gradient needs a uniform BINARY tree so every pre-order op has exactly one sibling. IQ-TREE's
    // unrooted tree has a trifurcating root; re-root by inserting a degree-2 node R on one root edge
    // with a zero-length R↔oldRoot branch (lnL-invariant for reversible models — verified by parity).
    if (dograd){
        int oR=T.root, deg=T.nodes[oR].child.size();
        if (deg==3){
            int cL=T.nodes[oR].child.back(); double LcL=T.nodes[oR].blen.back();
            T.nodes[oR].child.pop_back(); T.nodes[oR].blen.pop_back();
            int R=T.nodes.size(); T.nodes.push_back(Node());
            T.nodes[R].child={oR,cL}; T.nodes[R].blen={0.0,LcL}; T.root=R;
        } else if (deg!=2){ fprintf(stderr,"reroot: root degree %d unsupported\n",deg); return 1; }
    }
    int nnodes=T.nodes.size();
    int rootdeg = T.nodes[T.root].child.size();
    printf("[tree] nodes=%d root_children=%d%s\n", nnodes, rootdeg, dograd?" (re-rooted binary for gradient)":"");

    // ---- BEAGLE buffer-index remap (CRITICAL) ----
    // BEAGLE requires tip partials at buffer indices [0, tipCount); internal-node partials must use
    // distinct indices >= tipCount. The Newick parser assigns DFS node IDs (root=0, interleaved), so
    // tips can land >= tipCount -> beagleSetTipPartials returns OUT_OF_RANGE (rc=-5). Remap:
    //   leaf node id   -> its alignment tip index  (already unique in [0, ntax))
    //   internal id    -> tipCount, tipCount+1, ...  (in [tipCount, nnodes))
    // Scratch buffers for the N-ary root then start safely at nnodes.
    vector<int> buf(nnodes, -1);
    { int internalCtr = ntax;
      for (int id=0; id<nnodes; id++)
          buf[id] = (T.nodes[id].leaf >= 0) ? T.nodes[id].leaf : internalCtr++;
    }

    // ---- LG exchangeabilities + base (LG) frequencies ----
    double R[20][20], f[20]; fill_LG(R,f);

    // ---- model setup: choose frequencies + category rates/weights ----
    // Parity for heavier models is CPU-plugin ≡ GPU-plugin on the SAME fixed tree (the rigorous
    // GPU-correctness test). r10 rates/weights are representative valid FreeRate values (Σw=1, Σwr=1),
    // NOT IQ-TREE-fitted — the de-risk question is GPU compute correctness+speed, not the fitted params.
    vector<double> catRates, catWeights;       // primary category set (pass A for r10split)
    vector<double> catRatesB, catWeightsB;      // 2nd category group — r10split only (>8-cat workaround)
    bool hasF = (model.find('f')!=string::npos);
    if (hasF) for(int k=0;k<20;k++) f[k]=fEmp[k];   // +F: empirical frequencies
    if (model=="g1"){
        NCAT=1; catRates={1.0}; catWeights={1.0};   // single-rate LG: diagnostic isolating GPU NCAT-dependence of the gradient
    } else if (model=="g4"){
        NCAT=4; catRates={0.1362,0.4756,0.9994,2.3887}; catWeights={0.25,0.25,0.25,0.25};
    } else if (model=="fig4"){
        // LG+F+I+G4: cat0 = invariant (rate 0), cats 1-4 = gamma (reuse α=0.9963 rates); renormalise Σwr=1
        double pinv=0.1, g[4]={0.1362,0.4756,0.9994,2.3887};
        NCAT=5; catRates={0.0,g[0],g[1],g[2],g[3]};
        double wg=(1.0-pinv)/4.0; catWeights={pinv,wg,wg,wg,wg};
        double mean=0; for(int k=0;k<NCAT;k++) mean+=catWeights[k]*catRates[k];
        for(int k=0;k<NCAT;k++) catRates[k]/=mean;   // mean rate -> 1
    } else if (model=="r8"){
        NCAT=8;   // FreeRate with 8 categories — at/below BEAGLE-CUDA's kMatrixBlockSize=8 cap
        double br[8]={0.06,0.18,0.36,0.60,0.92,1.40,2.20,4.00};
        double bw[8]={0.07,0.11,0.14,0.17,0.17,0.14,0.11,0.09};
        catRates.assign(br,br+8); catWeights.assign(bw,bw+8);
        double sw=0; for(double w:catWeights) sw+=w; for(double&w:catWeights) w/=sw;       // Σw=1
        double mean=0; for(int k=0;k<NCAT;k++) mean+=catWeights[k]*catRates[k];
        for(int k=0;k<NCAT;k++) catRates[k]/=mean;                                         // Σwr=1
    } else if (model=="r10" || model=="r10split"){
        // Build the full 10-category FreeRate set (Σw=1, Σwr=1), then either use all 10 (r10, fails on
        // stock BEAGLE CUDA) or split 5+5 across two ≤8-cat passes combined per-site (r10split).
        double br[10]={0.05,0.15,0.30,0.50,0.75,1.00,1.40,2.00,3.00,5.00};
        double bw[10]={0.05,0.08,0.10,0.12,0.15,0.15,0.12,0.10,0.08,0.05};
        vector<double> r(br,br+10), w(bw,bw+10);
        double sw=0; for(double x:w) sw+=x; for(double&x:w) x/=sw;                          // Σw=1
        double mean=0; for(int k=0;k<10;k++) mean+=w[k]*r[k]; for(double&x:r) x/=mean;       // Σwr=1
        if (model=="r10"){ NCAT=10; catRates=r; catWeights=w; }
        else { // r10split: pass A = cats 0-4, pass B = cats 5-9 (global weights kept, NOT renormalised)
            NCAT=5;
            catRates ={r[0],r[1],r[2],r[3],r[4]}; catWeights ={w[0],w[1],w[2],w[3],w[4]};
            catRatesB={r[5],r[6],r[7],r[8],r[9]}; catWeightsB={w[5],w[6],w[7],w[8],w[9]};
        }
    } else { fprintf(stderr,"unknown model '%s' (g4|fig4|r8|r10|r10split)\n",model.c_str()); return 1; }
    printf("[model] %s  freqs=%s  NCAT=%d\n", model.c_str(), hasF?"empirical":"LG", NCAT);

    // ---- LG eigendecomposition (reversible-model symmetric trick) ----
    double Q[20][20];
    for (int i=0;i<NS;i++){ double row=0; for(int j=0;j<NS;j++){ if(i!=j){ Q[i][j]=R[i][j]*f[j]; row+=Q[i][j]; } } Q[i][i]=-row; }
    // normalize so mean rate = 1: scale = sum_i f[i]*(-Q[i][i])
    double mu=0; for(int i=0;i<NS;i++) mu += f[i]*(-Q[i][i]);
    for (int i=0;i<NS;i++) for(int j=0;j<NS;j++) Q[i][j]/=mu;
    // B = diag(sqrt f) Q diag(1/sqrt f)  (symmetric)
    double sq[20], B[20][20];
    for (int i=0;i<NS;i++) sq[i]=sqrt(f[i]);
    for (int i=0;i<NS;i++) for(int j=0;j<NS;j++) B[i][j]=sq[i]*Q[i][j]/sq[j];
    // symmetrize numerically
    for (int i=0;i<NS;i++) for(int j=i+1;j<NS;j++){ double m=0.5*(B[i][j]+B[j][i]); B[i][j]=B[j][i]=m; }
    double eval[20], V[20][20]; jacobi_eig(B,eval,V); // B destroyed; V columns = eigenvectors
    // U = diag(1/sqrt f) V ; Uinv = V^T diag(sqrt f)
    vector<double> U(NS*NS), Uinv(NS*NS), evals(NS);
    for (int i=0;i<NS;i++){ evals[i]=eval[i];
        for (int j=0;j<NS;j++){ U[i*NS+j]=V[i][j]/sq[i]; Uinv[i*NS+j]=V[j][i]*sq[j]; } }
    printf("[eig] eigenvalues[0..3]= %.6f %.6f %.6f %.6f (one ~0 expected)\n", evals[0],evals[1],evals[2],evals[3]);

    // ---- BEAGLE instance ----
    int tipCount=ntax;
    int internalCount=nnodes-ntax;
    // Tips use COMPACT state buffers (setTipStates) — NCAT-independent, ~0.4MB/tip vs 160MB/tip at
    // NCAT=10. With compactBufferCount=ntax, the partial-buffer index space is [tipCount, tipCount+
    // partialsBufferCount); internals [tipCount,nnodes) + N-ary scratch [nnodes,..) live there.
    int compactBufferCount  = ntax;
    // Gradient mode: one pre-order partial buffer per node mapped as preB_(id)=nnodes+id, so max index
    // = nnodes+(nnodes-1) = 2*nnodes-1 = 397. The GPU root-seed uses beagleSetPartials, whose CUDA range
    // check is `bufferIndex >= kPartialsBufferCount` (==partialsBufferCount, NOT tipCount+that) → the
    // index MUST be < partialsBufferCount. So partialsBufferCount has to EXCEED the max pre-order index
    // 397 ⇒ use 2*nnodes+4 = 402. (The earlier 302 made setPartials(397) fail rc=-5 OUT_OF_RANGE; CPU's
    // setRootPrePartials checks the wider global space so it tolerated 302.) VRAM: GPU pads AA 20→32
    // states, so 402 partials × (nptn·32·NCAT·8) = 41 GB at NCAT=4 ⇒ needs A100-80GB (dgxa100), not V100.
    // CPU (20 states): 402 × (nptn·20·NCAT·8) = 25.7 GB, fits a 90 GB node.
    // Matrices (gradient): P per edge in [0,nnodes), identity at nnodes, differential (generator) at
    // dQidx=nnodes+1, and — for the GPU manual-transpose path (bug 16) — a TRANSPOSED copy of each edge P
    // at TP(c)=nnodes+2+buf[c] ∈ [nnodes+2, 2*nnodes+2). beagleCalculateEdgeDerivatives takes a DIFFERENTIAL
    // matrix (set via beagleSetDifferentialMatrix), NOT per-edge dP/dt. ⇒ matrixBufferCount = 2*nnodes+2.
    int partialsBufferCount = dograd ? (2*nnodes + 4)                // 402; > max pre-order index 397 (setPartials range)
                                     : (internalCount + rootdeg + 2); // internals + N-ary scratch (tips compact)
    int matrixBufferCount   = dograd ? (2*nnodes + 2) : (nnodes + 2); // edges + identity + diff + transposed edges (grad)
    int scaleBufferCount    = nnodes + rootdeg + 4;   // one per op + cumulative + slack
    // Scaling: manual per-op log-scalers prevent underflow on deep trees. BUT BEAGLE 4.0.1's CUDA
    // rescaling kernel hard-exits ("Not yet implemented! Try slow reweighing.") once stateCount×NCAT
    // makes the scale-grid Y>1 (e.g. 20×10). useScale=0 disables scaling (SCALING_NONE) — only valid
    // if double precision doesn't underflow on this tree; lnL value is identical when it doesn't.
    long scaleflag = useScale ? (BEAGLE_FLAG_SCALING_MANUAL | BEAGLE_FLAG_SCALERS_LOG)
                              : 0L;  // no scaling flags == no rescaling
    // Pre-order transposes are done MANUALLY via beagleTransposeTransitionMatrices (matching BEAGLE's own
    // hmctest), NOT via BEAGLE_FLAG_PREORDER_TRANSPOSE_AUTO — AUTO did not transpose matrix1 in 4.0.1
    // (job 170129095/618: AUTO+Q and AUTO+Qᵀ both wrong), so the flag is omitted to avoid double-transpose.
    long pref = (dev=="gpu") ? (BEAGLE_FLAG_PROCESSOR_GPU|BEAGLE_FLAG_FRAMEWORK_CUDA|BEAGLE_FLAG_PRECISION_DOUBLE|scaleflag)
                             : (BEAGLE_FLAG_PROCESSOR_CPU|BEAGLE_FLAG_PRECISION_DOUBLE|scaleflag);
    printf("[scale] %s\n", useScale ? "manual log-scalers" : "NONE (double-precision, no rescaling)");
    BeagleInstanceDetails details;
    int inst = beagleCreateInstance(tipCount, partialsBufferCount, compactBufferCount, NS, nptn,
                                    1 /*eigen*/, matrixBufferCount, NCAT, scaleBufferCount,
                                    NULL, 0, pref, 0, &details);
    if (inst<0){ fprintf(stderr,"beagleCreateInstance failed: %d\n",inst); return 3; }
    printf("[beagle] instance=%d  resource=%d (%s)  impl=%s\n", inst, details.resourceNumber,
           details.resourceName?details.resourceName:"?", details.implName?details.implName:"?");

    // tip states (compact) — indexed by REMAPPED buffer index buf[id] (∈ [0,tipCount)). Unambiguous
    // AA -> its 0..19 state; ambiguous (B/Z/J/X/gap) -> state NS (==20), BEAGLE's "fully ambiguous".
    {
        vector<int> st(nptn);
        for (int id=0; id<nnodes; id++){
            if (T.nodes[id].leaf < 0) continue;
            const string& sqs=seqs[T.nodes[id].leaf];
            for (int p=0;p<nptn;p++){ int idx=aa_index(sqs[p]); st[p] = (idx<0)? NS : idx; }
            BC(beagleSetTipStates(inst, buf[id], st.data()));
        }
    }
    fprintf(stderr,"[chk] tip partials set\n"); fflush(stderr);
    // pattern weights (all 1)
    { vector<double> w(nptn,1.0); BC(beagleSetPatternWeights(inst, w.data())); }
    // eigen decomposition, state freqs, category rates/weights
    BC(beagleSetEigenDecomposition(inst, 0, U.data(), Uinv.data(), evals.data()));
    BC(beagleSetStateFrequencies(inst, 0, f));
    BC(beagleSetCategoryRates(inst, catRates.data()));
    BC(beagleSetCategoryWeights(inst, 0, catWeights.data()));
    fprintf(stderr,"[chk] model set (eigen/freq/rates/weights)\n"); fflush(stderr);
    if (g_beagle_errors){ fprintf(stderr,"[FATAL] %d BEAGLE setup error(s) — aborting before compute\n", g_beagle_errors); return 4; }

    // ---- build postorder operation list + edge list ----
    // All partial/matrix references use the REMAPPED buffer index buf[node]. Matrix buffer for a
    // child's branch == buf[child] (matrixBufferCount covers this); identity at buf-space nnodes.
    vector<int> matIdx; vector<double> matLen;
    vector<BeagleOperation> ops;
    int IDENT = nnodes;        // matrix buffer index for the zero-length identity (N-ary root combine)
    int nextScratch = nnodes;  // partial scratch buffers start at nnodes (internals end at nnodes-1)
    function<void(int)> post = [&](int u){
        Node& N=T.nodes[u];
        for (size_t k=0;k<N.child.size();k++){
            post(N.child[k]);
            int c=N.child[k];
            matIdx.push_back(buf[c]); matLen.push_back(N.blen[k]); // matrix index buf[c], branch len
        }
        if (N.leaf>=0) return; // tip: partial already set
        // combine children pairwise into this node's partial (buffer buf[u]).
        // root may be N-ary; non-root internal is binary in IQ-TREE newick.
        vector<int> ch(N.child.begin(), N.child.end());
        if (ch.size()==2){
            BeagleOperation op = { buf[u], BEAGLE_OP_NONE, BEAGLE_OP_NONE,
                                   buf[ch[0]], buf[ch[0]], buf[ch[1]], buf[ch[1]] };
            ops.push_back(op);
        } else {
            // pairwise: acc = (ch0 via M_ch0) x (ch1 via M_ch1) -> scratch; then x ch_i via identity
            int acc = nextScratch++;
            BeagleOperation op0 = { acc, BEAGLE_OP_NONE, BEAGLE_OP_NONE,
                                    buf[ch[0]], buf[ch[0]], buf[ch[1]], buf[ch[1]] };
            ops.push_back(op0);
            for (size_t k=2;k<ch.size();k++){
                int dest = (k==ch.size()-1)? buf[u] : nextScratch++;
                // acc combined with identity matrix (IDENT, len 0), ch[k] via its own matrix buf[ch[k]]
                BeagleOperation op = { dest, BEAGLE_OP_NONE, BEAGLE_OP_NONE,
                                       acc, IDENT, buf[ch[k]], buf[ch[k]] };
                ops.push_back(op);
                acc=dest;
            }
        }
    };
    post(T.root);
    // manual scaling: give each operation its own scale-write buffer; accumulate into cumScale.
    // When useScale is off, write BEAGLE_OP_NONE so no scale buffers are touched.
    vector<int> scaleIdx(ops.size());
    for (size_t k=0;k<ops.size();k++){ ops[k].destinationScaleWrite= useScale?(int)k:BEAGLE_OP_NONE; ops[k].destinationScaleRead=BEAGLE_OP_NONE; scaleIdx[k]=(int)k; }
    int cumScale = useScale ? (int)ops.size() : BEAGLE_OP_NONE;
    printf("[ops] edges=%zu internal-ops=%zu cumScaleBuf=%d\n", matIdx.size(), ops.size(), cumScale);

    // ---- evaluate lnL (timed, reps) ----
    auto evalLnL = [&](double& lnl)->double {
        auto t0=Clock::now();
        // transition matrices for all edges (+ identity at IDENT with len 0)
        vector<int> mi=matIdx; vector<double> ml=matLen;
        mi.push_back(IDENT); ml.push_back(0.0);
        static int once=0; if(!once) fprintf(stderr,"[chk] updateTransitionMatrices (n=%zu)\n",mi.size());
        BC(beagleUpdateTransitionMatrices(inst, 0, mi.data(), NULL, NULL, ml.data(), mi.size()));
        if(!once) fprintf(stderr,"[chk] updatePartials (nops=%zu)\n",ops.size());
        BC(beagleUpdatePartials(inst, ops.data(), ops.size(), BEAGLE_OP_NONE));
        if (useScale){
            if(!once) fprintf(stderr,"[chk] reset+accumulate scale\n");
            BC(beagleResetScaleFactors(inst, cumScale));
            BC(beagleAccumulateScaleFactors(inst, scaleIdx.data(), (int)scaleIdx.size(), cumScale));
        }
        int rootIndex=buf[T.root];
        int catWeightsIdx=0, stateFreqIdx=0, cumScaleIdx=cumScale;
        if(!once){ fprintf(stderr,"[chk] calculateRootLogLikelihoods\n"); once=1; }
        BC(beagleCalculateRootLogLikelihoods(inst, &rootIndex, &catWeightsIdx, &stateFreqIdx,
                                          &cumScaleIdx, 1, &lnl));
        auto t1=Clock::now();
        return now_ms(t0,t1);
    };
    double lnl=0;
    if (model=="r10split"){
        // >8-category workaround: evaluate the 10 rate categories as two ≤5-cat passes (same instance,
        // re-set rates/weights between passes), then combine per-site: L=ΣA+ΣB, lnL=Σ log(exp(logA)+exp(logB)).
        // Mathematically identical to a single NCAT=10 model ⇒ must match the CPU r10 lnL exactly.
        vector<double> logA(nptn), logB(nptn);
        auto evalSplit=[&](double& combined)->double {
            auto t0=Clock::now();
            BC(beagleSetCategoryRates(inst,catRates.data()));  BC(beagleSetCategoryWeights(inst,0,catWeights.data()));
            double la; evalLnL(la); BC(beagleGetSiteLogLikelihoods(inst, logA.data()));
            BC(beagleSetCategoryRates(inst,catRatesB.data())); BC(beagleSetCategoryWeights(inst,0,catWeightsB.data()));
            double lb; evalLnL(lb); BC(beagleGetSiteLogLikelihoods(inst, logB.data()));
            double s=0; for(int p=0;p<nptn;p++){ double a=logA[p],b=logB[p],m=(a>b?a:b); s += m + log1p(exp((a<b?a:b)-m)); }
            combined=s; auto t1=Clock::now(); return now_ms(t0,t1);
        };
        double warm=evalSplit(lnl);
        printf("[lnL] %.4f   (warmup %.2f ms)   model=r10split (5+5 cat-split)  parity target -7554280.5776  |delta|=%.4f\n",
               lnl, warm, fabs(lnl-(-7554280.5776)));
        double tmin=1e18,tsum=0;
        for (int r=0;r<reps;r++){ double t=evalSplit(lnl); tmin=min(tmin,t); tsum+=t; }
        printf("[timing r10split %s] 2-pass lnL eval: min=%.3f ms  mean=%.3f ms  over %d reps\n",
               dev.c_str(), tmin, tsum/reps, reps);
    } else {
        double warm=evalLnL(lnl);
        if (model=="g4")
            printf("[lnL] %.4f   (warmup %.2f ms)   parity target -7541976.853  |delta|=%.4f\n",
                   lnl, warm, fabs(lnl-(-7541976.853)));
        else
            printf("[lnL] %.4f   (warmup %.2f ms)   model=%s (parity = CPU plugin vs GPU plugin)\n",
                   lnl, warm, model.c_str());
        double tmin=1e18,tsum=0;
        for (int r=0;r<reps;r++){ double t=evalLnL(lnl); tmin=min(tmin,t); tsum+=t; }
        printf("[timing %s %s] lnL eval: min=%.3f ms  mean=%.3f ms  over %d reps\n",
               model.c_str(), dev.c_str(), tmin, tsum/reps, reps);
    }

    // ---- branch-length gradient: pre-order pass + edge derivatives (Ji et al. O(N)) ----
    // The exact piece Mode-L got wrong (10⁵⁴ overflow). BEAGLE computes it; we FD-validate + time it.
    if (dograd){
        vector<int> bufToNode(nnodes,-1);
        for (int id=0; id<nnodes; id++) bufToNode[buf[id]]=id;
        auto preB_  = [&](int id){ return nnodes + id; };            // pre-order partial buffer for node id
        auto TP     = [&](int c ){ return nnodes + 2 + buf[c]; };    // transposed copy of edge-c's P matrix (GPU only)

        // ---- ONE differential (generator) matrix shared by all edges ----
        // beagleCalculateEdgeDerivatives takes a DIFFERENTIAL matrix (the infinitesimal generator Q),
        // NOT a per-edge dP/dt. d/dt log L uses dP_c/dt = Q·r_c·P_c, and BEAGLE's derivative kernel
        // IGNORES categoryRates, so the per-category rate r_c must be baked into each S×S block:
        //   diff[c·NS·NS + i·NS + j] = Q[i][j]·catRates[c]      (Q = normalized generator from §eig)
        // Set ONCE (branch-independent) via beagleSetDifferentialMatrix (→ setTransitionMatrix(idx,m,0)).
        int dQidx = nnodes + 1;                                      // spare matrix buffer (edges fill [0,nnodes), identity at nnodes)
        // GPU transpose convention (matches BEAGLE hmctest exactly): for stateCount>4 BOTH (a) the pre-order
        // transition matrix1 AND (b) the differential matrix must be the TRANSPOSE; matrix2 (sibling) stays
        // forward. CPU (transpose offset 0) uses everything untransposed (its kernel transposes matrix1
        // internally). So on GPU we set diff = Qᵀ·r_c here, and transpose each edge P at runtime into TP(c).
        bool gpuT = (dev=="gpu");
        {
            vector<double> diff((size_t)NS*NS*NCAT);
            for (int c=0;c<NCAT;c++) for(int i=0;i<NS;i++) for(int j=0;j<NS;j++)
                diff[(size_t)c*NS*NS + i*NS + j] = (gpuT ? Q[j][i] : Q[i][j]) * catRates[c];
            BC(beagleSetDifferentialMatrix(inst, dQidx, diff.data()));
        }

        // pre-order op list (binary tree ⇒ each non-root node has exactly one sibling). Convention
        // (BEAGLE upPrePartials + Ji et al. Eq.7): {dest=pre[c], sW, sR, parent's pre-partial,
        // current node's matrix [TRANSPOSED on GPU = TP(c)], sibling's post-partial, sibling's matrix [forward]}.
        vector<BeagleOperation> preops;
        function<void(int)> prewalk = [&](int u){
            auto& ch=T.nodes[u].child;
            for (size_t a=0; a<ch.size(); a++){
                int c=ch[a], sib=ch[a^1];
                int mat1 = gpuT ? TP(c) : buf[c];   // matrix1: transposed edge-P on GPU, plain on CPU
                BeagleOperation op = { preB_(c), BEAGLE_OP_NONE, BEAGLE_OP_NONE,
                                       preB_(u), mat1, buf[sib], buf[sib] };
                preops.push_back(op); prewalk(c);
            }
        };
        prewalk(T.root);

        // differentiate every non-root edge (matIdx order). derivativeMatrixIndices ALL point at the
        // single differential matrix dQidx (NOT per-edge dP/dt).
        int nE=(int)matIdx.size();
        vector<int> postB(nE), preBuf(nE), dMidx(nE, dQidx), echild(nE), catWArr(nE, 0), matIdxT(nE);
        for (int k=0;k<nE;k++){ int c=bufToNode[matIdx[k]]; postB[k]=buf[c]; preBuf[k]=preB_(c); echild[k]=c; matIdxT[k]=TP(c); }
        // outSumDerivatives (size nE) IS the gradient d lnL/dt per edge. outDerivatives (per-edge × per-
        // pattern, size nE·nptn) MUST be NULL — passing an nE-sized buffer overflows BEAGLE's write
        // (count·kPatternCount doubles) → the earlier CPU SEGFAULT. outSumSquaredDerivatives also NULL.
        vector<double> sumDeriv(nE,0.0);
        // root pre-order partial = state frequencies, replicated over every pattern × category.
        vector<double> rootSeed((size_t)nptn*NS*NCAT);
        for (int c=0;c<NCAT;c++) for(int p=0;p<nptn;p++) for(int s=0;s<NS;s++)
            rootSeed[((size_t)c*nptn+p)*NS+s] = f[s];

        auto evalGrad=[&]()->double {
            auto t0=Clock::now();
            BC(beagleUpdateTransitionMatrices(inst, 0, matIdx.data(), NULL, NULL, matLen.data(), nE)); // P only
            // GPU: transpose each edge P into TP(c) for the pre-order matrix1 (manual transpose, like hmctest).
            if (gpuT) BC(beagleTransposeTransitionMatrices(inst, matIdx.data(), matIdxT.data(), nE));
            BC(beagleUpdatePartials(inst, ops.data(), ops.size(), BEAGLE_OP_NONE));
            // Seed root pre-partial. CUDA's beagleSetRootPrePartials is a stub (NO_IMPLEMENTATION); seed
            // the buffer directly with frequencies via beagleSetPartials instead. CPU uses the real call.
            if (dev=="gpu") { BC(beagleSetPartials(inst, preB_(T.root), rootSeed.data())); }
            else            { int sfIdx=0, rootPre=preB_(T.root); BC(beagleSetRootPrePartials(inst, &rootPre, &sfIdx, 1)); }
            BC(beagleUpdatePrePartials(inst, preops.data(), preops.size(), BEAGLE_OP_NONE));
            BC(beagleCalculateEdgeDerivatives(inst, postB.data(), preBuf.data(), dMidx.data(),
                                             catWArr.data(), nE,
                                             NULL, sumDeriv.data(), NULL));
            auto t1=Clock::now(); return now_ms(t0,t1);
        };
        // unscaled lnL helper for finite-difference validation
        auto lnlOf=[&](const vector<double>& ml)->double {
            BC(beagleUpdateTransitionMatrices(inst,0,matIdx.data(),NULL,NULL,ml.data(),nE));
            BC(beagleUpdatePartials(inst, ops.data(), ops.size(), BEAGLE_OP_NONE));
            int ri=buf[T.root],cw=0,sf=0,cs=BEAGLE_OP_NONE; double l;
            BC(beagleCalculateRootLogLikelihoods(inst,&ri,&cw,&sf,&cs,1,&l)); return l;
        };

        fprintf(stderr,"[chk] gradient: %d edges, %zu pre-ops, dQidx=%d\n", nE, preops.size(), dQidx); fflush(stderr);
        double warm=evalGrad();
        // FD-check the 5 longest edges (skip ~0-length incl. the spurious re-root edge).
        // The objective is lnL ≈ -7.5e6, so a too-small eps makes (L(+eps)-L(-eps)) lose ~all sig figs to
        // roundoff: eps=1e-5 gives ~0.7% noise (NOT a gradient error — analytic/FD ratios scatter both
        // sides of 1). Sweep eps and take the best-converged FD; pass at 5e-3. central FD = (L+ - L-)/2eps.
        auto centralFD=[&](int k,double eps)->double{ vector<double> ml=matLen;
            ml[k]=matLen[k]+eps; double lp=lnlOf(ml); ml[k]=matLen[k]-eps; double lm=lnlOf(ml);
            return (lp-lm)/(2*eps); };
        vector<int> ord(nE); for(int i=0;i<nE;i++) ord[i]=i;
        sort(ord.begin(),ord.end(),[&](int a,int b){return matLen[a]>matLen[b];});
        printf("[grad %s %s] %d edges; warmup %.2f ms; FD check (eps sweep, best of {1e-2,1e-3,1e-4}):\n",
               model.c_str(), dev.c_str(), nE, warm);
        const double epsS[3]={1e-2,1e-3,1e-4};
        int checked=0; double worst=0;
        for (int ii=0; ii<nE && checked<5; ii++){
            int k=ord[ii]; if (matLen[k]<1e-3) continue;     // skip ~0-length edges (FD ill-conditioned)
            double an=sumDeriv[k], bestfd=0, bestrel=1e18; double besteps=0;
            for (double eps: epsS){ double fd=centralFD(k,eps);
                double rel=fabs(an-fd)/(fabs(an)+fabs(fd)+1e-12);
                if (rel<bestrel){ bestrel=rel; bestfd=fd; besteps=eps; } }
            worst=max(worst, bestrel);
            printf("   child=%d len=%.5f  analytic=%.6f  bestFD=%.6f (eps=%.0e)  rel=%.2e\n",
                   echild[k], matLen[k], an, bestfd, besteps, bestrel);
            checked++;
        }
        printf("[grad %s %s] FD check: gradient (sumDerivatives) worst rel=%.2e over %d edges => %s\n",
               model.c_str(), dev.c_str(), worst, checked, (worst<5e-3)?"PASS":"FAIL");
        // re-run gradient eval to time it (matrices restored to base lengths by lnlOf's last call above)
        double tmin=1e18,tsum=0; for (int r=0;r<reps;r++){ double t=evalGrad(); tmin=min(tmin,t); tsum+=t; }
        printf("[timing %s %s GRAD] lnL+grad eval: min=%.3f ms  mean=%.3f ms  over %d reps\n",
               model.c_str(), dev.c_str(), tmin, tsum/reps, reps);
    }

    beagleFinalizeInstance(inst);
    return 0;
}
