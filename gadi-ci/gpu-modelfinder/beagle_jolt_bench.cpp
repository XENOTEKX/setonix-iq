// beagle_jolt_bench.cpp — matched apples-to-apples LIKELIHOOD + GRADIENT eval benchmark for the JOLT-vs-BEAGLE-4.0 study.
//
// Measures the per-eval wall time of:
//   (A) ONE post-order partial-likelihood sweep + root log-likelihood        (== JOLT k1_node)
//   (B) ONE pre-order partial-likelihood sweep + all-branch edge derivatives (== JOLT kj_pre all-branch gradient)
// on the BEAGLE v4.0.0 tensor-cores branch, selectable on the FP64 tensor-core resource (BEAGLE_FLAG_VECTOR_TENSOR)
// vs the standard FP64 CUDA-core resource. Same dimensions / FP64 / tree / model on both => fair race; lnL is
// bit-checkable tensor-vs-cuda, and the gradient is self-checked by finite differences of the lnL.
//
// Model: s-state (default 20=AA) reversible generalized-JC (uniform freqs, equal exchangeabilities); exact Q is
// irrelevant to timing and IDENTICAL across resources => exact tensor-vs-cuda parity. Discrete +G4 by default.
// Tree: rooted caterpillar of nTaxa tips (nTaxa-1 internal nodes). Tip data: random partials. NO scaling (NORM_LH).
//
// Usage: beagle_jolt_bench <states> <nTaxa> <nPatterns> <rateCats> <reps> <warmup> <mode:tensor|cuda|cpu> [dograd=1]
//
#include <libhmsbeagle/beagle.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>
#include <random>
#include <unistd.h>

using clk = std::chrono::high_resolution_clock;
static double ms_since(clk::time_point t0){ return std::chrono::duration<double,std::milli>(clk::now()-t0).count(); }

// generalized-JC eigendecomposition on s states (uniform stationary freq). Q = mu(J - sI), mu=1/(s-1) (mean rate 1).
// eval: 0 (once), -s/(s-1) (s-1 times). evec col0=ones; cols 1..s-1 = Helmert contrasts. ivec = diag(1/||col||^2) evec^T.
static void build_jc_eigen(int s, std::vector<double>& evec, std::vector<double>& ivec, std::vector<double>& eval){
    evec.assign(s*s,0.0); ivec.assign(s*s,0.0); eval.assign(s,0.0);
    const double lambda = -double(s)/double(s-1);
    eval[0]=0.0; for(int j=1;j<s;++j) eval[j]=lambda;
    std::vector<double> nrm2(s,0.0);
    for(int i=0;i<s;++i) evec[i*s+0]=1.0; nrm2[0]=s;
    for(int k=1;k<s;++k){ for(int i=0;i<k;++i) evec[i*s+k]=1.0; evec[k*s+k]=-double(k); nrm2[k]=double(k)*double(k+1); }
    for(int j=0;j<s;++j) for(int i=0;i<s;++i) ivec[j*s+i]=evec[i*s+j]/nrm2[j];
}

int main(int argc, char** argv){
    int s        = argc>1 ? atoi(argv[1]) : 20;
    int nTaxa    = argc>2 ? atoi(argv[2]) : 100;
    int nPat     = argc>3 ? atoi(argv[3]) : 10000;
    int nCat     = argc>4 ? atoi(argv[4]) : 4;
    int reps     = argc>5 ? atoi(argv[5]) : 50;
    int warmup   = argc>6 ? atoi(argv[6]) : 5;
    const char* mode = argc>7 ? argv[7] : "tensor";
    int dograd   = argc>8 ? atoi(argv[8]) : 1;

    fprintf(stdout,"=== beagle_jolt_bench  s=%d nTaxa=%d nPat=%d nCat=%d reps=%d warmup=%d mode=%s grad=%d ===\n",
            s,nTaxa,nPat,nCat,reps,warmup,mode,dograd);

    const int nInternal = nTaxa-1;
    const int nNodes    = 2*nTaxa-1;
    const int rootIdx   = nNodes-1;
    const int PRE       = nNodes;           // pre-order partial buffer of node v = PRE+v
    const int MT        = nNodes;           // transposed forward matrix of node v = MT+v
    const int DQ        = 2*nNodes;         // generator Q  (scaled per category)
    const int DQT       = 2*nNodes+1;       // generator Q^T

    BeagleResourceList* rl = beagleGetResourceList();
    fprintf(stdout,"Resources (%d):\n",rl->length);
    for(int i=0;i<rl->length;++i)
        fprintf(stdout,"  [%d] %s | flags=0x%lx\n",i,rl->list[i].name,rl->list[i].supportFlags);

    long prefer = BEAGLE_FLAG_SCALERS_RAW;
    long require = BEAGLE_FLAG_EIGEN_REAL;
    if(strcmp(mode,"cpu")==0){ prefer |= BEAGLE_FLAG_PROCESSOR_CPU | BEAGLE_FLAG_PRECISION_DOUBLE; }
    else { prefer |= BEAGLE_FLAG_PROCESSOR_GPU | BEAGLE_FLAG_PRECISION_DOUBLE; require |= BEAGLE_FLAG_FRAMEWORK_CUDA;
           if(strcmp(mode,"tensor")==0) require |= BEAGLE_FLAG_VECTOR_TENSOR; }

    BeagleInstanceDetails det;
    int inst = beagleCreateInstance(
        nTaxa,                 // tipCount
        dograd? 2*nNodes : nNodes, // partialsBufferCount (post 0..nNodes-1 [tips+internal]; pre PRE+v if grad)
        0,                     // compactBufferCount
        s, nPat,
        1,                     // eigenBufferCount
        dograd? 2*nNodes+2 : nNodes, // matrixBufferCount
        nCat,
        0,                     // scaleBufferCount (NORM_LH)
        NULL, 0, prefer, require, &det);
    if(inst<0){ fprintf(stderr,"FAILED to create BEAGLE instance (mode=%s ret=%d) — likely OOM at these dims\n",mode,inst); return 2; }
    fprintf(stdout,"Using resource %d: %s | impl=%s | flags=0x%lx\n",det.resourceNumber,det.resourceName,det.implName,det.flags);
    const bool gotTensor = (det.flags & BEAGLE_FLAG_VECTOR_TENSOR)!=0;
    fprintf(stdout,"  VECTOR_TENSOR active: %s\n", gotTensor?"YES":"no");

    // model
    std::vector<double> evec,ivec,eval; build_jc_eigen(s,evec,ivec,eval);
    beagleSetEigenDecomposition(inst,0,evec.data(),ivec.data(),eval.data());
    std::vector<double> freqs(s,1.0/s);          beagleSetStateFrequencies(inst,0,freqs.data());
    std::vector<double> weights(nCat,1.0/nCat);  beagleSetCategoryWeights(inst,0,weights.data());
    std::vector<double> rates(nCat);
    { double sum=0; for(int c=0;c<nCat;++c){ rates[c]=0.2+1.6*c/std::max(1,nCat-1); sum+=rates[c]; }
      for(int c=0;c<nCat;++c) rates[c]*=nCat/sum; }
    beagleSetCategoryRates(inst,rates.data());
    std::vector<double> pw(nPat,1.0);            beagleSetPatternWeights(inst,pw.data());

    std::mt19937 rng(12345); std::uniform_real_distribution<double> U(0.0,1.0);
    { std::vector<double> tp(nPat*s);
      for(int t=0;t<nTaxa;++t){ for(int k=0;k<nPat*s;++k) tp[k]=U(rng); beagleSetTipPartials(inst,t,tp.data()); } }

    // generalized-JC generator Q (per category, scaled by rate) for the derivative matrix
    if(dograd){
        std::vector<double> Q(nCat*s*s,0.0), QT(nCat*s*s,0.0);
        for(int c=0;c<nCat;++c) for(int i=0;i<s;++i) for(int j=0;j<s;++j){
            double q=(i==j)? -1.0 : 1.0/(s-1); Q[c*s*s+i*s+j]=q*rates[c]; }
        for(int c=0;c<nCat;++c) for(int i=0;i<s;++i) for(int j=0;j<s;++j) QT[c*s*s+j*s+i]=Q[c*s*s+i*s+j];
        beagleSetTransitionMatrix(inst,DQ ,Q.data(),0.0);
        beagleSetTransitionMatrix(inst,DQT,QT.data(),0.0);
    }

    // topology: edges indexed by child node; matrix buffer of node v = v.
    std::vector<int> edgeNodes; std::vector<double> edgeLen0;
    for(int c=0;c<nNodes;++c){ if(c==rootIdx) continue; edgeNodes.push_back(c); edgeLen0.push_back(0.1); }
    std::vector<double> edgeLen = edgeLen0;

    // post-order ops
    std::vector<BeagleOperation> ops;
    for(int k=0;k<nInternal;++k){
        int dest=nTaxa+k, c1=(k==0)?0:(nTaxa+k-1), c2=(k==0)?1:(k+1);
        BeagleOperation op={dest,BEAGLE_OP_NONE,BEAGLE_OP_NONE,c1,c1,c2,c2}; ops.push_back(op);
    }

    int wIdx=0,fIdx=0,cumScale=BEAGLE_OP_NONE; double logL=0.0;

    auto upd_mat = [&](){ beagleUpdateTransitionMatrices(inst,0,edgeNodes.data(),NULL,NULL,edgeLen.data(),(int)edgeNodes.size()); };
    auto lnl_only = [&](){ beagleUpdatePartials(inst,ops.data(),(int)ops.size(),BEAGLE_OP_NONE);
                           beagleCalculateRootLogLikelihoods(inst,&rootIdx,&wIdx,&fIdx,&cumScale,1,&logL); };
    auto full_eval = [&](){ upd_mat(); lnl_only(); };

    // ---------- (A) lnL timing ----------
    upd_mat(); for(int w=0;w<warmup;++w) lnl_only();
    fprintf(stdout,"lnL = %.10f\n",logL);
    double tFull=0,tLnl=0;
    for(int r=0;r<warmup;++r) full_eval();
    { auto t0=clk::now(); for(int r=0;r<reps;++r) full_eval(); tFull=ms_since(t0)/reps; }
    upd_mat(); for(int r=0;r<warmup;++r) lnl_only();
    { auto t0=clk::now(); for(int r=0;r<reps;++r) lnl_only(); tLnl=ms_since(t0)/reps; }

    // ---------- (B) gradient: pre-order sweep + all-branch edge derivatives ----------
    double tGrad=0, tGradPre=0; int gradOK=-1; double fd_worst=-1;
    if(dograd){
        // transposed forward matrices: transpose node v's matrix into MT+v
        std::vector<int> tin(edgeNodes), tout; for(int v:edgeNodes) tout.push_back(MT+v);
        // pre-order operations: {dest=PRE+v, sW,sR, partials1=PRE+parent, matrices1=MT+v, partials2=post(sibling), matrices2=sibling}
        // parent/sibling for the caterpillar:
        auto parent_of=[&](int v)->int{
            if(v==0||v==1) return nTaxa;                 // tips 0,1 -> node N
            if(v>=2 && v<nTaxa) return nTaxa+(v-1);      // tip t>=2 -> node N+(t-1)
            if(v>=nTaxa && v<rootIdx) return v+1;        // internal chain
            return -1; };
        auto sibling_of=[&](int v)->int{
            int p=parent_of(v);
            // children of internal node p=N+k: k==0 -> (0,1); k>=1 -> (N+k-1, k+1)
            int k=p-nTaxa, ca,cb; if(k==0){ca=0;cb=1;} else {ca=nTaxa+k-1; cb=k+1;}
            return (v==ca)? cb : ca; };
        // build ops in top-down order: parents before children. Process internal chain root..N, emitting child ops.
        std::vector<BeagleOperation> preOps;
        // root's pre-partial is seeded (freqs); for each internal node p from root down, emit ops for its two children.
        for(int p=rootIdx; p>=nTaxa; --p){
            int k=p-nTaxa, ca,cb; if(k==0){ca=0;cb=1;} else {ca=nTaxa+k-1; cb=k+1;}
            for(int v: {ca,cb}){
                int sib=(v==ca)?cb:ca;
                BeagleOperation op={ PRE+v, BEAGLE_OP_NONE, BEAGLE_OP_NONE, PRE+p, MT+v, sib, sib };
                preOps.push_back(op);
            }
        }
        // seed root pre-order partial = stationary freqs over all (cat,pattern)
        std::vector<double> rootPre((size_t)nCat*nPat*s);
        { size_t o=0; for(int c=0;c<nCat;++c) for(int p=0;p<nPat;++p) for(int st=0;st<s;++st) rootPre[o++]=freqs[st]; }

        std::vector<int> postIdx, preIdx, dMat, wList;
        for(int v:edgeNodes){ postIdx.push_back(v); preIdx.push_back(PRE+v); dMat.push_back(DQ); wList.push_back(0); }
        int nE=(int)edgeNodes.size();
        std::vector<double> sumDeriv(nE,0.0);

        auto grad_eval = [&](){
            beagleUpdatePartials(inst,ops.data(),(int)ops.size(),BEAGLE_OP_NONE);        // post-order (current partials)
            beagleTransposeTransitionMatrices(inst,tin.data(),tout.data(),nE);
            beagleSetPartials(inst,PRE+rootIdx,rootPre.data());
            beagleUpdatePrePartials(inst,preOps.data(),(int)preOps.size(),BEAGLE_OP_NONE);
            beagleCalculateEdgeDerivatives(inst,postIdx.data(),preIdx.data(),dMat.data(),wList.data(),nE,
                                           NULL,sumDeriv.data(),NULL);
        };
        auto grad_pre_only = [&](){ // pre-order sweep + derivatives, matrices+post cached
            beagleUpdatePrePartials(inst,preOps.data(),(int)preOps.size(),BEAGLE_OP_NONE);
            beagleCalculateEdgeDerivatives(inst,postIdx.data(),preIdx.data(),dMat.data(),wList.data(),nE,
                                           NULL,sumDeriv.data(),NULL);
        };

        upd_mat(); grad_eval();   // warm + populate sumDeriv
        // ---- FD self-check on a few edges: d(lnL)/d t_e vs (lnL(t+h)-lnL(t-h))/2h ----
        auto lnL_at = [&](std::vector<double>& el)->double{
            edgeLen=el; upd_mat(); beagleUpdatePartials(inst,ops.data(),(int)ops.size(),BEAGLE_OP_NONE);
            double L; beagleCalculateRootLogLikelihoods(inst,&rootIdx,&wIdx,&fIdx,&cumScale,1,&L); return L; };
        edgeLen=edgeLen0; upd_mat(); grad_eval();              // analytic gradient at base point
        std::vector<double> base=edgeLen0; double h=1e-5; fd_worst=0;
        int checkN=std::min(nE,6);
        for(int e=0;e<checkN;++e){
            std::vector<double> ep=base, em=base; ep[e]+=h; em[e]-=h;
            double Lp=lnL_at(ep), Lm=lnL_at(em); double fd=(Lp-Lm)/(2*h);
            double an=sumDeriv[e]; double rel=std::fabs(fd-an)/std::max(1.0,std::fabs(fd));
            fprintf(stdout,"  FD-check edge %d (node %d): analytic=%.6f  FD=%.6f  rel=%.3e\n",e,edgeNodes[e],an,fd,rel);
            fd_worst=std::max(fd_worst,rel);
        }
        gradOK = (fd_worst < 1e-4)?1:0;
        fprintf(stdout,"  gradient FD self-check: worst rel=%.3e -> %s\n",fd_worst,gradOK?"PASS":"FAIL");
        edgeLen=edgeLen0; upd_mat();

        // timing
        for(int r=0;r<warmup;++r) grad_eval();
        { auto t0=clk::now(); for(int r=0;r<reps;++r) grad_eval(); tGrad=ms_since(t0)/reps; }
        beagleUpdatePartials(inst,ops.data(),(int)ops.size(),BEAGLE_OP_NONE);
        beagleTransposeTransitionMatrices(inst,tin.data(),tout.data(),nE);
        beagleSetPartials(inst,PRE+rootIdx,rootPre.data());
        for(int r=0;r<warmup;++r) grad_pre_only();
        { auto t0=clk::now(); for(int r=0;r<reps;++r) grad_pre_only(); tGradPre=ms_since(t0)/reps; }
    }

    fprintf(stdout,"RESULT mode=%s tensor=%d resource=%d  lnL=%.10f  full_eval_ms=%.4f  lnL_only_ms=%.4f"
                   "  grad_full_ms=%.4f  grad_preonly_ms=%.4f  gradFD=%d  fd_worst=%.3e\n",
            mode, gotTensor?1:0, det.resourceNumber, logL, tFull, tLnl, tGrad, tGradPre, gradOK, fd_worst);

    fflush(stdout);
    _exit(0); // BEAGLE benchmarking atexit dump double-frees on this build; measurement already printed.
}
