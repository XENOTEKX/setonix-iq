// ============================================================================
// tc_decider.cu  —  Part 12 / T.0 KILL-SWITCH DECIDER (standalone, nvcc).
//
// Question it answers (the gate for the whole tensor-core lever):
//   Is an FP64 tensor-core (DMMA) matvec for the 20->32-padded  out = M . in
//   matvec  >= ~1.3x faster than JOLT's CURRENT register-resident scalar matvec,
//   at matched dims N = {100K, 1M}, on A100 (sm_80) / H200 (sm_90), with the
//   result matching the scalar oracle to rel <= 1e-12 ?
//
// The primitive measured is EXACTLY the inner matvec of JOLT's accum_child /
// kj_pre (gpu_lnl_intree.cu:50-51, 81-83):
//      out[c][x][ptn] = sum_i  M[c][x][i] * in[c][i][ptn]
// laid out state-major / pattern-minor:  idx = (c*NSP + s)*nptn + ptn.
// Only the SUMMATION ORDER differs between the two kernels, so the rel error is
// a pure measure of FP64-DMMA-vs-scalar accumulation order (expected ~1e-15..1e-13;
// NOT bit-identical -- that is the documented parity concession, part12 sec XII.1).
//
// DMMA path uses nvcuda::wmma double fragments (m8,n8,k4) -- the documented T.0
// "prototype with wmma" path (T.1 would move to raw mma.sync PTX for fusion).
// Padding 20->32 follows BEAGLE's kPaddedStateCount; padded states/cols are zeroed
// so both kernels compute the identical math.
//
// Build:  nvcc -O3 -std=c++14 -gencode arch=compute_80,code=sm_80 \
//                            -gencode arch=compute_90,code=sm_90 \
//              tc_decider.cu -o tc_decider
// Run:    ./tc_decider [nptn] [ncat] [reps] [warmup]      (defaults 100000 4 50 5)
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

#define NS   20            // real amino-acid state count
#define NSP  32            // padded state count (BEAGLE kPaddedStateCount for <=32)

#define GCK(call) do{ cudaError_t e=(call); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s @ %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__); exit(1);} }while(0)

// Scalar path reads matrices from __constant__ (mirrors JOLT's g_U/g_Uinv).
// DMMA path reads from GLOBAL (dMat) and stages to shared (mirrors BEAGLE, and
// honestly includes the shared-staging cost part12 sec XII.1 flags). Both hold
// the identical matrices. NCAT*NSP*NSP*8 bytes (<= 64KB constant for ncat<=8).
#define MAXCAT 8
__constant__ double cM[MAXCAT*NSP*NSP];

// --------------------------------------------------------------------------
// (1) SCALAR baseline -- faithful copy of accum_child's inner matvec.
//     one thread per pattern; loops cat inside (as k1_node does); registers.
// --------------------------------------------------------------------------
__global__ void scalar_matvec(int nptn, int ncat,
        const double* __restrict__ in, double* __restrict__ out) {
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    for (int c=0;c<ncat;c++){
        const double* M  = cM + (size_t)c*NSP*NSP;
        const double* pc = in + (size_t)(c*NSP)*nptn + ptn;
        double* o        = out + (size_t)(c*NSP)*nptn + ptn;
        for (int x=0;x<NS;x++){ double v=0.0;
            for (int i=0;i<NS;i++) v += M[x*NSP+i]*pc[(size_t)i*nptn];
            o[(size_t)x*nptn]=v; }
    }
}

// --------------------------------------------------------------------------
// (2) DMMA path -- wmma double m8n8k4. C[NSP x Ntile] = M[NSP x NSP] . B[NSP x Ntile].
//     Each warp owns 8 patterns (N-tile=8); loops the 4 M-row-tiles x 8 K-tiles.
//     Mapping onto the EXISTING state-major/pattern-minor layout (no repack):
//       matrix_a (MxK = out-state x in-state): M[c] row-major, ld=NSP        (row_major)
//       matrix_b (KxN = in-state  x pattern ): in[c], state-stride=nptn, ptn-stride=1
//                 -> row=K(state), col=N(ptn) is row_major with ld=nptn
//       acc      (MxN = out-state x pattern ): out[c], ld=nptn               (mem_row_major)
//     Padded states 20..31 are zero in both M and in, so the sum == scalar's.
// --------------------------------------------------------------------------
#if __CUDA_ARCH__ >= 800
__global__ void dmma_matvec(int nptn, int ncat,
        const double* __restrict__ Mg, const double* __restrict__ in, double* __restrict__ out) {
    const int WARP = 32;
    int c = blockIdx.y;                       // one category per grid.y
    // stage this category's NSP x NSP matrix into shared (BEAGLE-style; real DMMA cost)
    __shared__ double sM[NSP*NSP];
    for (int t=threadIdx.x; t<NSP*NSP; t+=blockDim.x) sM[t] = Mg[(size_t)c*NSP*NSP + t];
    __syncthreads();

    int warpId   = (blockIdx.x*blockDim.x + threadIdx.x) / WARP;
    int patBase  = warpId*8;                 // this warp's 8-pattern tile
    if (patBase + 8 > nptn) return;          // nptn assumed multiple of 8
    const double* B  = in + (size_t)(c*NSP)*nptn + patBase;
    double*       C  = out+ (size_t)(c*NSP)*nptn + patBase;

    for (int mt=0; mt<NSP/8; ++mt){           // 4 output-state row tiles
        wmma::fragment<wmma::accumulator,8,8,4,double> acc;
        wmma::fill_fragment(acc, 0.0);
        for (int kt=0; kt<NSP/4; ++kt){       // 8 contraction tiles (K=4 each)
            wmma::fragment<wmma::matrix_a,8,8,4,double,wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b,8,8,4,double,wmma::row_major> fb;
            wmma::load_matrix_sync(fa, sM + (size_t)(mt*8)*NSP + (kt*4), NSP);   // shared
            wmma::load_matrix_sync(fb, B  + (size_t)(kt*4)*nptn,        nptn);   // global, ld=nptn
            wmma::mma_sync(acc, fa, fb, acc);
        }
        wmma::store_matrix_sync(C + (size_t)(mt*8)*nptn, acc, nptn, wmma::mem_row_major);
    }
}
#else
__global__ void dmma_matvec(int,int,const double*,const double*,double*){ /* no FP64 TC < sm_80 */ }
#endif

// --------------------------------------------------------------------------
static double now_ms(cudaEvent_t a, cudaEvent_t b){ float m; cudaEventElapsedTime(&m,a,b); return m; }

int main(int argc, char** argv){
    int nptn  = argc>1 ? atoi(argv[1]) : 100000;
    int ncat  = argc>2 ? atoi(argv[2]) : 4;
    int reps  = argc>3 ? atoi(argv[3]) : 50;
    int warm  = argc>4 ? atoi(argv[4]) : 5;
    if (ncat>MAXCAT){ fprintf(stderr,"ncat<=%d\n",MAXCAT); return 1; }
    if (nptn%8){ fprintf(stderr,"nptn must be a multiple of 8 (wmma N-tile)\n"); return 1; }

    cudaDeviceProp prop; GCK(cudaGetDeviceProperties(&prop,0));
    printf("======== tc_decider  GPU=%s sm_%d%d  nptn=%d ncat=%d reps=%d ========\n",
           prop.name, prop.major, prop.minor, nptn, ncat, reps);
    if (prop.major < 8){ printf("  [SKIP] sm_%d%d has no FP64 tensor cores; scalar-only.\n",prop.major,prop.minor); }

    size_t nelem = (size_t)ncat*NSP*nptn;
    // host buffers (padded states zeroed)
    double* hM  = (double*)calloc((size_t)ncat*NSP*NSP, sizeof(double));
    double* hIn = (double*)calloc(nelem, sizeof(double));
    srand(12345);
    for (int c=0;c<ncat;c++){
        for (int x=0;x<NS;x++) for (int i=0;i<NS;i++)            // real 20x20, moderate magnitude
            hM[(size_t)c*NSP*NSP + x*NSP+i] = (rand()/(double)RAND_MAX - 0.5)*0.5;
        for (int i=0;i<NS;i++) for (int p=0;p<nptn;p++)          // positive partials in [0,1)
            hIn[(size_t)(c*NSP+i)*nptn + p] = rand()/(double)RAND_MAX;
    }
    double *dIn,*dOutS,*dOutT,*dMat;
    GCK(cudaMalloc(&dIn,  nelem*sizeof(double)));
    GCK(cudaMalloc(&dOutS,nelem*sizeof(double)));
    GCK(cudaMalloc(&dOutT,nelem*sizeof(double)));
    GCK(cudaMalloc(&dMat, (size_t)ncat*NSP*NSP*sizeof(double)));
    GCK(cudaMemcpy(dIn, hIn, nelem*sizeof(double), cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(dMat, hM, (size_t)ncat*NSP*NSP*sizeof(double), cudaMemcpyHostToDevice));
    GCK(cudaMemset(dOutS,0,nelem*sizeof(double)));
    GCK(cudaMemset(dOutT,0,nelem*sizeof(double)));
    GCK(cudaMemcpyToSymbol(cM, hM, (size_t)ncat*NSP*NSP*sizeof(double)));

    cudaEvent_t e0,e1; GCK(cudaEventCreate(&e0)); GCK(cudaEventCreate(&e1));

    // ---- scalar ----
    int tpb=128, grid=(nptn+tpb-1)/tpb;
    for (int r=0;r<warm;r++) scalar_matvec<<<grid,tpb>>>(nptn,ncat,dIn,dOutS);
    GCK(cudaDeviceSynchronize());
    double sMin=1e30,sSum=0;
    for (int r=0;r<reps;r++){
        GCK(cudaEventRecord(e0));
        scalar_matvec<<<grid,tpb>>>(nptn,ncat,dIn,dOutS);
        GCK(cudaEventRecord(e1)); GCK(cudaEventSynchronize(e1));
        double m=now_ms(e0,e1); sSum+=m; if(m<sMin)sMin=m;
    }
    GCK(cudaGetLastError());

    // ---- dmma ----
    double tMin=1e30,tSum=0; bool ranTC=false;
    if (prop.major>=8){
        ranTC=true;
        int warpsPerBlock=8, tpb2=warpsPerBlock*32;
        dim3 g2((nptn/8 + warpsPerBlock-1)/warpsPerBlock, ncat);   // x: 8-ptn tiles, y: category
        for (int r=0;r<warm;r++) dmma_matvec<<<g2,tpb2>>>(nptn,ncat,dMat,dIn,dOutT);
        GCK(cudaDeviceSynchronize());
        for (int r=0;r<reps;r++){
            GCK(cudaEventRecord(e0));
            dmma_matvec<<<g2,tpb2>>>(nptn,ncat,dMat,dIn,dOutT);
            GCK(cudaEventRecord(e1)); GCK(cudaEventSynchronize(e1));
            double m=now_ms(e0,e1); tSum+=m; if(m<tMin)tMin=m;
        }
        GCK(cudaGetLastError());
    }

    // ---- parity (scalar oracle vs dmma), real states only ----
    double maxrel=0.0; size_t worst=0;
    if (ranTC){
        double* hS=(double*)malloc(nelem*sizeof(double));
        double* hT=(double*)malloc(nelem*sizeof(double));
        GCK(cudaMemcpy(hS,dOutS,nelem*sizeof(double),cudaMemcpyDeviceToHost));
        GCK(cudaMemcpy(hT,dOutT,nelem*sizeof(double),cudaMemcpyDeviceToHost));
        for (int c=0;c<ncat;c++) for (int x=0;x<NS;x++) for (int p=0;p<nptn;p++){
            size_t idx=(size_t)(c*NSP+x)*nptn+p;
            double a=hS[idx], b=hT[idx];
            double rel=fabs(a-b)/(fabs(a)+1e-300);
            if (rel>maxrel){maxrel=rel; worst=idx;}
        }
        free(hS); free(hT);
    }

    // ---- report ----
    printf("  scalar matvec : min %.4f ms  mean %.4f ms\n", sMin, sSum/reps);
    if (ranTC){
        printf("  dmma   matvec : min %.4f ms  mean %.4f ms\n", tMin, tSum/reps);
        printf("  SPEEDUP (scalar/dmma) : min %.3fx  mean %.3fx\n", sMin/tMin, (sSum/reps)/(tSum/reps));
        printf("  PARITY max rel = %.3e (worst idx %zu)  -> %s (gate 1e-12)\n",
               maxrel, worst, (maxrel<=1e-12?"PASS":"FAIL"));
        double sp = sMin/tMin;
        printf("  T.0 VERDICT @ nptn=%d : %s\n", nptn,
               (maxrel<=1e-12 && sp>=1.3) ? "DMMA >= 1.3x AND parity OK -> candidate GO"
             : (maxrel> 1e-12)            ? "PARITY FAIL -> investigate before any GO"
             :                              "DMMA < 1.3x -> scalar already wins this dim");
    } else {
        printf("  (no FP64 tensor cores on this GPU -- run on A100/H200 for the DMMA arm)\n");
    }
    printf("======== done ========\n");
    free(hM); free(hIn); cudaFree(dIn); cudaFree(dOutS); cudaFree(dOutT);
    return 0;
}
