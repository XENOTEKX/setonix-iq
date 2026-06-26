// L2.0 DECISION SPIKE — does grid.z=K (K resident trees) amortize per-launch/per-sync
// latency on the REAL IQ-TREE GPU kernels at AA-100K shape, enough to make L2 worth 10-14 weeks?
//
// Decisive outputs:
//   (1) MEASURED occupancy: cudaOccupancyMaxActiveBlocksPerMultiprocessor for the real k1_node /
//       kj_derv_fused -> blocks/SM, device block capacity, and K-to-fill = capacity / GB(AA-100K).
//       (Settles red-team Blocker 1's register estimate with a real number.)
//   (2) Throughput: time K trees' representative postorder+gradient sweep via
//       (a) K-serial (grid.z=1, per-step sync — today's cadence), (b) grid.z=K batched,
//       (c) K CUDA streams. speedup = serial/variant. GO iff >= 3x at K=8.
//
// Kernels k1_node / accum_child / kj_derv_fused are COPIED VERBATIM from tree/gpu/gpu_lnl_intree.cu
// (so register pressure / memory pattern / occupancy are faithful). The _z variants are identical
// math with a blockIdx.z buffer offset (what L2 grid.z=K would launch). Timing only — numerics are
// already banked exact, so synthetic finite data is fine.
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#define NS_MAX 20
#define GCK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

__constant__ double g_Uinv[NS_MAX*NS_MAX];
__constant__ double g_U[NS_MAX*NS_MAX];
__constant__ double g_UinvRowSum[NS_MAX];
__constant__ double g_freq[NS_MAX];
__constant__ double g_catw[64];
__constant__ double g_val0[64*NS_MAX];
__constant__ double g_val1[64*NS_MAX];
__constant__ double g_val2[64*NS_MAX];
__constant__ double g_rscale[64];

// ---- VERBATIM from gpu_lnl_intree.cu:46 ----
__device__ __forceinline__ void accum_child(double* prod, int ns, int c, int ptn, int nptn,
        const double* __restrict__ ec, const double* __restrict__ p, const unsigned char* __restrict__ t) {
    const double* ecc = ec + (size_t)c*ns*ns;
    if (p) {
        const double* pc = p + (size_t)(c*ns)*nptn + ptn;
        for (int x=0;x<ns;x++){ double v=0.0;
            for (int i=0;i<ns;i++) v += ecc[x*ns+i]*pc[(size_t)i*nptn];
            prod[x]*=v; }
    } else {
        int s = t[ptn];
        for (int x=0;x<ns;x++){ double v=0.0;
            for (int i=0;i<ns;i++){ double Li = (s<ns)? g_Uinv[i*ns+s] : g_UinvRowSum[i]; v += ecc[x*ns+i]*Li; }
            prod[x]*=v; }
    }
}
// ---- VERBATIM from gpu_lnl_intree.cu:63 ----
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
// ---- VERBATIM from gpu_lnl_intree.cu:329 ----
__global__ void kj_derv_fused(int ns, int nptn, int ncat,
        const double* __restrict__ node, const double* __restrict__ dad,
        double pinv, const double* __restrict__ baseinvar,
        double* __restrict__ patlh, double* __restrict__ pdf, double* __restrict__ pddf,
        double* __restrict__ rnum, double* __restrict__ wnum){
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    double lh=0.0,d1=0.0,d2=0.0;
    for (int c=0;c<ncat;c++){
        double rc=0.0, lcc=0.0;
        for (int x=0;x<ns;x++){ int k=c*ns+x; size_t o=(size_t)k*nptn+ptn;
            double th=node[o]*dad[o];
            lcc+=g_val0[k]*th; rc+=g_val1[k]*th; d2+=g_val2[k]*th; }
        lh+=lcc; d1+=rc;
        if (rnum) rnum[(size_t)c*nptn+ptn]+=g_rscale[c]*rc;
        if (wnum) wnum[(size_t)c*nptn+ptn]=lcc;
    }
    double Lp=fabs(lh)+pinv*baseinvar[ptn]; double inv=1.0/Lp, r=d1*inv;
    patlh[ptn]=log(Lp); pdf[ptn]=r; pddf[ptn]=d2*inv-r*r;
}
// ---- _z BATCHED variants: identical math, blockIdx.z buffer offset (what L2 grid.z=K launches) ----
__global__ void k1_node_z(int ns,int nptn,int ncat,int isRoot,double* outb,double* patlhb,int nchild,
        const double* ec0,const double* p0b,const unsigned char* t0,
        const double* ec1,const double* p1b,const unsigned char* t1,
        const double* ec2,const double* p2b,const unsigned char* t2){
    int ptn=blockIdx.x*blockDim.x+threadIdx.x; if(ptn>=nptn) return;
    size_t slot=(size_t)ncat*ns*nptn; int z=blockIdx.z;
    double* out=outb+(size_t)z*slot; double* patlh=patlhb+(size_t)z*nptn;
    const double* p0=p0b?p0b+(size_t)z*slot:nullptr;
    const double* p1=p1b?p1b+(size_t)z*slot:nullptr;
    const double* p2=p2b?p2b+(size_t)z*slot:nullptr;
    double lh=0.0;
    for(int c=0;c<ncat;c++){
        double prod[NS_MAX]; for(int x=0;x<ns;x++) prod[x]=1.0;
        accum_child(prod,ns,c,ptn,nptn,ec0,p0,t0);
        if(nchild>1) accum_child(prod,ns,c,ptn,nptn,ec1,p1,t1);
        if(nchild>2) accum_child(prod,ns,c,ptn,nptn,ec2,p2,t2);
        if(isRoot){ double s=0.0; for(int x=0;x<ns;x++) s+=g_freq[x]*prod[x]; lh+=g_catw[c]*s; }
        else { double* o=out+(size_t)(c*ns)*nptn+ptn; for(int r=0;r<ns;r++){ double v=0.0;
            for(int x=0;x<ns;x++) v+=g_Uinv[r*ns+x]*prod[x]; o[(size_t)r*nptn]=v; } }
    }
    if(isRoot) patlh[ptn]=log(fabs(lh));
}
__global__ void kj_derv_fused_z(int ns,int nptn,int ncat,
        const double* nodeb,const double* dadb,double pinv,const double* baseinvar,
        double* patlhb,double* pdfb,double* pddfb,double* rnumb,double* wnumb){
    int ptn=blockIdx.x*blockDim.x+threadIdx.x; if(ptn>=nptn) return;
    size_t slot=(size_t)ncat*ns*nptn; int z=blockIdx.z;
    const double* node=nodeb+(size_t)z*slot; const double* dad=dadb+(size_t)z*slot;
    double* patlh=patlhb+(size_t)z*nptn; double* pdf=pdfb+(size_t)z*nptn; double* pddf=pddfb+(size_t)z*nptn;
    double* rnum=rnumb?rnumb+(size_t)z*ncat*nptn:nullptr;
    double lh=0.0,d1=0.0,d2=0.0;
    for(int c=0;c<ncat;c++){
        double rc=0.0,lcc=0.0;
        for(int x=0;x<ns;x++){ int k=c*ns+x; size_t o=(size_t)k*nptn+ptn;
            double th=node[o]*dad[o]; lcc+=g_val0[k]*th; rc+=g_val1[k]*th; d2+=g_val2[k]*th; }
        lh+=lcc; d1+=rc;
        if(rnum) rnum[(size_t)c*nptn+ptn]+=g_rscale[c]*rc;
    }
    double Lp=fabs(lh)+pinv*baseinvar[ptn]; double inv=1.0/Lp,r=d1*inv;
    patlh[ptn]=log(Lp); pdf[ptn]=r; pddf[ptn]=d2*inv-r*r;
}
__global__ void fill_k(double* p, size_t n, double v){ size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) p[i]=v; }

int main(int argc,char**argv){
    const int ns=20, ncat=4, nptn=100000, TB=256;
    const int L = (argc>1)?atoi(argv[1]):128;   // representative node-steps per sweep
    const int R = (argc>2)?atoi(argv[2]):5;     // timed sweeps (averaged)
    const int Kmax=12;
    const int GB=(nptn+TB-1)/TB;
    const size_t slot=(size_t)ncat*ns*nptn;     // 64 MB / tree
    cudaDeviceProp pr; GCK(cudaGetDeviceProperties(&pr,0));
    int nSM=pr.multiProcessorCount;
    printf("=== L2.0 SPIKE on %s : %d SMs, %.0f GB ===\n",pr.name,nSM,pr.totalGlobalMem/1e9);
    printf("AA-100K shape: ns=%d ncat=%d nptn=%d  TB=%d  GB=ceil(nptn/TB)=%d blocks/launch  L=%d steps R=%d reps\n",
           ns,ncat,nptn,TB,GB,L,R);

    // (1) MEASURED occupancy on the real kernels -> K-to-fill
    int occ_k1=0, occ_dv=0;
    GCK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_k1,k1_node,TB,0));
    GCK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_dv,kj_derv_fused,TB,0));
    cudaFuncAttributes a1,a2; cudaFuncGetAttributes(&a1,k1_node); cudaFuncGetAttributes(&a2,kj_derv_fused);
    printf("\n-- MEASURED occupancy --\n");
    printf("k1_node       : %d regs/thread, %d blocks/SM -> capacity %d blocks ; K-to-fill = %.2f\n",
           a1.numRegs,occ_k1,occ_k1*nSM,(double)(occ_k1*nSM)/GB);
    printf("kj_derv_fused : %d regs/thread, %d blocks/SM -> capacity %d blocks ; K-to-fill = %.2f\n",
           a2.numRegs,occ_dv,occ_dv*nSM,(double)(occ_dv*nSM)/GB);
    printf("(K-to-fill = device block capacity / one tree's 391 blocks. If <2, one tree already fills the GPU\n"
           " and grid.z=K just time-slices => batching can only recover per-launch overhead.)\n");

    // constants (arbitrary finite)
    std::vector<double> h(64*NS_MAX,0.03);
    GCK(cudaMemcpyToSymbol(g_Uinv,h.data(),sizeof(double)*ns*ns));
    GCK(cudaMemcpyToSymbol(g_U,h.data(),sizeof(double)*ns*ns));
    GCK(cudaMemcpyToSymbol(g_UinvRowSum,h.data(),sizeof(double)*ns));
    GCK(cudaMemcpyToSymbol(g_freq,h.data(),sizeof(double)*ns));
    GCK(cudaMemcpyToSymbol(g_catw,h.data(),sizeof(double)*ncat));
    GCK(cudaMemcpyToSymbol(g_val0,h.data(),sizeof(double)*ncat*ns));
    GCK(cudaMemcpyToSymbol(g_val1,h.data(),sizeof(double)*ncat*ns));
    GCK(cudaMemcpyToSymbol(g_val2,h.data(),sizeof(double)*ncat*ns));
    GCK(cudaMemcpyToSymbol(g_rscale,h.data(),sizeof(double)*ncat));

    // per-tree buffers (Kmax slices), reused: A,B,C children/inputs ; D out
    double *dA,*dB,*dC,*dD,*dpatlh,*dpdf,*dpddf,*drnum,*dbi,*dec;
    GCK(cudaMalloc(&dA,Kmax*slot*sizeof(double)));
    GCK(cudaMalloc(&dB,Kmax*slot*sizeof(double)));
    GCK(cudaMalloc(&dC,Kmax*slot*sizeof(double)));
    GCK(cudaMalloc(&dD,Kmax*slot*sizeof(double)));
    GCK(cudaMalloc(&dpatlh,(size_t)Kmax*nptn*sizeof(double)));
    GCK(cudaMalloc(&dpdf,(size_t)Kmax*nptn*sizeof(double)));
    GCK(cudaMalloc(&dpddf,(size_t)Kmax*nptn*sizeof(double)));
    GCK(cudaMalloc(&drnum,(size_t)Kmax*ncat*nptn*sizeof(double)));
    GCK(cudaMalloc(&dbi,(size_t)nptn*sizeof(double)));
    GCK(cudaMalloc(&dec,(size_t)ncat*ns*ns*sizeof(double)));
    auto fill=[&](double*p,size_t n,double v){ fill_k<<<(n+255)/256,256>>>(p,n,v); };
    fill(dA,Kmax*slot,0.5); fill(dB,Kmax*slot,0.5); fill(dC,Kmax*slot,0.5); fill(dD,Kmax*slot,0.5);
    fill(dbi,nptn,0.1); fill(dec,(size_t)ncat*ns*ns,0.05); fill(drnum,(size_t)Kmax*ncat*nptn,0.0);
    GCK(cudaDeviceSynchronize());

    cudaStream_t st[Kmax]; for(int k=0;k<Kmax;k++) GCK(cudaStreamCreate(&st[k]));
    const unsigned char* T=nullptr; // internal-node path (p!=null) => tips unused
    auto sweep_serial=[&](int K){
        for(int k=0;k<K;k++){ double off=k*slot;
            const double* A=dA+(size_t)k*slot; const double* B=dB+(size_t)k*slot; const double* C=dC+(size_t)k*slot;
            double* Dout=dD+(size_t)k*slot; double* pl=dpatlh+(size_t)k*nptn; (void)off;
            for(int s=0;s<L;s++){
                k1_node<<<GB,TB>>>(ns,nptn,ncat,0,Dout,pl,3,dec,A,T,dec,B,T,dec,C,T);
                kj_derv_fused<<<GB,TB>>>(ns,nptn,ncat,A,B,0.0,dbi,pl,dpdf+(size_t)k*nptn,dpddf+(size_t)k*nptn,
                                         drnum+(size_t)k*ncat*nptn,nullptr);
                GCK(cudaDeviceSynchronize());   // today's per-step sync cadence
            }
        }
    };
    auto sweep_batched=[&](int K){
        dim3 g(GB,1,K);
        for(int s=0;s<L;s++){
            k1_node_z<<<g,TB>>>(ns,nptn,ncat,0,dD,dpatlh,3,dec,dA,T,dec,dB,T,dec,dC,T);
            kj_derv_fused_z<<<g,TB>>>(ns,nptn,ncat,dA,dB,0.0,dbi,dpatlh,dpdf,dpddf,drnum,nullptr);
            GCK(cudaDeviceSynchronize());       // ONE sync for K trees' step
        }
    };
    auto sweep_streams=[&](int K){
        for(int s=0;s<L;s++){
            for(int k=0;k<K;k++){
                const double* A=dA+(size_t)k*slot; const double* B=dB+(size_t)k*slot; const double* C=dC+(size_t)k*slot;
                double* Dout=dD+(size_t)k*slot; double* pl=dpatlh+(size_t)k*nptn;
                k1_node<<<GB,TB,0,st[k]>>>(ns,nptn,ncat,0,Dout,pl,3,dec,A,T,dec,B,T,dec,C,T);
                kj_derv_fused<<<GB,TB,0,st[k]>>>(ns,nptn,ncat,A,B,0.0,dbi,pl,dpdf+(size_t)k*nptn,dpddf+(size_t)k*nptn,
                                                 drnum+(size_t)k*ncat*nptn,nullptr);
            }
            GCK(cudaDeviceSynchronize());       // all K streams' step
        }
    };
    auto timeit=[&](auto fn,int K)->double{
        fn(K); GCK(cudaDeviceSynchronize());                 // warmup
        auto t0=std::chrono::high_resolution_clock::now();
        for(int r=0;r<R;r++) fn(K);
        GCK(cudaDeviceSynchronize());
        auto t1=std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double,std::milli>(t1-t0).count()/R;
    };

    printf("\n-- THROUGHPUT (ms per K-tree sweep; speedup = serial/variant) --\n");
    printf("%4s %12s %12s %12s %10s %10s\n","K","serial_ms","batched_ms","streams_ms","sp_batch","sp_strm");
    double base_batched1=0; double best8=0;
    for(int K : {1,2,4,8,12}){
        double ts=timeit(sweep_serial,K);
        double tb=timeit(sweep_batched,K);
        double tm=timeit(sweep_streams,K);
        if(K==1) base_batched1=tb;
        double spb=ts/tb, spm=ts/tm;
        printf("%4d %12.2f %12.2f %12.2f %9.2fx %9.2fx\n",K,ts,tb,tm,spb,spm);
        if(K==8) best8=fmax(spb,spm);
    }
    printf("\nbatched scaling: batched_ms(K)/batched_ms(1) ~ K => time-sliced (no concurrency win); ~1 => fully concurrent\n");
    printf("\n================ VERDICT ================\n");
    printf("Best K=8 throughput speedup (batched or streams) vs serial = %.2fx\n",best8);
    printf("DECISION GATE: GO to full L2 iff >= 3.00x.  -> %s\n", best8>=3.0?"GO":"NO-GO (pivot to GPU+MPI hybrid)");
    printf("Interpretation: the speedup ceiling ~ 1 + T_overhead/T_compute. >=3x means kernels are launch-bound\n"
           "(batching wins). ~1.1-1.5x means one tree already saturates the GPU (red-team Blocker 1 confirmed).\n");
    return 0;
}
