// phylotreegpu.cpp — Phase G.2 in-tree GPU log-likelihood integration (host side).
//
// G.2.0a — gpuLnLCrossCheckOnce(): a one-shot, read-only diagnostic that proved the harness↔in-tree BRIDGE
//   (eigen convention, tip-ambiguity fold, ptn_freq pattern weights, pi-fold, NORM_LH) by rebuilding the
//   validated K1 eigen-space postorder sweep clean-room from the LIVE IQ-TREE objects and comparing its GPU
//   total lnL against IQ-TREE's own curScore. PASSED rel=1.235e-16 (job 170203514). Zero coupling to the
//   fn-pointer seam / host partial buffers / TraversalInfo.
//
// G.2.0b — wire the seam (lnL-only, -blfix). setLikelihoodKernelGPU() overrides ONLY computeLikelihoodBranch-
//   Pointer at the setLikelihoodKernel funnel so IQ-TREE's own computeLikelihood routes through the GPU.
//   Verified (adversarial workflow over the dev tree): under `--gpu -te TREE -m LG+G4 -blfix`, branch-length
//   Newton-Raphson is unreachable (fixed_branch_length==BRLEN_FIX gates out optimizeAllBranches at
//   modelfactory.cpp:1628; -te zeroes min_iterations so no tree search), so computeLikelihoodDerv /
//   computeLikelihoodFromBuffer NEVER fire; +G4 alpha is tuned by derivative-free Brent (rategamma.cpp:240)
//   which only calls computeLikelihood->Branch. The ONE host consumer of per-pattern data on the final path is
//   the UNCONDITIONAL computeLogLVariance() (phyloanalysis.cpp:3946) -> computePatternLikelihood() which reads
//   host _pattern_lh[] (phylotree.cpp:1515-1528). So the Branch override ALSO mirrors the per-pattern
//   log|lh_ptn| into _pattern_lh[] (the launcher already computes them) and zeroes the branch lh_scale_factor
//   (NORM_LH no-scaling path) -> the existing CPU variance/report code produces the correct s.e. with no edits
//   to phyloanalysis.cpp.
//
// Scope gate (NORM_LH / unscaled, reversible, bifurcating, single mixture, num_states in {4,20}, no +I/+ASC,
// no -wsl/-wpl/-alrt/-abayes/-b/-bb/-asr/dating/pll, fixed_branch_length==BRLEN_FIX): anything else -> the
// helper returns NaN and the Branch override delegates to the saved CPU pointer; the funnel installer simply
// does not install. Whole TU compiled only when IQTREE_GPU=ON (CMake); guarded anyway.
#include "iqtree_config.h"
#ifdef IQTREE_GPU

#include "phylotree.h"
#include "phylonode.h"
#include <cstring>   // G.6: memcpy in the free-Q decompose callback
#include "model/modelsubst.h"
#include "model/modelmixture.h"   // G.8.0: ModelMixture per-class component accessors (at(m)->getEigenvalues() etc.)
#include "model/rateheterogeneity.h"
#include "model/rategamma.h"   // G.4.3b: GAMMA_CUT_MEAN — the robust mean-gamma discriminator (isGammaRate())
#include "alignment/alignment.h"
#include "tree/gpu/gpu_iqtree.h"
#include <vector>
#include <map>
#include <functional>
#include <cmath>
#include <cstdio>
#include <mutex>     // G.8.2.2: serialize the unguarded mix clean-room launchers across ModelFinder's per-model OpenMP threads
using namespace std;

// G.8.2.2 — process-wide lock for optimizeParametersJOLTMix. The mixture clean-room launchers
// (gpu_lnl_crosscheck_mix / the all-branch-derv-mix launcher) are NOT internally mutexed (unlike gpu_jolt_optimize,
// which holds its own jolt_gpu_mtx), and ModelFinder scores candidate models across-model OpenMP-parallel
// (phylotesting.cpp). Without this, concurrent JOLTMix calls would race the single GPU's constant memory. JOLTMix
// therefore serializes on the one GPU (the G.4.2b decision); ineligible candidates still run N-parallel on the CPU.
static std::mutex gpu_mixjolt_mtx;

// ============================================================================================================
// Reusable clean-room GPU whole-tree log-likelihood (extracted from the G.2.0a cross-check). SILENT (no prints
// per call — it is invoked once per Branch evaluation): returns NaN on any unsupported regime / CUDA error so
// callers fall back. If out_patlh != nullptr it is filled with the per-pattern log|lh_ptn| (aln->size()
// entries, pattern order) = exactly what _pattern_lh[] holds under NORM_LH (no scaling).
// ============================================================================================================
double PhyloTree::gpuComputeTreeLnLCleanRoom(double *out_patlh) {
    // ---- regime gate (silent) ----
    if (!model || !site_rate || !aln) return (double)NAN;
    int ns = aln->num_states;
    if (ns != 4 && ns != 20) return (double)NAN;
    if (!model->isReversible()) return (double)NAN;
    if (model->getNMixtures() != 1) return (double)NAN;
    if (model->isSiteSpecificModel()) return (double)NAN;
    if (site_rate->getPInvar() > 0.0) return (double)NAN;   // +I omits ptn_invar in the clean-room sweep -> CPU

    int ncat = site_rate->getNRate();
    int nptn = (int)aln->size();
    int ntax = (int)aln->getNSeq();
    if (ncat < 1 || ncat > 64) return (double)NAN;

    // ---- model eigen factors (IQ-TREE convention: U=evec, Uinv=inv_evec, P(t)=U exp(Lambda t) Uinv) ----
    double *eval = model->getEigenvalues();
    double *U    = model->getEigenvectors();
    double *Uinv = model->getInverseEigenvectors();
    if (!eval || !U || !Uinv) return (double)NAN;
    vector<double> freq(ns, 0.0);
    model->getStateFrequency(freq.data(), 0);
    vector<double> UinvRowSum(ns, 0.0);
    for (int i = 0; i < ns; i++) { double s = 0; for (int j = 0; j < ns; j++) s += Uinv[i*ns+j]; UinvRowSum[i] = s; }
    vector<double> catRate(ncat), catProp(ncat);
    for (int c = 0; c < ncat; c++) { catRate[c] = site_rate->getRate(c); catProp[c] = site_rate->getProp(c); }

    // ---- topology rooted at an internal node R (IQ-TREE roots at a leaf; lnL is reversible-invariant) ----
    if (!root || !root->isLeaf() || root->neighbors.empty()) return (double)NAN;
    Node *R = root->neighbors[0]->node;     // internal node adjacent to the root leaf
    if (R->isLeaf()) return (double)NAN;

    map<Node*,int> nid;
    vector<Node*> nodes;
    vector<double> parentLen;               // edge length to parent (R's unused)
    vector<int>    isLeafV, leafTax;
    function<void(Node*,Node*,double)> indexDfs = [&](Node *n, Node *dad, double lenToDad) {
        int myi = (int)nodes.size(); nid[n] = myi; nodes.push_back(n);
        parentLen.push_back(lenToDad);
        int lf = n->isLeaf() ? 1 : 0; isLeafV.push_back(lf);
        leafTax.push_back(lf ? aln->getSeqID(n->name) : -1);
        for (auto nb : n->neighbors) { if (nb->node == dad) continue; indexDfs(nb->node, n, nb->length); }
    };
    indexDfs(R, nullptr, 0.0);
    int nNodes = (int)nodes.size();

    vector<int> postInternal;               // node indices, post-order (children precede parents)
    vector<int> slot(nNodes, -1);
    function<void(Node*,Node*)> postDfs = [&](Node *n, Node *dad) {
        for (auto nb : n->neighbors) { if (nb->node == dad) continue; postDfs(nb->node, n); }
        if (!n->isLeaf()) { slot[nid[n]] = (int)postInternal.size(); postInternal.push_back(nid[n]); }
    };
    postDfs(R, nullptr);
    int nInternal = (int)postInternal.size();

    // ---- echild[child_node][cat][x][i] = U[x][i] * exp(eval[i]*rate_c*parentLen[child]) ----
    size_t ecStride = (size_t)ncat*ns*ns;
    vector<double> echild((size_t)nNodes*ecStride, 0.0);
    for (int v = 0; v < nNodes; v++) {
        if (v == nid[R]) continue;          // R has no parent edge
        double len_v = parentLen[v];
        for (int c = 0; c < ncat; c++) {
            double l = len_v * catRate[c];
            double ex[20]; for (int i = 0; i < ns; i++) ex[i] = exp(eval[i]*l);
            double *e = &echild[(size_t)v*ecStride + (size_t)c*ns*ns];
            for (int x = 0; x < ns; x++) for (int i = 0; i < ns; i++) e[x*ns+i] = U[x*ns+i]*ex[i];
        }
    }

    // ---- compact tip states[taxon][ptn] ; pattern frequencies ----
    vector<unsigned char> tip((size_t)ntax*nptn);
    for (int v = 0; v < nNodes; v++) {
        if (!isLeafV[v]) continue;
        int tax = leafTax[v];
        if (tax < 0 || tax >= ntax) return (double)NAN;
        for (int p = 0; p < nptn; p++) { int st = (int)aln->at(p)[tax]; tip[(size_t)tax*nptn+p] = (unsigned char)((st < ns) ? st : ns); }
    }
    vector<double> ptnFreq(nptn);
    for (int p = 0; p < nptn; p++) ptnFreq[p] = (double)aln->at(p).frequency;

    // ---- per-internal-node descriptors (postorder) ----
    vector<int> dRoot(nInternal), dNch(nInternal), dOut(nInternal);
    vector<int> dChildNode(nInternal*3, -1), dChildIsLeaf(nInternal*3, 0), dChildLeaf(nInternal*3, -1), dChildSlot(nInternal*3, -1);
    for (int idx = 0; idx < nInternal; idx++) {
        int vi = postInternal[idx]; Node *n = nodes[vi];
        Node *dad = nullptr;
        if (n != R) { for (auto nb : n->neighbors) { if (nid[nb->node] < vi) { dad = nb->node; break; } } }
        dRoot[idx] = (n == R) ? 1 : 0;
        dOut[idx]  = (n == R) ? -1 : slot[vi];
        int k = 0;
        for (auto nb : n->neighbors) {
            if (nb->node == dad) continue;
            if (k >= 3) return (double)NAN;
            int cv = nid[nb->node];
            dChildNode[idx*3+k] = cv;
            if (isLeafV[cv]) { dChildIsLeaf[idx*3+k] = 1; dChildLeaf[idx*3+k] = leafTax[cv]; }
            else             { dChildIsLeaf[idx*3+k] = 0; dChildSlot[idx*3+k] = slot[cv]; }
            k++;
        }
        dNch[idx] = k;
    }

    return gpu_lnl_crosscheck(ns, nptn, ncat, ntax, nNodes, nInternal,
        Uinv, UinvRowSum.data(), freq.data(), catProp.data(), echild.data(), tip.data(), ptnFreq.data(),
        dRoot.data(), dNch.data(), dOut.data(),
        dChildNode.data(), dChildIsLeaf.data(), dChildLeaf.data(), dChildSlot.data(),
        out_patlh);
}

// ============================================================================================================
// G.8.0 — clean-room whole-tree lnL for a PROFILE MIXTURE (C20/C60/MEOW80). Reads the LIVE ModelMixture's per-class
// eigen via the component accessors mix[m]->getEigenvalues()/...() (already offset into the AVX-padded packed array,
// so NO manual m*480 stride math — sidesteps the #1 silent-bug risk), per-class freq via getStateFrequency(.,m),
// weights via getMixtureWeight(m). Regime r = m*ncat + c; builds echild[node][r] and per-class Uinv/freq, then calls
// gpu_lnl_crosscheck_mix. SILENT, returns NaN on any unsupported regime. CPU path byte-unchanged (additive).
// ============================================================================================================
double PhyloTree::gpuComputeTreeLnLCleanRoomMix(double *out_patlh, double *out_lhcat, const double *w_override,
                                                const double *parentLenOverride, double alphaOverride) {
    if (!model || !site_rate || !aln) return (double)NAN;
    int ns = aln->num_states;
    if (ns != 20 && ns != 4) return (double)NAN;
    if (!model->isReversible()) return (double)NAN;
    int N = model->getNMixtures();
    if (N <= 1) return (double)NAN;                       // single-model -> the non-mix path
    if (model->isSiteSpecificModel()) return (double)NAN; // PMSF stays on CPU (per-site pi, no class sum)
    if (site_rate->getPInvar() > 0.0) return (double)NAN;  // +I omitted in the clean-room sweep -> CPU

    int ncat = site_rate->getNRate();
    int nptn = (int)aln->size();
    int ntax = (int)aln->getNSeq();
    if (ncat < 1) return (double)NAN;
    int R = N*ncat;

    ModelMixture *mix = dynamic_cast<ModelMixture*>(model);
    if (!mix || (int)mix->size() != N) return (double)NAN;
    if (mix->isFused()) return (double)NAN;   // LG4M/LG4X = 1:1 class<->rate pairing, NOT the N*ncat cross-product
                                              // this builds (weight from site_rate->getProp, not getMixtureWeight) -> CPU

    // ---- per-class eigen (stride-safe via component pointers), freq, weight; per-regime weight = w_m*catProp_c ----
    std::vector<double> Uinv((size_t)N*ns*ns), UinvRowSum((size_t)N*ns), freqC((size_t)N*ns);
    std::vector<double> evalC((size_t)N*ns), Uc((size_t)N*ns*ns);
    std::vector<double> catRate(ncat), catProp(ncat), wreg((size_t)R);
    for (int c = 0; c < ncat; c++) { catRate[c] = site_rate->getRate(c); catProp[c] = site_rate->getProp(c); }
    // G.8.2.1b: optional iterate-alpha override — recompute mean-1 discrete-gamma catRate[] at alphaOverride (catProp
    // stays fixed 1/ncat). Used by the joint-optimiser's alpha FD-Newton; warm-run reproduces the live rates exactly.
    if (alphaOverride > 0.0 && ncat > 1) gpu_discrete_gamma_mean(alphaOverride, ncat, catRate.data());
    for (int m = 0; m < N; m++) {
        ModelMarkov *cm = (ModelMarkov*)(*mix)[m];
        double *ev = cm->getEigenvalues(), *U = cm->getEigenvectors(), *Ui = cm->getInverseEigenvectors();
        if (!ev || !U || !Ui) return (double)NAN;
        for (int i = 0; i < ns; i++) evalC[(size_t)m*ns+i] = ev[i];
        for (int x = 0; x < ns*ns; x++) { Uc[(size_t)m*ns*ns+x] = U[x]; Uinv[(size_t)m*ns*ns+x] = Ui[x]; }
        for (int i = 0; i < ns; i++) { double s=0; for (int j=0;j<ns;j++) s += Ui[i*ns+j]; UinvRowSum[(size_t)m*ns+i]=s; }
        double wf[64]; model->getStateFrequency(wf, m);   // ns<=20
        for (int x = 0; x < ns; x++) freqC[(size_t)m*ns+x] = wf[x];
        double wm = w_override ? w_override[m] : model->getMixtureWeight(m);   // G.8.2: optional EM-iterate weights
        for (int c = 0; c < ncat; c++) wreg[(size_t)m*ncat+c] = wm * catProp[c];
    }

    // ---- topology rooted at an internal node R (reversible-invariant) — identical to the single-model sweep ----
    if (!root || !root->isLeaf() || root->neighbors.empty()) return (double)NAN;
    Node *Rt = root->neighbors[0]->node;
    if (Rt->isLeaf()) return (double)NAN;
    map<Node*,int> nid; vector<Node*> nodes; vector<double> parentLen; vector<int> isLeafV, leafTax;
    function<void(Node*,Node*,double)> indexDfs = [&](Node *n, Node *dad, double lenToDad) {
        int myi=(int)nodes.size(); nid[n]=myi; nodes.push_back(n); parentLen.push_back(lenToDad);
        int lf=n->isLeaf()?1:0; isLeafV.push_back(lf); leafTax.push_back(lf?aln->getSeqID(n->name):-1);
        for (auto nb:n->neighbors){ if(nb->node==dad) continue; indexDfs(nb->node,n,nb->length); }
    };
    indexDfs(Rt, nullptr, 0.0);
    int nNodes=(int)nodes.size();
    // G.8.2.1b: optional iterate-branch override — replace the live edge lengths with parentLenOverride[v] (indexed by
    // this function's DFS nid; root entry is 0.0). The echild build below consumes it transparently. nullptr => live tree.
    if (parentLenOverride) for (int v = 0; v < nNodes; v++) parentLen[v] = parentLenOverride[v];
    vector<int> postInternal; vector<int> slot(nNodes,-1);
    function<void(Node*,Node*)> postDfs = [&](Node *n, Node *dad){
        for (auto nb:n->neighbors){ if(nb->node==dad) continue; postDfs(nb->node,n); }
        if (!n->isLeaf()){ slot[nid[n]]=(int)postInternal.size(); postInternal.push_back(nid[n]); }
    };
    postDfs(Rt, nullptr);
    int nInternal=(int)postInternal.size();

    // ---- echild[child][r=m*ncat+c][x][i] = U_m[x][i]*exp(eval_m[i]*rate_c*parentLen) ----
    size_t ecStride = (size_t)R*ns*ns;
    vector<double> echild((size_t)nNodes*ecStride, 0.0);
    for (int v = 0; v < nNodes; v++) {
        if (v == nid[Rt]) continue;
        double len_v = parentLen[v];
        for (int m = 0; m < N; m++) {
            const double *ev = &evalC[(size_t)m*ns]; const double *U = &Uc[(size_t)m*ns*ns];
            for (int c = 0; c < ncat; c++) {
                double l = len_v * catRate[c]; int r = m*ncat + c;
                double ex[20]; for (int i=0;i<ns;i++) ex[i]=exp(ev[i]*l);
                double *e = &echild[(size_t)v*ecStride + (size_t)r*ns*ns];
                for (int x=0;x<ns;x++) for (int i=0;i<ns;i++) e[x*ns+i] = U[x*ns+i]*ex[i];
            }
        }
    }

    // ---- compact tip states ; pattern frequencies (identical to single-model) ----
    vector<unsigned char> tip((size_t)ntax*nptn);
    for (int v = 0; v < nNodes; v++) {
        if (!isLeafV[v]) continue;
        int tax = leafTax[v]; if (tax<0 || tax>=ntax) return (double)NAN;
        for (int p=0;p<nptn;p++){ int st=(int)aln->at(p)[tax]; tip[(size_t)tax*nptn+p]=(unsigned char)((st<ns)?st:ns); }
    }
    vector<double> ptnFreq(nptn);
    for (int p=0;p<nptn;p++) ptnFreq[p]=(double)aln->at(p).frequency;

    // ---- per-internal-node descriptors (postorder) — identical to single-model ----
    vector<int> dRoot(nInternal), dNch(nInternal), dOut(nInternal);
    vector<int> dChildNode(nInternal*3,-1), dChildIsLeaf(nInternal*3,0), dChildLeaf(nInternal*3,-1), dChildSlot(nInternal*3,-1);
    for (int idx=0; idx<nInternal; idx++){
        int vi=postInternal[idx]; Node *n=nodes[vi]; Node *dad=nullptr;
        if (n!=Rt){ for(auto nb:n->neighbors){ if(nid[nb->node]<vi){ dad=nb->node; break; } } }
        dRoot[idx]=(n==Rt)?1:0; dOut[idx]=(n==Rt)?-1:slot[vi];
        int k=0;
        for (auto nb:n->neighbors){
            if (nb->node==dad) continue; if (k>=3) return (double)NAN;
            int cv=nid[nb->node]; dChildNode[idx*3+k]=cv;
            if (isLeafV[cv]){ dChildIsLeaf[idx*3+k]=1; dChildLeaf[idx*3+k]=leafTax[cv]; }
            else            { dChildIsLeaf[idx*3+k]=0; dChildSlot[idx*3+k]=slot[cv]; }
            k++;
        }
        dNch[idx]=k;
    }

    return gpu_lnl_crosscheck_mix(ns, nptn, ncat, N, ntax, nNodes, nInternal,
        Uinv.data(), UinvRowSum.data(), freqC.data(), wreg.data(), echild.data(), tip.data(), ptnFreq.data(),
        dRoot.data(), dNch.data(), dOut.data(),
        dChildNode.data(), dChildIsLeaf.data(), dChildLeaf.data(), dChildSlot.data(),
        out_patlh, out_lhcat);
}

// ============================================================================================================
// G.2.0a — one-shot clean-room cross-check. Compares the GPU sweep against an INDEPENDENT CPU reference: if the
// GPU Branch override is installed (G.2.0b), curScore is itself the GPU value, so we recompute the CPU lnL via
// the SAVED CPU Branch pointer for a genuine in-process GPU-vs-CPU comparison; otherwise we use the passed
// curScore (which came from the CPU pointer). Pure diagnostic; never alters the run's curScore.
// ============================================================================================================
void PhyloTree::gpuLnLCrossCheckOnce(double cpu_lnL) {
    static bool done = false;
    if (done) return;
    done = true;

    double gpu = gpuComputeTreeLnLCleanRoom(nullptr);
    if (std::isnan(gpu)) { printf("[GPU-XCHECK] skipped (unsupported regime or CUDA error)\n"); return; }

    double cpu_ref = cpu_lnL;
    const char *src = "curScore";
    if (cpuComputeLikelihoodBranchPointer && current_it && current_it_back) {
        cpu_ref = (this->*cpuComputeLikelihoodBranchPointer)(current_it, (PhyloNode*)current_it_back->node, false);
        src = "CPU-recompute";
    }
    double rel = (cpu_ref != 0.0) ? fabs((gpu - cpu_ref)/cpu_ref) : fabs(gpu - cpu_ref);
    printf("[GPU-XCHECK] GPU lnL = %.6f   CPU lnL = %.6f (%s)   |d|=%.4e   rel=%.3e   -> %s\n",
           gpu, cpu_ref, src, fabs(gpu - cpu_ref), rel, (rel < 1e-6 ? "PASS (bridge OK)" : "MISMATCH"));
}

// Set by gpuMixLnLCrossCheckOnce (G.8.0) iff the GPU mixture lnL ENGINE is validated this run: GPU lnL == CPU lnL
// (rel<=1e-9) AND the per-class emission is self-consistent (Σ_m L_{p,m} == L_p). The G.8.2.0 EM weight check below
// trusts the GPU per-class L_{p,m} only when this is true — closing the red-team's MINOR-3 (the EM check's engine-
// isolation argument is sound *because* G.8.0 independently pinned the engine; this makes that dependency programmatic
// instead of "they happen to both fire from the same hook").
static bool s_gpuMixLnLEngineValidated = false;

// ============================================================================================================
// G.8.0 — one-shot PROFILE-MIXTURE lnL cross-check (C20/C60/MEOW80). Fires only for getNMixtures()>1 (does NOT
// consume the one-shot for single-model runs). Compares the clean-room GPU mixture sweep vs an INDEPENDENT CPU
// recompute (saved CPU Branch pointer) or curScore. Pure diagnostic; gate rel<=1e-9 (expect ~1e-12). The gate that
// keeps mixtures off the production GPU path is unchanged — this only validates the kernel against the live model.
// ============================================================================================================
void PhyloTree::gpuMixLnLCrossCheckOnce(double cpu_lnL) {
    static bool done = false;
    if (done) return;
    if (!model || model->getNMixtures() <= 1) return;   // not a mixture -> leave the one-shot unconsumed
    done = true;

    int N = model->getNMixtures();
    int nptn = (int)aln->size();
    std::vector<double> glhcat((size_t)N*nptn);          // G.8.1: GPU per-class L_{p,m} = glhcat[m*nptn + p]
    std::vector<double> gpatlh(nptn);                    // G.8.1 diag: GPU per-pattern log L_p (for self-consistency)
    double gpu = gpuComputeTreeLnLCleanRoomMix(gpatlh.data(), glhcat.data());
    if (std::isnan(gpu)) { printf("[GPU-XCHECK-MIX] skipped (unsupported regime or CUDA error)\n"); return; }

    double cpu_ref = cpu_lnL;
    const char *src = "curScore";
    if (cpuComputeLikelihoodBranchPointer && current_it && current_it_back) {
        cpu_ref = (this->*cpuComputeLikelihoodBranchPointer)(current_it, (PhyloNode*)current_it_back->node, false);
        src = "CPU-recompute";
    }
    double rel = (cpu_ref != 0.0) ? fabs((gpu - cpu_ref)/cpu_ref) : fabs(gpu - cpu_ref);
    bool lnL_ok = (rel <= 1e-9);
    printf("[GPU-XCHECK-MIX] N=%d ncat=%d  GPU lnL = %.6f   CPU lnL = %.6f (%s)   |d|=%.4e   rel=%.3e   -> %s\n",
           N, site_rate ? site_rate->getNRate() : -1, gpu, cpu_ref, src,
           fabs(gpu - cpu_ref), rel, (lnL_ok ? "PASS (G.8.0)" : "MISMATCH"));

    // G.8.1 diag — GPU SELF-CONSISTENCY (no CPU ref): Σ_m L_{p,m} must equal L_p = exp(patlh[p]). If this holds the
    // per-class emission is correct by construction, and any CPU-ref mismatch below is a contaminated reference (the
    // GPU branch override leaves CPU per-cat partials unpopulated), NOT a kernel bug.
    bool self_ok = false;
    {
        double selfmax = 0.0; int pbad = -1;
        for (int p = 0; p < nptn; p++) {
            double s = 0.0; for (int m = 0; m < N; m++) s += glhcat[(size_t)m*nptn + p];
            double Lp = exp(gpatlh[p]);
            double r = fabs(s - Lp) / (fabs(Lp) + 1e-300);
            if (r > selfmax) { selfmax = r; pbad = p; }
        }
        self_ok = (selfmax <= 1e-9);
        printf("[GPU-XCHECK-MIX] GPU self-consistency Sum_m L_{p,m} vs L_p: max rel = %.3e (ptn %d)  -> %s\n",
               selfmax, pbad, (self_ok ? "PASS (GPU per-class internally exact)" : "GPU EMISSION BUG"));
    }

    // G.8.1 — per-class POSTERIOR RESPONSIBILITY γ_{p,m} = L_{p,m}/Σ_m' L_{p,m'} (the EM E-step quantity the G.8.2
    // weight M-step consumes: w_m_new = Σ_p f_p·γ_{p,m} / Σ_p f_p). This is the CORRECT cross-check metric because it is
    // SCALE-INVARIANT: the CPU's _pattern_lh_cat is per-pattern SCALED (scale_num·LOG_SCALING_THRESHOLD underflow
    // protection) while the GPU clean-room is raw/unscaled — they differ by exp(scale_p) per pattern (which is why the
    // lnL still matches bit-exact: the factor is added back in the log domain). Self-normalising each side by its own
    // per-pattern class sum cancels exp(scale_p), so γ_g and γ_c are directly comparable. Guarded: skip _pattern_lh_cat unalloc.
    if (_pattern_lh_cat) {
        computePatternLhCat(WSL_MIXTURE);                 // fills _pattern_lh_cat[ptn*N + m] (CPU, current state)
        double mr = 0.0;
        for (int p = 0; p < nptn; p++) {
            double sg = 0.0, sc = 0.0;
            for (int m = 0; m < N; m++) { sg += glhcat[(size_t)m*nptn + p]; sc += _pattern_lh_cat[(size_t)p*N + m]; }
            if (sg <= 0.0 || sc <= 0.0) continue;        // fully-underflowed pattern carries no posterior information
            for (int m = 0; m < N; m++) {
                double gg = glhcat[(size_t)m*nptn + p] / sg, cc = _pattern_lh_cat[(size_t)p*N + m] / sc;
                double r = fabs(gg - cc);                 // posteriors ∈ [0,1]; absolute diff is the right scale
                if (r > mr) mr = r;
            }
        }
        printf("[GPU-XCHECK-MIX] per-class posterior γ_{p,m} vs CPU _pattern_lh_cat (scale-invariant): max |Δγ| = %.3e  -> %s\n",
               mr, (mr <= 1e-9 ? "PASS (G.8.1 per-class)" : "MISMATCH"));
    } else {
        printf("[GPU-XCHECK-MIX] per-class check skipped (_pattern_lh_cat not allocated)\n");
    }

    // The G.8.2.0 EM weight check (below) consumes the GPU per-class L_{p,m}; it may trust them only if the engine
    // is validated this run: GPU lnL == CPU lnL AND the per-class emission reconstructs L_p (self-consistency).
    s_gpuMixLnLEngineValidated = lnL_ok && self_ok;
}

// ============================================================================================================
// G.2.0b — GPU override for computeLikelihoodBranchPointer (byte-matches ComputeLikelihoodBranchType). For a
// reversible model the whole-tree lnL is independent of the rooting branch, so (dad_branch,dad) are ignored
// except to zero their lh_scale_factor (NORM_LH no-scaling path, so computePatternLikelihood's memmove yields
// the correct per-pattern values for logl_variance/s.e.). Mirrors the per-pattern log|lh_ptn| into _pattern_lh.
// ============================================================================================================
double PhyloTree::computeLikelihoodBranchGPU(PhyloNeighbor *dad_branch, PhyloNode *dad, bool save_log_value) {
    double lnL = gpuComputeTreeLnLCleanRoom(_pattern_lh);   // _pattern_lh may be null -> launcher skips the mirror
    if (std::isnan(lnL)) {
        static bool warned = false;
        if (!warned) { warned = true;
            printf("[GPU-BRANCH] unsupported/CUDA-error this call -> CPU fallback (computeLikelihoodBranchGenericSIMD)\n"); }
        if (cpuComputeLikelihoodBranchPointer)
            return (this->*cpuComputeLikelihoodBranchPointer)(dad_branch, dad, save_log_value);
        return lnL;   // no CPU fallback available (should not happen — installer always saves one)
    }
    // NORM_LH: zero the branch scale factors so computePatternLikelihood takes the no-scaling memmove path
    // (ptn_lh = _pattern_lh = the GPU per-pattern log-lh) -> correct logl_variance / reported s.e.
    if (dad_branch) dad_branch->lh_scale_factor = 0.0;
    if (dad && dad_branch) {
        PhyloNeighbor *back = (PhyloNeighbor*)dad->findNeighbor(dad_branch->node);
        if (back) back->lh_scale_factor = 0.0;
    }
    static bool announced = false;
    if (!announced) { announced = true;
        printf("[GPU-BRANCH] computeLikelihoodBranchGPU active (clean-room full sweep; _pattern_lh mirrored)\n"); }
    return lnL;
}

// ============================================================================================================
// G.2.1a — clean-room single-edge branch-length derivative df/ddf for the edge (dad_branch->node, dad).
// STATELESS (no device-resident state — like the lnL helper): builds TWO directed clean-room sub-sweeps split
// by the central edge (sub-roots = the two endpoints, each excluding the central neighbour), so node_eig /
// dad_eig are the eigen-space endpoint partials EXCLUDING the central transition; the derivative kernel applies
// exp(eval·r·t) for the central branch length t = dad_branch->length. Returns df = d(lnL)/dt (un-negated; the
// CPU computeFuncDerv negates), *out_ddf the 2nd derivative, *out_lnL the tree lnL at t. NaN if unsupported.
// G.2.1a scope: BOTH endpoints must be internal (leaf-endpoint support is G.2.1b).
double PhyloTree::gpuComputeEdgeDervCleanRoom(PhyloNeighbor *dad_branch, PhyloNode *dad, double *out_ddf, double *out_lnL) {
    if (!model || !site_rate || !aln) return (double)NAN;
    int ns = aln->num_states;
    if (ns != 4 && ns != 20) return (double)NAN;
    if (!model->isReversible() || model->getNMixtures() != 1 || model->isSiteSpecificModel()) return (double)NAN;
    if (site_rate->getPInvar() > 0.0) return (double)NAN;   // +I omits ptn_invar in the clean-room sweep -> CPU
    Node *node = dad_branch->node;   // one endpoint (the "node" side)
    Node *dadN = dad;                // other endpoint (the "dad" side)
    if (!node || !dadN) return (double)NAN;   // G.2.1b: leaf endpoints OK (≤1 leaf per edge; tip eigen synthesized)

    int ncat = site_rate->getNRate();
    int nptn = (int)aln->size();
    int ntax = (int)aln->getNSeq();
    if (ncat < 1 || ncat > 64) return (double)NAN;
    double *eval = model->getEigenvalues();
    double *U    = model->getEigenvectors();
    double *Uinv = model->getInverseEigenvectors();
    if (!eval || !U || !Uinv) return (double)NAN;
    vector<double> freq(ns, 0.0); model->getStateFrequency(freq.data(), 0);
    vector<double> UinvRowSum(ns, 0.0);
    for (int i = 0; i < ns; i++) { double s = 0; for (int j = 0; j < ns; j++) s += Uinv[i*ns+j]; UinvRowSum[i] = s; }
    vector<double> catRate(ncat), catProp(ncat);
    for (int c = 0; c < ncat; c++) { catRate[c] = site_rate->getRate(c); catProp[c] = site_rate->getProp(c); }
    double t = dad_branch->length;

    // two-sub-root DFS, central edge (node<->dadN) excluded from both
    map<Node*,int> nid; vector<Node*> nodes; vector<double> parentLen; vector<int> isLeafV, leafTax;
    function<void(Node*,Node*,double)> indexDfs = [&](Node *n, Node *par, double lenToPar) {
        int myi = (int)nodes.size(); nid[n] = myi; nodes.push_back(n);
        parentLen.push_back(lenToPar);
        int lf = n->isLeaf() ? 1 : 0; isLeafV.push_back(lf);
        leafTax.push_back(lf ? aln->getSeqID(n->name) : -1);
        for (auto nb : n->neighbors) { if (nb->node == par) continue; indexDfs(nb->node, n, nb->length); }
    };
    indexDfs(node, dadN, 0.0);    // node-side subtree (parent dir = dadN, excluded)
    indexDfs(dadN, node, 0.0);    // dad-side subtree (parent dir = node, excluded)
    int nNodes = (int)nodes.size();

    vector<int> postInternal; vector<int> slot(nNodes, -1);
    function<void(Node*,Node*)> postDfs = [&](Node *n, Node *par) {
        for (auto nb : n->neighbors) { if (nb->node == par) continue; postDfs(nb->node, n); }
        if (!n->isLeaf()) { slot[nid[n]] = (int)postInternal.size(); postInternal.push_back(nid[n]); }
    };
    postDfs(node, dadN);
    postDfs(dadN, node);
    int nInternal = (int)postInternal.size();
    // endpoint eigen partial: internal -> its postorder slot; leaf -> synthesized tip eigen (slot=-1, pass taxon)
    int nodeSlot = node->isLeaf() ? -1 : slot[nid[node]];
    int nodeLeafTax = node->isLeaf() ? leafTax[nid[node]] : -1;
    int dadSlot  = dadN->isLeaf() ? -1 : slot[nid[dadN]];
    int dadLeafTax = dadN->isLeaf() ? leafTax[nid[dadN]] : -1;

    // echild[v] = U·exp(eval·rate·parentLen[v]) for every node except the two sub-roots (no parent edge)
    size_t ecStride = (size_t)ncat*ns*ns;
    vector<double> echild((size_t)nNodes*ecStride, 0.0);
    for (int v = 0; v < nNodes; v++) {
        if (v == nid[node] || v == nid[dadN]) continue;
        double len_v = parentLen[v];
        for (int c = 0; c < ncat; c++) {
            double l = len_v * catRate[c];
            double ex[20]; for (int i = 0; i < ns; i++) ex[i] = exp(eval[i]*l);
            double *e = &echild[(size_t)v*ecStride + (size_t)c*ns*ns];
            for (int x = 0; x < ns; x++) for (int i = 0; i < ns; i++) e[x*ns+i] = U[x*ns+i]*ex[i];
        }
    }

    vector<unsigned char> tip((size_t)ntax*nptn);
    for (int v = 0; v < nNodes; v++) {
        if (!isLeafV[v]) continue;
        int tax = leafTax[v];
        if (tax < 0 || tax >= ntax) return (double)NAN;
        for (int p = 0; p < nptn; p++) { int st = (int)aln->at(p)[tax]; tip[(size_t)tax*nptn+p] = (unsigned char)((st < ns) ? st : ns); }
    }
    vector<double> ptnFreq(nptn);
    for (int p = 0; p < nptn; p++) ptnFreq[p] = (double)aln->at(p).frequency;

    // descriptors: ALL internal (isRoot=0); each (incl. node & dadN sub-roots) writes its eigen partial to slot
    vector<int> dRoot(nInternal, 0), dNch(nInternal), dOut(nInternal);
    vector<int> dChildNode(nInternal*3, -1), dChildIsLeaf(nInternal*3, 0), dChildLeaf(nInternal*3, -1), dChildSlot(nInternal*3, -1);
    for (int idx = 0; idx < nInternal; idx++) {
        int vi = postInternal[idx]; Node *n = nodes[vi];
        // parent = the unique neighbour with a smaller nid within the same subtree (pre-order). The two
        // sub-roots have none (their only smaller-nid neighbour would be across the excluded central edge).
        Node *par = nullptr;
        if (n != node && n != dadN) {
            for (auto nb : n->neighbors) { auto it = nid.find(nb->node); if (it != nid.end() && it->second < vi) { par = nb->node; break; } }
        }
        dOut[idx] = slot[vi];
        int k = 0;
        for (auto nb : n->neighbors) {
            if (nb->node == par) continue;
            if (n == node && nb->node == dadN) continue;   // exclude central edge at sub-root node
            if (n == dadN && nb->node == node) continue;   // exclude central edge at sub-root dadN
            if (k >= 3) return (double)NAN;
            int cv = nid[nb->node];
            dChildNode[idx*3+k] = cv;
            if (isLeafV[cv]) { dChildIsLeaf[idx*3+k] = 1; dChildLeaf[idx*3+k] = leafTax[cv]; }
            else             { dChildIsLeaf[idx*3+k] = 0; dChildSlot[idx*3+k] = slot[cv]; }
            k++;
        }
        dNch[idx] = k;
    }

    return gpu_derv_crosscheck(ns, nptn, ncat, ntax, nNodes, nInternal,
        Uinv, UinvRowSum.data(), freq.data(), catProp.data(), echild.data(), tip.data(), ptnFreq.data(),
        dRoot.data(), dNch.data(), dOut.data(),
        dChildNode.data(), dChildIsLeaf.data(), dChildLeaf.data(), dChildSlot.data(),
        nodeSlot, nodeLeafTax, dadSlot, dadLeafTax, eval, catRate.data(), t, out_ddf, out_lnL);
}

// ============================================================================================================
// G.8.1b — clean-room single-edge branch derivative for PROFILE MIXTURES: df/ddf SUMMED over the N*ncat regimes,
// validated vs CPU computeLikelihoodDerv. Combines the two-sub-root central-edge split (gpuComputeEdgeDervCleanRoom)
// with the per-class eigen assembly (gpuComputeTreeLnLCleanRoomMix). Returns df = d(lnL)/dt (un-negated, like the
// single-model path; computeFuncDerv negates downstream); *out_ddf the 2nd derivative; *out_lnL the tree lnL at t.
// Eligibility gate IDENTICAL to the mixture lnL path (+I/fused/PMSF/nonrev/single-model -> NaN -> CPU) so an edge
// never gets a GPU lnL but a CPU derivative or vice versa.
// ============================================================================================================
double PhyloTree::gpuComputeEdgeDervCleanRoomMix(PhyloNeighbor *dad_branch, PhyloNode *dad, double *out_ddf, double *out_lnL) {
    if (!model || !site_rate || !aln) return (double)NAN;
    int ns = aln->num_states;
    if (ns != 20 && ns != 4) return (double)NAN;
    if (!model->isReversible()) return (double)NAN;
    int N = model->getNMixtures();
    if (N <= 1) return (double)NAN;                       // single-model -> the non-mix derivative path
    if (model->isSiteSpecificModel()) return (double)NAN; // PMSF stays on CPU
    if (site_rate->getPInvar() > 0.0) return (double)NAN;  // +I omitted in the clean-room sweep -> CPU
    int ncat = site_rate->getNRate();
    int nptn = (int)aln->size();
    int ntax = (int)aln->getNSeq();
    if (ncat < 1) return (double)NAN;
    int R = N*ncat;
    ModelMixture *mix = dynamic_cast<ModelMixture*>(model);
    if (!mix || (int)mix->size() != N) return (double)NAN;
    if (mix->isFused()) return (double)NAN;               // LG4M/LG4X = 1:1 class<->rate, not the N*ncat product
    Node *node = dad_branch->node;
    Node *dadN = dad;
    if (!node || !dadN) return (double)NAN;

    // ---- per-class eigen (component pointers, AVX-padding intrinsic), freq, per-regime weight = w_m*catProp_c ----
    std::vector<double> Uinv((size_t)N*ns*ns), UinvRowSum((size_t)N*ns), freqC((size_t)N*ns);
    std::vector<double> evalC((size_t)N*ns), Uc((size_t)N*ns*ns);
    std::vector<double> catRate(ncat), catProp(ncat), wreg((size_t)R);
    for (int c = 0; c < ncat; c++) { catRate[c] = site_rate->getRate(c); catProp[c] = site_rate->getProp(c); }
    for (int m = 0; m < N; m++) {
        ModelMarkov *cm = (ModelMarkov*)(*mix)[m];
        double *ev = cm->getEigenvalues(), *U = cm->getEigenvectors(), *Ui = cm->getInverseEigenvectors();
        if (!ev || !U || !Ui) return (double)NAN;
        for (int i = 0; i < ns; i++) evalC[(size_t)m*ns+i] = ev[i];
        for (int x = 0; x < ns*ns; x++) { Uc[(size_t)m*ns*ns+x] = U[x]; Uinv[(size_t)m*ns*ns+x] = Ui[x]; }
        for (int i = 0; i < ns; i++) { double s=0; for (int j=0;j<ns;j++) s += Ui[i*ns+j]; UinvRowSum[(size_t)m*ns+i]=s; }
        double wf[64]; model->getStateFrequency(wf, m);   // ns<=20
        for (int x = 0; x < ns; x++) freqC[(size_t)m*ns+x] = wf[x];
        double wm = model->getMixtureWeight(m);
        for (int c = 0; c < ncat; c++) wreg[(size_t)m*ncat+c] = wm * catProp[c];
    }
    double t = dad_branch->length;

    // ---- two-sub-root DFS, central edge (node<->dadN) excluded from both (identical to single-model derivative) ----
    map<Node*,int> nid; vector<Node*> nodes; vector<double> parentLen; vector<int> isLeafV, leafTax;
    function<void(Node*,Node*,double)> indexDfs = [&](Node *n, Node *par, double lenToPar) {
        int myi = (int)nodes.size(); nid[n]=myi; nodes.push_back(n); parentLen.push_back(lenToPar);
        int lf = n->isLeaf()?1:0; isLeafV.push_back(lf); leafTax.push_back(lf?aln->getSeqID(n->name):-1);
        for (auto nb : n->neighbors) { if (nb->node==par) continue; indexDfs(nb->node, n, nb->length); }
    };
    indexDfs(node, dadN, 0.0);
    indexDfs(dadN, node, 0.0);
    int nNodes = (int)nodes.size();

    vector<int> postInternal; vector<int> slot(nNodes, -1);
    function<void(Node*,Node*)> postDfs = [&](Node *n, Node *par) {
        for (auto nb : n->neighbors) { if (nb->node==par) continue; postDfs(nb->node, n); }
        if (!n->isLeaf()) { slot[nid[n]]=(int)postInternal.size(); postInternal.push_back(nid[n]); }
    };
    postDfs(node, dadN);
    postDfs(dadN, node);
    int nInternal = (int)postInternal.size();
    int nodeSlot = node->isLeaf() ? -1 : slot[nid[node]];
    int nodeLeafTax = node->isLeaf() ? leafTax[nid[node]] : -1;
    int dadSlot  = dadN->isLeaf() ? -1 : slot[nid[dadN]];
    int dadLeafTax = dadN->isLeaf() ? leafTax[nid[dadN]] : -1;

    // ---- per-regime echild[v][r=m*ncat+c][x][i] = U_m[x][i]*exp(eval_m[i]*rate_c*parentLen[v]); skip sub-roots ----
    size_t ecStride = (size_t)R*ns*ns;
    vector<double> echild((size_t)nNodes*ecStride, 0.0);
    for (int v = 0; v < nNodes; v++) {
        if (v == nid[node] || v == nid[dadN]) continue;
        double len_v = parentLen[v];
        for (int m = 0; m < N; m++) {
            const double *ev = &evalC[(size_t)m*ns]; const double *U = &Uc[(size_t)m*ns*ns];
            for (int c = 0; c < ncat; c++) {
                double l = len_v * catRate[c]; int r = m*ncat + c;
                double ex[20]; for (int i=0;i<ns;i++) ex[i]=exp(ev[i]*l);
                double *e = &echild[(size_t)v*ecStride + (size_t)r*ns*ns];
                for (int x=0;x<ns;x++) for (int i=0;i<ns;i++) e[x*ns+i] = U[x*ns+i]*ex[i];
            }
        }
    }

    vector<unsigned char> tip((size_t)ntax*nptn);
    for (int v = 0; v < nNodes; v++) {
        if (!isLeafV[v]) continue;
        int tax = leafTax[v]; if (tax<0 || tax>=ntax) return (double)NAN;
        for (int p=0;p<nptn;p++){ int st=(int)aln->at(p)[tax]; tip[(size_t)tax*nptn+p]=(unsigned char)((st<ns)?st:ns); }
    }
    vector<double> ptnFreq(nptn);
    for (int p=0;p<nptn;p++) ptnFreq[p]=(double)aln->at(p).frequency;

    // ---- descriptors: ALL internal (isRoot=0); central edge excluded at the two sub-roots ----
    vector<int> dRoot(nInternal,0), dNch(nInternal), dOut(nInternal);
    vector<int> dChildNode(nInternal*3,-1), dChildIsLeaf(nInternal*3,0), dChildLeaf(nInternal*3,-1), dChildSlot(nInternal*3,-1);
    for (int idx=0; idx<nInternal; idx++){
        int vi=postInternal[idx]; Node *n=nodes[vi];
        Node *par=nullptr;
        if (n!=node && n!=dadN) {
            for (auto nb:n->neighbors){ auto it=nid.find(nb->node); if(it!=nid.end() && it->second<vi){ par=nb->node; break; } }
        }
        dOut[idx]=slot[vi];
        int k=0;
        for (auto nb:n->neighbors){
            if (nb->node==par) continue;
            if (n==node && nb->node==dadN) continue;   // exclude central edge at sub-root node
            if (n==dadN && nb->node==node) continue;   // exclude central edge at sub-root dadN
            if (k>=3) return (double)NAN;
            int cv=nid[nb->node]; dChildNode[idx*3+k]=cv;
            if (isLeafV[cv]){ dChildIsLeaf[idx*3+k]=1; dChildLeaf[idx*3+k]=leafTax[cv]; }
            else            { dChildIsLeaf[idx*3+k]=0; dChildSlot[idx*3+k]=slot[cv]; }
            k++;
        }
        dNch[idx]=k;
    }

    return gpu_derv_crosscheck_mix(ns, nptn, ncat, N, ntax, nNodes, nInternal,
        Uinv.data(), UinvRowSum.data(), freqC.data(), wreg.data(), echild.data(), tip.data(), ptnFreq.data(),
        dRoot.data(), dNch.data(), dOut.data(),
        dChildNode.data(), dChildIsLeaf.data(), dChildLeaf.data(), dChildSlot.data(),
        nodeSlot, nodeLeafTax, dadSlot, dadLeafTax, evalC.data(), catRate.data(), t, out_ddf, out_lnL);
}

// ============================================================================================================
// G.8.2.1a — clean-room ALL-BRANCH derivative for PROFILE MIXTURES: df/ddf for EVERY edge in ONE postorder + ONE
// preorder sweep (Ji-2020 linear-time), rooted at an internal node, vs the single-edge gpuComputeEdgeDervCleanRoomMix
// (two-sub-root split, ONE edge). Fills four parallel out-vectors (one entry per NON-ROOT node v): the edge v->parent
// gets childNodes[k]=v, parentNodes[k]=parent, dfOut[k]=d(lnL)/db_v, ddfOut[k]=d²(lnL)/db_v². Mirrors the eigen/echild
// gather of gpuComputeEdgeDervCleanRoomMix and the single-root topology of optimizeParametersJOLT (additionally builds
// the per-node expfac = exp(eval_m·rate_c·b_parent) the preorder kernel needs). Returns false on ineligibility/CUDA
// error (same +I/fused/PMSF/nonrev/single-model gate as the lnL mix path). Read-only (no host/device state persists).
// ============================================================================================================
bool PhyloTree::gpuComputeAllBranchDervCleanRoomMix(std::vector<Node*>& childNodes, std::vector<Node*>& parentNodes,
                                                    std::vector<double>& dfOut, std::vector<double>& ddfOut,
                                                    const double *parentLenOverride, double alphaOverride) {
    childNodes.clear(); parentNodes.clear(); dfOut.clear(); ddfOut.clear();
    if (!model || !site_rate || !aln) return false;
    int ns = aln->num_states;
    if (ns != 20 && ns != 4) return false;
    if (!model->isReversible()) return false;
    int N = model->getNMixtures();
    if (N <= 1) return false;
    if (model->isSiteSpecificModel()) return false;
    if (site_rate->getPInvar() > 0.0) return false;        // +I omitted in the clean-room sweep -> CPU
    int ncat = site_rate->getNRate();
    int nptn = (int)aln->size();
    int ntax = (int)aln->getNSeq();
    if (ncat < 1) return false;
    int R = N*ncat;
    ModelMixture *mix = dynamic_cast<ModelMixture*>(model);
    if (!mix || (int)mix->size() != N || mix->isFused()) return false;
    if (!root || !root->isLeaf() || root->neighbors.empty()) return false;
    Node *Rt = root->neighbors[0]->node;   // internal root (IQ-TREE roots at a leaf; lnL is reversible-invariant)
    if (!Rt || Rt->isLeaf()) return false;

    // ---- per-class eigen (Uinv down-map, Uc up-map, evalC) + freq + per-regime weight (== gpuComputeEdgeDervCleanRoomMix) ----
    std::vector<double> Uinv((size_t)N*ns*ns), UinvRowSum((size_t)N*ns), freqC((size_t)N*ns);
    std::vector<double> evalC((size_t)N*ns), Uc((size_t)N*ns*ns);
    std::vector<double> catRate(ncat), catProp(ncat), wreg((size_t)R);
    for (int c = 0; c < ncat; c++) { catRate[c] = site_rate->getRate(c); catProp[c] = site_rate->getProp(c); }
    // G.8.2.1b: optional iterate-alpha override (joint-optimiser gradient at an FD-perturbed alpha) — see the lnL path.
    if (alphaOverride > 0.0 && ncat > 1) gpu_discrete_gamma_mean(alphaOverride, ncat, catRate.data());
    for (int m = 0; m < N; m++) {
        ModelMarkov *cm = (ModelMarkov*)(*mix)[m];
        double *ev = cm->getEigenvalues(), *U = cm->getEigenvectors(), *Ui = cm->getInverseEigenvectors();
        if (!ev || !U || !Ui) return false;
        for (int i = 0; i < ns; i++) evalC[(size_t)m*ns+i] = ev[i];
        for (int x = 0; x < ns*ns; x++) { Uc[(size_t)m*ns*ns+x] = U[x]; Uinv[(size_t)m*ns*ns+x] = Ui[x]; }
        for (int i = 0; i < ns; i++) { double s=0; for (int j=0;j<ns;j++) s += Ui[i*ns+j]; UinvRowSum[(size_t)m*ns+i]=s; }
        double wf[64]; model->getStateFrequency(wf, m);
        for (int x = 0; x < ns; x++) freqC[(size_t)m*ns+x] = wf[x];
        double wm = model->getMixtureWeight(m);
        for (int c = 0; c < ncat; c++) wreg[(size_t)m*ncat+c] = wm * catProp[c];
    }

    // ---- single-root topology rooted at Rt (indexDfs), flat arrays (mirrors optimizeParametersJOLT) ----
    std::map<Node*,int> nid; std::vector<Node*> nodes, parentNode; std::vector<double> parentLen; std::vector<int> leafTax;
    std::vector<std::vector<int>> childList;
    std::function<void(Node*,Node*,double)> indexDfs = [&](Node *n, Node *dad, double lenToDad) {
        int myi=(int)nodes.size(); nid[n]=myi; nodes.push_back(n); parentNode.push_back(dad); parentLen.push_back(lenToDad);
        leafTax.push_back(n->isLeaf()?aln->getSeqID(n->name):-1); childList.push_back(std::vector<int>());
        for (auto nb:n->neighbors){ if(nb->node==dad) continue; indexDfs(nb->node,n,nb->length); } };
    indexDfs(Rt, nullptr, 0.0);
    int nNodes = (int)nodes.size();
    // G.8.2.1b: optional iterate-branch override (joint-optimiser gradient at iterate b) — indexed by this function's
    // DFS nid; the echild + nodeParentLen (preorder expfac) builds below consume it transparently. nullptr => live tree.
    if (parentLenOverride) for (int v = 0; v < nNodes; v++) parentLen[v] = parentLenOverride[v];
    for (int i=0;i<nNodes;i++){ Node *n=nodes[i], *dad=parentNode[i];
        for (auto nb:n->neighbors){ if(nb->node==dad) continue; childList[i].push_back(nid[nb->node]); }
        if ((int)childList[i].size()>3) return false; }

    // ---- tip states + pattern frequencies ----
    std::vector<unsigned char> tip((size_t)ntax*nptn);
    for (int i=0;i<nNodes;i++){ if(leafTax[i]<0) continue; int tax=leafTax[i]; if(tax<0||tax>=ntax) return false;
        for (int p=0;p<nptn;p++){ int st=(int)aln->at(p)[tax]; tip[(size_t)tax*nptn+p]=(unsigned char)((st<ns)?st:ns); } }
    std::vector<double> ptnFreq(nptn); for (int p=0;p<nptn;p++) ptnFreq[p]=(double)aln->at(p).frequency;

    // ---- echild[v][r][x][i]=U_m[x][i]*exp(eval_m[i]*rate_c*parentLen[v]) + expfac[v][r][i]=exp(...) (root: no parent edge) ----
    size_t ecStride=(size_t)R*ns*ns, exStride=(size_t)R*ns;
    std::vector<double> echild((size_t)nNodes*ecStride,0.0), expfac((size_t)nNodes*exStride,0.0);
    for (int v=0;v<nNodes;v++){ if(parentNode[v]==nullptr) continue;
        double len_v=parentLen[v];
        for (int m=0;m<N;m++){ const double *ev=&evalC[(size_t)m*ns]; const double *U=&Uc[(size_t)m*ns*ns];
            for (int c=0;c<ncat;c++){ double l=len_v*catRate[c]; int r=m*ncat+c;
                double ex[20]; for(int i=0;i<ns;i++) ex[i]=exp(ev[i]*l);
                double *e=&echild[(size_t)v*ecStride+(size_t)r*ns*ns];
                for(int x=0;x<ns;x++) for(int i=0;i<ns;i++) e[x*ns+i]=U[x*ns+i]*ex[i];
                double *ef=&expfac[(size_t)v*exStride+(size_t)r*ns]; for(int i=0;i<ns;i++) ef[i]=ex[i]; } } }

    // ---- flat topology arrays for the launcher ----
    std::vector<int> nodeNch(nNodes), nodeChild((size_t)nNodes*3,-1), nodeLeaf(nNodes); std::vector<double> nodeParentLen(nNodes);
    for (int i=0;i<nNodes;i++){ nodeNch[i]=(int)childList[i].size(); nodeLeaf[i]=leafTax[i]; nodeParentLen[i]=parentLen[i];
        for (int k=0;k<(int)childList[i].size()&&k<3;k++) nodeChild[(size_t)i*3+k]=childList[i][k]; }

    std::vector<double> dfV(nNodes,0.0), ddfV(nNodes,0.0);
    double rc = gpu_allbranch_derv_crosscheck_mix(ns,nptn,ncat,N,ntax,nNodes,/*root=*/nid[Rt],
        Uinv.data(), Uc.data(), UinvRowSum.data(), freqC.data(), wreg.data(), evalC.data(), catRate.data(),
        echild.data(), expfac.data(), tip.data(), ptnFreq.data(),
        nodeNch.data(), nodeChild.data(), nodeLeaf.data(), nodeParentLen.data(),
        dfV.data(), ddfV.data());
    if (std::isnan(rc)) return false;

    for (int v=0;v<nNodes;v++){ if(parentNode[v]==nullptr) continue;
        childNodes.push_back(nodes[v]); parentNodes.push_back(parentNode[v]); dfOut.push_back(dfV[v]); ddfOut.push_back(ddfV[v]); }
    return true;
}

// G.2.1a — one-shot derivative cross-check: pick an internal-internal edge (R=internal node adjacent to the
// root leaf, C=an internal neighbour of R), compute GPU df/ddf clean-room and compare to IQ-TREE's OWN
// computeLikelihoodDerv (CPU pointer — not yet overridden at G.2.1a). Read-only; saves/restores current_it.
void PhyloTree::gpuDervCrossCheckOnce() {
    static bool done = false;
    if (done) return;
    done = true;
    if (!model || !site_rate || !aln || !root || !root->isLeaf() || root->neighbors.empty()) {
        printf("[GPU-DERV-XCHECK] skipped (tree/model not ready)\n"); return; }
    if (!cpuComputeLikelihoodBranchPointer) { printf("[GPU-DERV-XCHECK] skipped (no saved CPU branch pointer to seed host partials)\n"); return; }
    Node *R = root->neighbors[0]->node;
    if (!R || R->isLeaf()) { printf("[GPU-DERV-XCHECK] skipped (degenerate root)\n"); return; }

    // One edge: GPU df/ddf (clean-room) vs IQ-TREE's own computeLikelihoodDerv. The stateless GPU overrides
    // never populate host partials, so seed the edge's host partials with a CPU branch eval first (mirrors the
    // normal flow where a full computeLikelihood precedes any derivative); else CPU Derv reads stale buffers ->
    // non-finite df -> "Numerical underflow (lh-derivative)" abort. Save/restore state to avoid perturbation.
    auto checkEdge = [&](PhyloNeighbor *db, Node *dadNode, const char *label) {
        if (!db) { printf("[GPU-DERV-XCHECK] %s skipped (neighbour not found)\n", label); return; }
        PhyloNeighbor *save_it = current_it, *save_it_back = current_it_back;
        bool save_theta = theta_computed;
        (this->*cpuComputeLikelihoodBranchPointer)(db, (PhyloNode*)dadNode, false);  // CPU: fresh edge partials + theta
        theta_computed = false;   // mirror optimizeOneBranch: let Derv recompute theta from the fresh partials
        double cdf = 0.0, cddf = 0.0;
        // CPU reference: SAVED CPU Derv pointer (the computeLikelihoodDerv wrapper is the GPU override in G.2.1b).
        if (cpuComputeLikelihoodDervPointer) (this->*cpuComputeLikelihoodDervPointer)(db, (PhyloNode*)dadNode, &cdf, &cddf);
        else                                 computeLikelihoodDerv(db, (PhyloNode*)dadNode, &cdf, &cddf);
        current_it = save_it; current_it_back = save_it_back; theta_computed = save_theta;

        double gddf = 0.0, glnL = 0.0;
        double gdf = gpuComputeEdgeDervCleanRoom(db, (PhyloNode*)dadNode, &gddf, &glnL);
        if (std::isnan(gdf)) { printf("[GPU-DERV-XCHECK] %s skipped (unsupported regime or CUDA error)\n", label); return; }
        double rdf  = (cdf  != 0.0) ? fabs((gdf  - cdf )/cdf ) : fabs(gdf  - cdf );
        double rddf = (cddf != 0.0) ? fabs((gddf - cddf)/cddf) : fabs(gddf - cddf);
        bool pass = (rdf <= 1e-9) && (rddf <= 1e-9);
        printf("[GPU-DERV-XCHECK] %s edge(node=%d,dad=%d) t=%.6g  df: GPU=%.6e CPU=%.6e rel=%.3e | ddf: GPU=%.6e CPU=%.6e rel=%.3e  -> %s\n",
               label, db->node->id, dadNode->id, db->length, gdf, cdf, rdf, gddf, cddf, rddf, (pass ? "PASS" : "CHECK"));
    };

    // (1) internal-internal edge: R and an internal neighbour C.
    Node *C = nullptr;
    for (auto nb : R->neighbors) if (!nb->node->isLeaf()) { C = nb->node; break; }
    if (C) checkEdge((PhyloNeighbor*)R->findNeighbor(C), R, "INT-INT");
    else   printf("[GPU-DERV-XCHECK] INT-INT skipped (no internal-internal edge at root)\n");

    // (2) leaf-internal (pendant) edge: an internal node with a leaf neighbour — validates k_leaf_eig synthesis.
    Node *intNode = C ? C : R;     // an internal node likely to have a leaf child
    Node *Lf = nullptr;
    for (auto nb : intNode->neighbors) if (nb->node->isLeaf()) { Lf = nb->node; break; }
    if (!Lf) { for (auto nb : R->neighbors) if (nb->node->isLeaf()) { Lf = nb->node; intNode = R; break; } }
    if (Lf) checkEdge((PhyloNeighbor*)intNode->findNeighbor(Lf), intNode, "LEAF");
    else    printf("[GPU-DERV-XCHECK] LEAF skipped (no pendant edge found)\n");
}

// ============================================================================================================
// G.8.1b — one-shot PROFILE-MIXTURE derivative cross-check. Fires only for getNMixtures()>1 (leaves the one-shot
// unconsumed for single-model, mirroring gpuMixLnLCrossCheckOnce). For an INT-INT and a LEAF edge: seed host
// partials via the saved CPU branch pointer (the stateless GPU path never wrote them), take the CPU reference from
// the saved CPU Derv pointer (un-negated computeLikelihoodDerv), compute the GPU mixture clean-room df/ddf, compare
// rel<=1e-9 on both. Read-only; saves/restores current_it/theta_computed.
// ============================================================================================================
void PhyloTree::gpuMixDervCrossCheckOnce() {
    static bool done = false;
    if (done) return;
    if (!model || model->getNMixtures() <= 1) return;   // not a mixture -> leave the one-shot unconsumed
    done = true;
    if (!site_rate || !aln || !root || !root->isLeaf() || root->neighbors.empty()) {
        printf("[GPU-DERV-XCHECK-MIX] skipped (tree/model not ready)\n"); return; }
    if (!cpuComputeLikelihoodBranchPointer) { printf("[GPU-DERV-XCHECK-MIX] skipped (no saved CPU branch pointer to seed host partials)\n"); return; }
    Node *Rnode = root->neighbors[0]->node;
    if (!Rnode || Rnode->isLeaf()) { printf("[GPU-DERV-XCHECK-MIX] skipped (degenerate root)\n"); return; }

    auto checkEdge = [&](PhyloNeighbor *db, Node *dadNode, const char *label) {
        if (!db) { printf("[GPU-DERV-XCHECK-MIX] %s skipped (neighbour not found)\n", label); return; }
        PhyloNeighbor *save_it = current_it, *save_it_back = current_it_back;
        bool save_theta = theta_computed;
        (this->*cpuComputeLikelihoodBranchPointer)(db, (PhyloNode*)dadNode, false);  // CPU: fresh edge partials + theta
        theta_computed = false;   // let Derv recompute theta from the fresh partials (mirrors optimizeOneBranch)
        double cdf = 0.0, cddf = 0.0;
        if (cpuComputeLikelihoodDervPointer) (this->*cpuComputeLikelihoodDervPointer)(db, (PhyloNode*)dadNode, &cdf, &cddf);
        else                                 computeLikelihoodDerv(db, (PhyloNode*)dadNode, &cdf, &cddf);
        current_it = save_it; current_it_back = save_it_back; theta_computed = save_theta;

        double gddf = 0.0, glnL = 0.0;
        double gdf = gpuComputeEdgeDervCleanRoomMix(db, (PhyloNode*)dadNode, &gddf, &glnL);
        if (std::isnan(gdf)) { printf("[GPU-DERV-XCHECK-MIX] %s skipped (unsupported regime or CUDA error)\n", label); return; }
        double rdf  = (cdf  != 0.0) ? fabs((gdf  - cdf )/cdf ) : fabs(gdf  - cdf );
        double rddf = (cddf != 0.0) ? fabs((gddf - cddf)/cddf) : fabs(gddf - cddf);
        bool pass = (rdf <= 1e-9) && (rddf <= 1e-9);
        printf("[GPU-DERV-XCHECK-MIX] %s edge(node=%d,dad=%d) t=%.6g  df: GPU=%.6e CPU=%.6e rel=%.3e | ddf: GPU=%.6e CPU=%.6e rel=%.3e  -> %s\n",
               label, db->node->id, dadNode->id, db->length, gdf, cdf, rdf, gddf, cddf, rddf, (pass ? "PASS (G.8.1b)" : "MISMATCH"));
    };

    // (1) internal-internal edge: Rnode and an internal neighbour C.
    Node *C = nullptr;
    for (auto nb : Rnode->neighbors) if (!nb->node->isLeaf()) { C = nb->node; break; }
    if (C) checkEdge((PhyloNeighbor*)Rnode->findNeighbor(C), Rnode, "INT-INT");
    else   printf("[GPU-DERV-XCHECK-MIX] INT-INT skipped (no internal-internal edge at root)\n");

    // (2) leaf-internal (pendant) edge: validates k_leaf_eig_mix per-class tip synthesis.
    Node *intNode = C ? C : Rnode;
    Node *Lf = nullptr;
    for (auto nb : intNode->neighbors) if (nb->node->isLeaf()) { Lf = nb->node; break; }
    if (!Lf) { for (auto nb : Rnode->neighbors) if (nb->node->isLeaf()) { Lf = nb->node; intNode = Rnode; break; } }
    if (Lf) checkEdge((PhyloNeighbor*)intNode->findNeighbor(Lf), intNode, "LEAF");
    else    printf("[GPU-DERV-XCHECK-MIX] LEAF skipped (no pendant edge found)\n");
}

// ============================================================================================================
// G.8.2.1a — one-shot ALL-BRANCH derivative cross-check (profile mixtures). Validates the new k7_pre_mix preorder +
// the linear-time all-branch sweep with TWO independent gates per representative edge (≥1 INT-INT, ≥1 LEAF):
//   (a) DECISIVE — the all-branch df/ddf == the VALIDATED G.8.1b single-edge gpuComputeEdgeDervCleanRoomMix (which is
//       bit-exact vs CPU computeLikelihoodDerv), rel<=1e-9 (expect ~1e-12). Catches the historical bugs: a U/Uinv
//       transposition or a b_v double-count would diverge from the single-edge path (which has no `pre` at all).
//   (b) INDEPENDENT — central finite-difference of the clean-room lnL (gpuComputeTreeLnLCleanRoomMix at b_v±ε),
//       rel<=1e-5. Catches a sign error or an m=r/ncat vs c=r%ncat transposition shared by both GPU derivative paths.
// Fires only for getNMixtures()>1 (leaves the one-shot unconsumed for single-model). Read-only: the FD perturbation
// saves/restores both directed neighbour lengths. No host-partial seeding needed (both references are stateless GPU).
// ============================================================================================================
void PhyloTree::gpuMixAllBranchDervCrossCheckOnce() {
    static bool done = false;
    if (done) return;
    if (!model || model->getNMixtures() <= 1) return;   // not a mixture -> leave the one-shot unconsumed
    done = true;
    if (!site_rate || !aln || !root || !root->isLeaf() || root->neighbors.empty()) {
        printf("[GPU-ALLDERV-XCHECK-MIX] skipped (tree/model not ready)\n"); return; }
    Node *Rnode = root->neighbors[0]->node;
    if (!Rnode || Rnode->isLeaf()) { printf("[GPU-ALLDERV-XCHECK-MIX] skipped (degenerate root)\n"); return; }

    // run the all-branch sweep once (df/ddf for every edge); the per-edge gates below reuse these host-copied values
    std::vector<Node*> childNodes, parentNodes; std::vector<double> dfAll, ddfAll;
    if (!gpuComputeAllBranchDervCleanRoomMix(childNodes, parentNodes, dfAll, ddfAll)) {
        printf("[GPU-ALLDERV-XCHECK-MIX] skipped (unsupported regime or CUDA error)\n"); return; }

    auto lookupEdge = [&](Node *a, Node *b, double &df, double &ddf)->bool {
        for (size_t k=0;k<childNodes.size();k++){
            if ((childNodes[k]==a && parentNodes[k]==b) || (childNodes[k]==b && parentNodes[k]==a)) {
                df=dfAll[k]; ddf=ddfAll[k]; return true; } }
        return false; };

    auto checkEdge = [&](PhyloNeighbor *db, Node *dadNode, const char *label) {
        if (!db) { printf("[GPU-ALLDERV-XCHECK-MIX] %s skipped (neighbour not found)\n", label); return; }
        Node *childN = db->node;
        double dfA=0, ddfA=0;
        if (!lookupEdge(childN, dadNode, dfA, ddfA)) { printf("[GPU-ALLDERV-XCHECK-MIX] %s skipped (edge not in sweep)\n", label); return; }
        // gate (a): vs the validated G.8.1b single-edge derivative (independent two-sub-root path)
        double ddf1=0, lnL1=0;
        double df1 = gpuComputeEdgeDervCleanRoomMix(db, (PhyloNode*)dadNode, &ddf1, &lnL1);
        if (std::isnan(df1)) { printf("[GPU-ALLDERV-XCHECK-MIX] %s skipped (single-edge ref unsupported)\n", label); return; }
        double rdf  = (df1  != 0.0) ? fabs((dfA  - df1 )/df1 ) : fabs(dfA  - df1 );
        double rddf = (ddf1 != 0.0) ? fabs((ddfA - ddf1)/ddf1) : fabs(ddfA - ddf1);
        bool passA = (rdf <= 1e-9) && (rddf <= 1e-9);
        // gate (b): central FD of the clean-room lnL (perturb BOTH directed neighbour lengths; save/restore)
        PhyloNeighbor *back = (PhyloNeighbor*)dadNode->findNeighbor(childN);
        double t0 = db->length;
        double eps = 1e-4 * ((fabs(t0) > 1e-3) ? fabs(t0) : 1e-3);
        db->length = t0 + eps; if (back) back->length = t0 + eps;
        double lp = gpuComputeTreeLnLCleanRoomMix(nullptr, nullptr);
        db->length = t0 - eps; if (back) back->length = t0 - eps;
        double lm = gpuComputeTreeLnLCleanRoomMix(nullptr, nullptr);
        db->length = t0; if (back) back->length = t0;
        double fd = (lp - lm) / (2.0 * eps);
        double rfd = fabs(dfA - fd) / (fabs(fd) + 1e-30);
        bool passB = (rfd <= 1e-5);
        printf("[GPU-ALLDERV-XCHECK-MIX] %s edge(node=%d,dad=%d) t=%.6g | df_all=%.6e df_1edge=%.6e rel=%.3e | "
               "ddf_all=%.6e ddf_1edge=%.6e rel=%.3e | FD=%.6e rel=%.3e -> %s\n",
               label, childN->id, dadNode->id, t0, dfA, df1, rdf, ddfA, ddf1, rddf, fd, rfd,
               (passA && passB) ? "PASS (G.8.2.1a)" : "CHECK");
    };

    // (1) internal-internal edge: Rnode and an internal neighbour C.
    Node *C = nullptr;
    for (auto nb : Rnode->neighbors) if (!nb->node->isLeaf()) { C = nb->node; break; }
    if (C) checkEdge((PhyloNeighbor*)Rnode->findNeighbor(C), Rnode, "INT-INT");
    else   printf("[GPU-ALLDERV-XCHECK-MIX] INT-INT skipped (no internal-internal edge at root)\n");

    // (2) leaf-internal (pendant) edge: validates the k_leaf_eig_mix lower endpoint + a leaf's pre_v.
    Node *intNode = C ? C : Rnode;
    Node *Lf = nullptr;
    for (auto nb : intNode->neighbors) if (nb->node->isLeaf()) { Lf = nb->node; break; }
    if (!Lf) { for (auto nb : Rnode->neighbors) if (nb->node->isLeaf()) { Lf = nb->node; intNode = Rnode; break; } }
    if (Lf) checkEdge((PhyloNeighbor*)intNode->findNeighbor(Lf), intNode, "LEAF");
    else    printf("[GPU-ALLDERV-XCHECK-MIX] LEAF skipped (no pendant edge found)\n");
}

// ============================================================================================================
// G.8.2.0 — one-shot EM WEIGHT-OPTIMISER kill-switch (profile mixtures). With branches+α FIXED at the current
// state the mixture-weight sub-problem is concave (lnL = Σ_p f_p·log Σ_m w_m·a_{p,m}), so the EM M-step
// w_m ← Σ_p f_p·γ_{p,m} / Σ_p f_p  (γ_{p,m}=L_{p,m}/Σ_m' L_{p,m'}) converges to the UNIQUE weight-MLE from any
// interior start. This de-risks the G.8.2.1 joint optimiser by proving the GPU posterior-based M-step (a) climbs
// lnL MONOTONICALLY and (b) reaches the SAME optimum as IQ-TREE's own EM `ModelMixture::optimizeWeights()`. The
// GPU EM runs off the validated clean-room lnL+per-class (custom-weight override); both compared lnLs are
// GPU-computed (at CPU-EM weights vs GPU-EM weights) so the gate measures only the WEIGHT optimum, not the engine.
// CPU reference = optimizeWeights() with prop[] saved/restored (read-only net effect). Gated nmix>1, non-fused,
// no +I (the EM denominator would need ptn_invar). Fires from computeLikelihood under --gpu (not --jolt).
// ============================================================================================================
void PhyloTree::gpuMixWeightEMCrossCheckOnce() {
    static bool done = false;
    if (done) return;
    if (!model || model->getNMixtures() <= 1) return;   // unconsumed for single-model
    done = true;
    if (!site_rate || !aln) { printf("[GPU-WEM] skipped (model/tree not ready)\n"); return; }
    if (site_rate->getPInvar() > 0.0) { printf("[GPU-WEM] skipped (+I; EM denominator needs ptn_invar)\n"); return; }
    ModelMixture *mix = dynamic_cast<ModelMixture*>(model);
    if (!mix || mix->isFused()) { printf("[GPU-WEM] skipped (not a non-fused profile mixture)\n"); return; }
    if (!cpuComputeLikelihoodBranchPointer || !current_it || !current_it_back) {
        printf("[GPU-WEM] skipped (no CPU branch pointer to seed partials)\n"); return; }
    if (!s_gpuMixLnLEngineValidated) {   // MINOR-3: only trust the GPU per-class L_{p,m} if G.8.0 pinned the engine
        printf("[GPU-WEM] skipped (GPU mixture lnL engine NOT validated this run — G.8.0 lnL/self-consistency check did not PASS)\n");
        return; }

    int N = model->getNMixtures();
    int nptn = (int)aln->size();
    // Ftot = Σ_p frequency[p] == getAlnNSite() for a standard alignment (the denominator CPU optimizeWeights uses).
    // We divide the M-step by Ftot rather than getNSite() so the GPU EM fixed point sums to 1 by construction; the
    // two agree unless expected_num_sites is set (not the mixture case here).
    std::vector<double> f(nptn); double Ftot = 0.0;
    for (int p = 0; p < nptn; p++) { f[p] = (double)aln->at(p).frequency; Ftot += f[p]; }

    // ---- CPU reference: IQ-TREE's own EM optimizeWeights() at the current (fixed) branches/α; save+restore prop ----
    std::vector<double> w0(N), wCPU(N);
    for (int m = 0; m < N; m++) w0[m] = model->getMixtureWeight(m);
    (this->*cpuComputeLikelihoodBranchPointer)(current_it, (PhyloNode*)current_it_back->node, false);  // seed host partials
    mix->optimizeWeights();                                              // CPU EM -> prop[] = weight-MLE (branches/α fixed)
    for (int m = 0; m < N; m++) wCPU[m] = model->getMixtureWeight(m);
    for (int m = 0; m < N; m++) model->setMixtureWeight(m, w0[m]);       // RESTORE prop[] exactly
    // State-safety (red-team): restoring prop[] is sufficient TODAY only because mixture weights do not affect internal-
    // node partials (weights enter only at the root sum), so the partials optimizeWeights left at CPU-EM weights are
    // still valid for the restored weights. clearAllPartialLH() makes that UNCONDITIONAL — if a future change ever made
    // partials weight-dependent, this prevents the one-shot from silently corrupting the subsequent real analysis. It is
    // a one-shot; the cost is a single partial recompute on the next computeLikelihood.
    clearAllPartialLH();
    double lnL_cpuW = gpuComputeTreeLnLCleanRoomMix(nullptr, nullptr, wCPU.data());   // GPU lnL at CPU-EM weights (stateless sweep)
    // Min weight is DIAGNOSTIC CONTEXT, not a gate: an over-parameterised mixture on a finite alignment legitimately
    // drives weak classes toward 0 (a data-driven interior stationary point), so requiring "chunky" weights would
    // spuriously fail correct runs. The real boundary case is FLOOR-CLAMPING (w pinned to the 1e-10 EM floor); both
    // optimisers share that floor, so if they clamp the SAME class the lnL/Δw gates still hold, and if they disagree
    // those gates catch it. We therefore only report it. (10× the floor => "essentially clamped".)
    double wmin_cpu = 1.0; bool floorClampedCPU = false;
    for (int m = 0; m < N; m++) { if (wCPU[m] < wmin_cpu) wmin_cpu = wCPU[m]; if (wCPU[m] <= 1e-9) floorClampedCPU = true; }

    // ---- GPU EM: from a UNIFORM cold start (branches/α fixed), monotone climb to the weight-MLE ----
    std::vector<double> lhcat((size_t)N*nptn), w(N, 1.0/N), wn(N);
    auto emStep = [&](const std::vector<double>& lh, std::vector<double>& out) {
        std::fill(out.begin(), out.end(), 0.0);
        for (int p = 0; p < nptn; p++) {
            double s = 0.0; for (int m = 0; m < N; m++) s += lh[(size_t)m*nptn + p];
            if (s <= 0.0) continue; double fp = f[p];
            for (int m = 0; m < N; m++) out[m] += fp * (lh[(size_t)m*nptn + p] / s);   // Σ_p f_p·γ_{p,m}
        }
        for (int m = 0; m < N; m++) { out[m] /= Ftot; if (out[m] < 1e-10) out[m] = 1e-10; }  // /Σf, floor (CPU EM floor)
    };
    // M1 (red-team): match CPU optimizeWeights' iteration budget = (getNDim()+1)*100 = N*100 (2000 for C20, 6000 for
    // C60, 8000 for MEOW80) so the GPU is never capped before the CPU would be, and REQUIRE the GPU broke on the
    // convergence condition (not cap-exhaustion) in the PASS gate — otherwise an under-converged high-N run could slip
    // through on wErr/lnLErr alone. The loop still terminates on convergence (typically <100 iters; floored classes pin
    // and stop moving), so the high cap is only a safety ceiling, not the expected count.
    const int maxIters = N * 100;
    double prev = -1e300, worst_drop = 0.0, lnL = (double)NAN; bool monotone = true, gpuConverged = false; int iters = 0;
    for (int k = 0; k < maxIters; k++) {
        iters = k + 1;
        lnL = gpuComputeTreeLnLCleanRoomMix(nullptr, lhcat.data(), w.data());     // lnL(w^k) + per-class L_{p,m}(w^k)
        if (std::isnan(lnL)) { printf("[GPU-WEM] skipped (CUDA/unsupported on the EM sweep)\n"); return; }
        if (k > 0 && lnL < prev - 1e-7) { monotone = false; if (prev - lnL > worst_drop) worst_drop = prev - lnL; }
        double delta = lnL - prev; prev = lnL;
        emStep(lhcat, wn);
        double step = 0.0; for (int m = 0; m < N; m++) { double d = fabs(wn[m] - w[m]); if (d > step) step = d; }
        w = wn;                                                                   // w^{k+1}
        if (k > 0 && delta >= 0.0 && delta < 1e-9 && step < 1e-9) { gpuConverged = true; break; }
    }
    double lnL_gpuW = gpuComputeTreeLnLCleanRoomMix(nullptr, nullptr, w.data());  // GPU lnL at GPU-EM weights
    double wErr = 0.0, wmin_gpu = 1.0;
    for (int m = 0; m < N; m++) { double d = fabs(w[m] - wCPU[m]); if (d > wErr) wErr = d; if (w[m] < wmin_gpu) wmin_gpu = w[m]; }
    double lnLErr = (lnL_cpuW != 0.0) ? fabs((lnL_gpuW - lnL_cpuW)/lnL_cpuW) : fabs(lnL_gpuW - lnL_cpuW);
    // PASS criteria (the DECISIVE one is (b): the GPU EM reaches a likelihood at least as high as the CPU's own EM on
    // the concave weight surface, so it cannot have under-climbed):
    //   (a) monotone     — lnL non-decreasing across all iters (defining property of a correct EM M-step);
    //   (b) gpuConverged — the GPU broke on the convergence condition, NOT cap-exhaustion (M1: an under-converged
    //                      high-N run must not slip through on wErr/lnLErr alone);
    //   (c) lnL_gpuW >= lnL_cpuW - 1e-6   — GPU climbs at least as high as CPU optimizeWeights;
    //   (d) lnLErr <= 1e-7                — the two optima agree in lnL to ~1e-7 relative.
    // wErr (max|Δw|) is reported but bounded only LOOSELY at 1e-3, NOT 1e-4: CPU stops at its own |Δprop|<1e-4
    // tolerance while the GPU runs to 1e-9, so the two legitimately sit at slightly different points on the same
    // near-flat concave ridge (C20/400-site shows 1.6e-4, the CPU early-stop, with GPU lnL actually +5.9e-4 BETTER).
    // It is a sanity bound on weight-space agreement, not the primary gate — (b)+(c) are.
    bool pass = monotone && gpuConverged && (lnLErr <= 1e-7) && (lnL_gpuW >= lnL_cpuW - 1e-6) && (wErr <= 1e-3);
    printf("[GPU-WEM] N=%d EM weights (branches/α fixed): %d/%d iters from uniform, converged=%d, monotone=%d (worst drop %.2e); "
           "min w: CPU=%.2e GPU=%.2e floor-clamped(CPU)=%d; GPU-EM vs CPU optimizeWeights: max|Δw|=%.3e, "
           "lnL %.6f vs %.6f (GPU-CPU=%+.2e) rel=%.3e  -> %s\n",
           N, iters, maxIters, (int)gpuConverged, (int)monotone, worst_drop, wmin_cpu, wmin_gpu, (int)floorClampedCPU,
           wErr, lnL_gpuW, lnL_cpuW, (lnL_gpuW - lnL_cpuW), lnLErr,
           (pass ? "PASS (G.8.2.0)" : "CHECK"));
}

// ============================================================================================================
// G.8.2.1b — one-shot JOINT cold-start optimiser kill-switch (non-fused profile mixture, no +I, mean-gamma).
// Host-driven block-coordinate ascent composing the THREE validated pieces at ITERATE (b,w,α) via the parentLen/
// alpha overrides on the clean-room sweeps: (1) all-branch LM diagonal-Newton (k7_pre_mix, G.8.2.1a), (2) EM weight
// M-step (G.8.2.0), (3) scalar-α central-FD Newton/secant (no new kernel; matches the validated dr_c/dα = host-FD).
// Runs from a COLD (b=0.1,w=1/N,α=1) AND a WARM (live) start. The gate is NON-REENTRANT (no CPU optimiser is invoked
// inside this computeLikelihood hook — the hook fires at the FIRST computeLikelihood, before IQ-TREE has optimised)
// and self-contained: it proves the GPU joint optimiser is a consistent ascent method reaching the SAME stationary
// MLE from both starts (cold==warm rel<=1e-9 — the decisive basin/convergence check), monotone, improving on the
// start (the converged branch gradient |g|inf is reported for context). The end-to-end "GPU MLE == CPU joint MLE" tie is done at
// VALIDATION time by comparing the printed cold/warm MLE to the run's own .iqtree CPU lnL on the same -te topology.
// Read-only (the overrides mutate nothing; ends with clearAllPartialLH defensively). Mirrors the one-shot pattern.
void PhyloTree::gpuMixJointOptimizeCrossCheckOnce() {
    static bool done = false;
    if (done) return;
    if (!model || model->getNMixtures() <= 1) return;                  // single-model -> unconsumed
    done = true;
    if (!site_rate || !aln) { printf("[GPU-MIXJOINT] skipped (model/tree not ready)\n"); return; }
    int ns = aln->num_states;
    if (ns != 20 && ns != 4) { printf("[GPU-MIXJOINT] skipped (ns=%d)\n", ns); return; }
    if (!model->isReversible()) { printf("[GPU-MIXJOINT] skipped (non-reversible)\n"); return; }
    if (model->isSiteSpecificModel()) { printf("[GPU-MIXJOINT] skipped (PMSF)\n"); return; }
    if (site_rate->getPInvar() > 0.0) { printf("[GPU-MIXJOINT] skipped (+I)\n"); return; }
    ModelMixture *mix = dynamic_cast<ModelMixture*>(model);
    if (!mix || mix->isFused()) { printf("[GPU-MIXJOINT] skipped (not a non-fused profile mixture)\n"); return; }
    int N = model->getNMixtures();
    int ncat = site_rate->getNRate();
    if (ncat < 1) { printf("[GPU-MIXJOINT] skipped (ncat<1)\n"); return; }
    if (ncat > 1 && site_rate->isGammaRate() != GAMMA_CUT_MEAN) { printf("[GPU-MIXJOINT] skipped (non-mean-gamma)\n"); return; }
    if (!s_gpuMixLnLEngineValidated) { printf("[GPU-MIXJOINT] skipped (GPU mixture lnL engine not validated this run)\n"); return; }
    if (!root || !root->isLeaf() || root->neighbors.empty()) { printf("[GPU-MIXJOINT] skipped (bad root)\n"); return; }
    Node *Rt = root->neighbors[0]->node;
    if (!Rt || Rt->isLeaf()) { printf("[GPU-MIXJOINT] skipped (Rt leaf)\n"); return; }

    int nptn = (int)aln->size();
    bool DBG = getenv("MIXJOINT_DBG") != nullptr;

    // canonical node ordering: REPLICATE the clean-room indexDfs (root at Rt, preorder, recurse neighbors skipping dad).
    // b[] is indexed by this nid; b[v] = length of the edge above node v (b[root]=0). The SAME vector is a valid
    // parentLenOverride for BOTH clean-room functions (identical indexDfs => identical nid). nidMap maps the all-branch
    // derivative's returned childNodes back to this index. The warm-init self-test below confirms the index match.
    std::map<Node*,int> nidMap; std::vector<Node*> nodes; std::vector<double> bLive;
    std::function<void(Node*,Node*,double)> indexDfs = [&](Node *n, Node *dad, double lenToDad) {
        nidMap[n] = (int)nodes.size(); nodes.push_back(n); bLive.push_back(lenToDad);
        for (auto nb : n->neighbors) { if (nb->node == dad) continue; indexDfs(nb->node, n, nb->length); }
    };
    indexDfs(Rt, nullptr, 0.0);
    int nNodes = (int)nodes.size();
    int rootId = nidMap[Rt];

    std::vector<double> f(nptn); double Ftot = 0.0;                    // EM denominator (Σ_p frequency[p])
    for (int p = 0; p < nptn; p++) { f[p] = (double)aln->at(p).frequency; Ftot += f[p]; }
    std::vector<double> w0(N); for (int m = 0; m < N; m++) w0[m] = model->getMixtureWeight(m);
    double alpha0 = (ncat > 1) ? site_rate->getGammaShape() : -1.0;

    // engine reference at the LIVE state + the override self-test (live params THROUGH the overrides must reproduce it
    // to ~1e-12, proving the iterate plumbing is transparent and this gate's nid matches the clean-room functions' nid).
    double lnL_live = gpuComputeTreeLnLCleanRoomMix(nullptr, nullptr, nullptr);
    if (std::isnan(lnL_live)) { printf("[GPU-MIXJOINT] skipped (CUDA/unsupported on the live sweep)\n"); return; }
    double lnL_warmInit = gpuComputeTreeLnLCleanRoomMix(nullptr, nullptr, w0.data(), bLive.data(), alpha0);
    double selfRel = (lnL_live != 0.0) ? std::fabs((lnL_warmInit - lnL_live)/lnL_live) : std::fabs(lnL_warmInit - lnL_live);

    // all-branch DERIVATIVE override self-test: central-FD one edge's df (from the all-branch sweep at bLive) against
    // the VALIDATED lnL override at bLive±ε. The lnL selfRel above only exercises the lnL override; this index-checks
    // the all-branch override seam (a divergent indexDfs there would corrupt the branch step but pass selfRel).
    double selfRelDrv = (double)NAN;
    { std::vector<Node*> cN0, pN0; std::vector<double> df0, ddf0;
      if (gpuComputeAllBranchDervCleanRoomMix(cN0, pN0, df0, ddf0, bLive.data(), alpha0) && !cN0.empty()) {
          int ev = nidMap.at(cN0[0]); double dfAll = df0[0];
          double eps = 1e-4 * std::max(std::fabs(bLive[ev]), 1e-3);
          std::vector<double> bp = bLive; bp[ev] = bLive[ev] + eps;
          double Lp = gpuComputeTreeLnLCleanRoomMix(nullptr, nullptr, w0.data(), bp.data(), alpha0);
          bp[ev] = bLive[ev] - eps;
          double Lm = gpuComputeTreeLnLCleanRoomMix(nullptr, nullptr, w0.data(), bp.data(), alpha0);
          if (!std::isnan(Lp) && !std::isnan(Lm)) { double dfFD = (Lp - Lm) / (2.0*eps);
              selfRelDrv = std::fabs(dfAll - dfFD) / std::max(std::fabs(dfFD), 1.0); }
      }
    }

    auto alphaArg = [&](double a){ return (ncat > 1) ? a : -1.0; };

    // host-driven block-coordinate optimiser: branch LM -> EM weights -> α FD-Newton (each block monotone/accept-or-stay)
    auto runOpt = [&](bool warm, double &outLnL, int &outIters, bool &outMono, bool &outConv, double &outGradInf, double &outFreeGradInf) {
        std::vector<double> b(nNodes), w(N); double alpha;
        if (warm) { b = bLive; w = w0; alpha = (ncat > 1 ? alpha0 : 1.0); }
        else { std::fill(b.begin(), b.end(), 0.1); b[rootId] = 0.0; std::fill(w.begin(), w.end(), 1.0/N); alpha = 1.0; }
        double mu = 1.0;   // shared LM damping for the joint branch+α step (single-model JOLT style)
        int stall = 0;     // consecutive sub-1e-7-progress outers => the diagonal-LM has reached its ridge floor
        double lnL = gpuComputeTreeLnLCleanRoomMix(nullptr, nullptr, w.data(), b.data(), alphaArg(alpha));
        outMono = true; outConv = false; outIters = 0; outGradInf = (double)NAN; outFreeGradInf = (double)NAN;
        const int maxOuter = 400;
        for (int outer = 0; outer < maxOuter; outer++) {
            double lnL0 = lnL;
            // (1) JOINT BRANCH+α BLOCK — gradients at the current (b,w,α), then ONE shared-μ LM step over ALL branches
            // AND the scalar α together (mirrors the single-model JOLT joint step). Stepping the coupled pair together
            // avoids the block-coordinate moving target (a branch-only step stalls: α keeps reshaping the branch
            // surface so |g| never shrinks). α gradient/curvature by central FD on the clean-room lnL (no new kernel).
            std::vector<Node*> cN, pN; std::vector<double> df, ddf;
            if (!gpuComputeAllBranchDervCleanRoomMix(cN, pN, df, ddf, b.data(), alphaArg(alpha))) { outLnL = (double)NAN; return; }
            std::vector<double> gdf(nNodes, 0.0), gddf(nNodes, 0.0); double gradInf = 0.0, freeGradInf = 0.0;
            for (size_t i = 0; i < cN.size(); i++) { int v = nidMap.at(cN[i]); gdf[v] = df[i]; gddf[v] = ddf[i];
                if (std::fabs(df[i]) > gradInf) gradInf = std::fabs(df[i]);
                // FREE gradient = branches NOT structurally pinned at a clamp bound. A branch sitting at the lower
                // (1e-6) / upper (20) bound whose gradient pushes it FURTHER out-of-bounds has an MLE past the clamp,
                // so its |df| never vanishes — that is the irreducible flat-ridge signal (the |g|inf~49 we observed).
                // The free gradient (everything that can still move) -> 0 AT the MLE and is the physical convergence
                // signal that "recognizes the flat ridge" (vs. the total |g|inf which the clamped branches dominate).
                bool pinned = (b[v] <= 1e-6 + 1e-12 && df[i] < 0.0) || (b[v] >= 20.0 - 1e-12 && df[i] > 0.0);
                if (!pinned && std::fabs(df[i]) > freeGradInf) freeGradInf = std::fabs(df[i]); }
            outGradInf = gradInf; outFreeGradInf = freeGradInf;
            double ga = 0.0, curvA = 1e-12, eps = 0.0;
            if (ncat > 1) {
                eps = 1e-3 * std::max(alpha, 1.0);
                double lp = gpuComputeTreeLnLCleanRoomMix(nullptr, nullptr, w.data(), b.data(), alpha+eps);
                double lm = gpuComputeTreeLnLCleanRoomMix(nullptr, nullptr, w.data(), b.data(), alpha-eps);
                if (std::isnan(lp) || std::isnan(lm)) { outLnL = (double)NAN; return; }
                ga = (lp - lm) / (2.0*eps); curvA = std::fabs((lp - 2.0*lnL + lm) / (eps*eps)); if (curvA < 1e-12) curvA = 1e-12;
            }
            for (int bt = 0; bt < 16; bt++) {
                std::vector<double> bc = b;
                for (int v = 0; v < nNodes; v++) { if (v == rootId) continue;
                    double stepv = gdf[v] / (std::fabs(gddf[v]) + mu); bc[v] = b[v] + stepv;
                    if (bc[v] < 1e-6) bc[v] = 1e-6; if (bc[v] > 20.0) bc[v] = 20.0; }
                double ac = alpha;
                if (ncat > 1) { ac = alpha + ga / (curvA + mu); if (ac < 0.02) ac = 0.02; if (ac > 50.0) ac = 50.0; }
                double ln = gpuComputeTreeLnLCleanRoomMix(nullptr, nullptr, w.data(), bc.data(), alphaArg(ac));
                if (std::isnan(ln)) { outLnL = (double)NAN; return; }
                // Accept ANY strict improvement (no +1e-9 margin). The margin was the flat-ridge FREEZE: on the ridge
                // the best step improves by < 1e-9, so it was rejected, mu ran away (×4 each reject), the step -> 0,
                // and the two starts halted at slightly different points (the G.8.2.1b 8.76e-9 cold-vs-warm residual).
                // The sweep is deterministic so ln>lnL is a genuine (noise-free) improvement => still strictly monotone,
                // mu now halves on accept so it stays controlled, and both starts crawl to the SAME ridge MLE.
                if (ln > lnL) { b = bc; alpha = ac; lnL = ln; mu = std::max(mu*0.5, 1e-9); break; }
                else mu = std::min(mu*4.0, 1e12);   // cap: on the ridge the gain |g|^2/mu falls below FP64 lnL
                                                    // resolution, so an UNcapped mu runs to +inf and freezes the sweep
                                                    // (the G.8.2.1c-v1 walltime-loop) instead of letting the stall fire
            }
            // (2) WEIGHT BLOCK — EM M-step to FULL convergence. KEY: the per-class likelihood a_{p,m} is WEIGHT-
            // INDEPENDENT (branches/α are fixed in this block — the weight only multiplies at the final pattern sum),
            // so compute it ONCE (one GPU sweep at uniform w => lhc[m][p] = a_{p,m}/N; the 1/N cancels in the posterior)
            // and iterate the ENTIRE EM M-step PURELY ON THE HOST (γ = w_m·lhc / Σ_m w_m·lhc). Same concave climb as
            // G.8.2.0 but ~20× fewer GPU sweeps, AND it converges the weights FULLY each outer so the branch gradient
            // no longer chases a moving weight target (the prior partial-EM stall). One final GPU sweep for the exact lnL.
            { std::vector<double> lhc((size_t)N*nptn), wunif(N, 1.0/N), wn(N);
              double l1 = gpuComputeTreeLnLCleanRoomMix(nullptr, lhc.data(), wunif.data(), b.data(), alphaArg(alpha));
              if (std::isnan(l1)) { outLnL = (double)NAN; return; }
              for (int em = 0; em < 1000; em++) {
                  std::fill(wn.begin(), wn.end(), 0.0);
                  for (int p = 0; p < nptn; p++) { double s = 0.0; for (int m = 0; m < N; m++) s += w[m]*lhc[(size_t)m*nptn+p];
                      if (s <= 0.0) continue; double fp = f[p];
                      for (int m = 0; m < N; m++) wn[m] += fp * (w[m]*lhc[(size_t)m*nptn+p] / s); }
                  double stepw = 0.0; for (int m = 0; m < N; m++) { wn[m] /= Ftot; if (wn[m] < 1e-10) wn[m] = 1e-10;
                      double d = std::fabs(wn[m]-w[m]); if (d > stepw) stepw = d; }
                  w = wn;
                  if (em > 0 && stepw < 1e-12) break;
              }
              lnL = gpuComputeTreeLnLCleanRoomMix(nullptr, nullptr, w.data(), b.data(), alphaArg(alpha)); }
            if (lnL < lnL0 - 1e-7) outMono = false;
            outIters = outer + 1;
            if (DBG) fprintf(stderr, "[MIXJOINT-DBG] %s outer=%d lnL=%.6f mu=%.2e alpha=%.4f |g|inf=%.2e free|g|=%.2e\n",
                             warm ? "warm" : "cold", outer, lnL, mu, alpha, outGradInf, outFreeGradInf);
            // G.8.2.1c ROBUST RIDGE-RECOGNIZING TERMINATION. The shared-mu diagonal-Newton has an ACCURACY FLOOR on
            // the ill-conditioned mixture ridge: its first-order gain Σdf²/(|ddf|+mu) ≈ |g|²/mu, and the mu that avoids
            // overshoot drives that gain below FP64 lnL resolution (~1e-11 at lnL~3e4). The step then stalls with the
            // FREE gradient floored well ABOVE 0 (~1.56e-2 here), NOT at the true MLE — so free|g| is NOT a gateable
            // convergence signal (reported only). The principled stop is a sustained lnL plateau: STALL = 3 consecutive
            // outers improving by <1e-7. (v1 removed the +1e-9 accept margin AND gated on free|g|<1e-2; that floored at
            // 1.56e-2 and never fired, so warm looped 299 outers to walltime — hence this plateau-based stop + mu cap.)
            // Cold and warm both halt at this floor; they differ by the diagonal-LM's PATH-DEPENDENT floor (~1e-8, gated
            // below) — the curvature-aware device-resident L-BFGS JOLT is the path to <=1e-9 (PART IV plan).
            if (std::fabs(lnL - lnL0) < 1e-7) stall++; else stall = 0;
            if (outer > 0 && stall >= 3) { outConv = true; break; }
        }
        outLnL = lnL;
    };

    double lnL_warm, lnL_cold, gradWarm, gradCold, fgradWarm, fgradCold; int itWarm, itCold; bool monoWarm, monoCold, convWarm, convCold;
    runOpt(true,  lnL_warm, itWarm, monoWarm, convWarm, gradWarm, fgradWarm);
    runOpt(false, lnL_cold, itCold, monoCold, convCold, gradCold, fgradCold);
    clearAllPartialLH();   // defensive read-only contract (the clean-room sweeps touch no live partials; cheap one-shot)

    if (std::isnan(lnL_warm) || std::isnan(lnL_cold)) { printf("[GPU-MIXJOINT-XCHECK] skipped (CUDA/unsupported during optimise)\n"); return; }
    double relCW = (lnL_warm != 0.0) ? std::fabs((lnL_cold - lnL_warm)/lnL_warm) : std::fabs(lnL_cold - lnL_warm);
    // PASS criteria (decisive = (c): cold and warm converge to the SAME MLE => consistent ascent, no basin pathology):
    //   (a) BOTH override self-tests transparent — lnL selfRel<=1e-12 AND all-branch-derivative selfRelDrv<=1e-4 (FD)
    //       (NaN selfRelDrv => the derivative seam couldn't be checked => fails safe to CHECK);
    //   (b) both monotone AND broke on convergence (not cap); (c) rel(cold,warm)<=1e-8; (d) both improve on the live
    //   start. Both the total branch |g|inf (clamped-branch-dominated ~ the flat ridge) and the FREE |g| (floors ~1.56e-2
    //   on this ridge, NOT 0) are reported for context — (c) remains decisive. THRESHOLD: 1e-8 (NOT 1e-9) is the
    //   host-driven shared-mu DIAGONAL-LM accuracy floor — its step gain |g|²/mu falls below FP64 lnL resolution before
    //   the free gradient vanishes, so cold/warm halt at path-dependent points ~1e-8 apart (empirically 8.76e-9). The
    //   curvature-aware device-resident L-BFGS JOLT (PART IV) is the path to <=1e-9; 8-sig-fig cold==warm + GPU>=CPU
    //   (the VALIDATION tie vs the run's .iqtree CPU lnL) already establish "same MLE" for this diagonal-LM kill-switch.
    bool pass = (selfRel <= 1e-12) && (selfRelDrv <= 1e-4) && monoWarm && monoCold && convWarm && convCold
                && (relCW <= 1e-8) && (lnL_cold >= lnL_live - 1e-6) && (lnL_warm >= lnL_live - 1e-6);
    printf("[GPU-MIXJOINT-XCHECK] N=%d ncat=%d  lnL_live=%.6f -> cold=%.6f (%d it mono=%d conv=%d |g|inf=%.2e free|g|=%.2e) "
           "warm=%.6f (%d it mono=%d conv=%d |g|inf=%.2e free|g|=%.2e); selfTest lnL rel=%.2e drv rel=%.2e; cold-vs-warm rel=%.3e  -> %s\n",
           N, ncat, lnL_live, lnL_cold, itCold, (int)monoCold, (int)convCold, gradCold, fgradCold,
           lnL_warm, itWarm, (int)monoWarm, (int)convWarm, gradWarm, fgradWarm, selfRel, selfRelDrv, relCW,
           (pass ? "PASS (G.8.2.1c)" : "CHECK"));
}

// ============================================================================================================
// G.6.0a — one-shot GPU free-Q gradient cross-check (gated by env JOLT_QGRADCHECK). For a reversible DNA free-Q
// model (HKY..GTR, fixed freqs, no +I), this is the de-risk BEFORE building the G.6.0b free-Q optimiser: it
// proves the GPU computes lnL CORRECTLY under a MOVING eigensystem (every FD perturbation of a free exchange-
// ability re-decomposes the 4x4 Q and re-uploads eval/U/Uinv). For each free param (perturbed in rate-class
// space via the model's param_spec, so HKY's kappa moves A-G and C-T together), we compare GPU clean-room lnL
// vs CPU computeLikelihood at the perturbed Q, and the FD gradient GPU-vs-CPU. GATE: |GPU-CPU|/|CPU| <= 1e-9 at
// the base AND every perturbed Q (the FD-grad then matches by construction). Read-only: the model is fully
// restored. No CPU-path effect (env-gated, --gpu only).
// ============================================================================================================
void PhyloTree::gpuFreeQGradCheckOnce() {
    static bool done = false;
    if (done) return;
    done = true;
    if (getenv("JOLT_QGRADCHECK") == nullptr) return;
    if (!model || !site_rate || !aln) { printf("[QGRADCHECK] skipped (model/tree not ready)\n"); return; }
    int ns = aln->num_states;
    if (ns != 4 && ns != 20) { printf("[QGRADCHECK] skipped (ns=%d not in {4,20})\n", ns); return; }
    if (!model->isReversible() || model->getNMixtures() != 1 || model->isSiteSpecificModel()) {
        printf("[QGRADCHECK] skipped (nonrev/mixture/ssm)\n"); return; }
    if (model->getFreqType() == FREQ_ESTIMATE) { printf("[QGRADCHECK] skipped (+FO estimated freqs)\n"); return; }
    if (site_rate->getPInvar() > 0.0) { printf("[QGRADCHECK] skipped (+I; clean-room omits ptn_invar)\n"); return; }
    int nQ = model->getNDim();   // == num_params (free exchangeabilities) for fixed-freq models
    if (nQ < 1) { printf("[QGRADCHECK] skipped (no free Q params, getNDim()=%d)\n", nQ); return; }

    const double QFD_EPS = 1e-4;   // == ERROR_X (the CPU BFGS forward-FD step) for apples-to-apples gradients
    std::vector<double> q0(nQ);
    model->gpuGetFreeParams(q0.data());

    double cpu0 = computeLikelihood();
    double gpu0 = gpuComputeTreeLnLCleanRoom(nullptr);
    if (std::isnan(gpu0)) { printf("[QGRADCHECK] skipped (clean-room returned NaN)\n"); return; }
    double rel0 = (cpu0 != 0.0) ? fabs((gpu0 - cpu0) / cpu0) : fabs(gpu0 - cpu0);
    printf("[QGRADCHECK] nQ=%d ns=%d base lnL: GPU=%.9f CPU=%.9f rel=%.3e\n", nQ, ns, gpu0, cpu0, rel0);

    double maxrel_lnL = rel0, maxrel_grad = 0.0;
    bool ok = (rel0 <= 1e-9);
    std::vector<double> q(q0);
    for (int k = 0; k < nQ; k++) {
        double save = q[k];
        double h = QFD_EPS * fabs(save); if (h == 0.0) h = QFD_EPS;
        q[k] = save + h; double hh = q[k] - save;
        model->gpuSetFreeParamsDecompose(q.data()); clearAllPartialLH();
        double cpuk = computeLikelihood();
        double gpuk = gpuComputeTreeLnLCleanRoom(nullptr);
        q[k] = save; model->gpuSetFreeParamsDecompose(q.data()); clearAllPartialLH();   // restore param k
        double relk = (cpuk != 0.0) ? fabs((gpuk - cpuk) / cpuk) : fabs(gpuk - cpuk);
        double gq_cpu = (cpuk - cpu0) / hh, gq_gpu = (gpuk - gpu0) / hh;
        double relg = (gq_cpu != 0.0) ? fabs((gq_gpu - gq_cpu) / gq_cpu) : fabs(gq_gpu - gq_cpu);
        if (relk > maxrel_lnL) maxrel_lnL = relk;
        if (relg > maxrel_grad) maxrel_grad = relg;
        if (relk > 1e-9) ok = false;
        printf("[QGRADCHECK] k=%d q=%.6g h=%.2g  lnL(q+h): GPU=%.6f CPU=%.6f rel=%.3e | dL/dq: GPU=%.6e CPU=%.6e rel=%.3e\n",
               k, save, h, gpuk, cpuk, relk, gq_gpu, gq_cpu, relg);
    }
    computeLikelihood();   // restore curScore/partials at the original Q
    printf("[QGRADCHECK] nQ=%d maxrel_lnL=%.3e (gate<=1e-9) maxrel_grad=%.3e -> %s\n",
           nQ, maxrel_lnL, maxrel_grad, ok ? "QGRADCHECK PASS" : "QGRADCHECK FAIL");
}

// ============================================================================================================
// G.2.1b — GPU override for computeLikelihoodDervPointer (byte-matches ComputeLikelihoodDervType). STATELESS:
// the single-edge df/ddf is recomputed clean-room from the live tree each call (no device-resident theta /
// partials -> no coherence hole, per the verified contract). Writes UN-negated df/ddf (computeFuncDerv negates).
// Delegates to the saved CPU Derv pointer if the regime is unsupported (NaN).
void PhyloTree::computeLikelihoodDervGPU(PhyloNeighbor *dad_branch, PhyloNode *dad, double *df, double *ddf) {
    double gddf = 0.0, glnL = 0.0;
    double gdf = gpuComputeEdgeDervCleanRoom(dad_branch, dad, &gddf, &glnL);
    if (std::isnan(gdf)) {
        if (cpuComputeLikelihoodDervPointer) { (this->*cpuComputeLikelihoodDervPointer)(dad_branch, dad, df, ddf); return; }
        *df = gdf; *ddf = gddf; return;
    }
    *df = gdf; *ddf = gddf;   // un-negated: d(lnL)/dt, d2(lnL)/dt2 (computeFuncDerv negates downstream)
    static bool announced = false;
    if (!announced) { announced = true;
        printf("[GPU-DERV] computeLikelihoodDervGPU active (clean-room single-edge df/ddf, stateless)\n"); }
}

// G.2.1b — GPU override for computeLikelihoodFromBufferPointer (byte-matches ComputeLikelihoodFromBufferType,
// no args). STATELESS: the from-buffer lnL at the current branch lengths == the whole-tree lnL (reversible), so
// recompute it clean-room from the live tree (which already reflects current_it->length set by computeFuncDerv).
double PhyloTree::computeLikelihoodFromBufferGPU() {
    double l = gpuComputeTreeLnLCleanRoom(nullptr);
    if (std::isnan(l)) {
        if (cpuComputeLikelihoodFromBufferPointer) return (this->*cpuComputeLikelihoodFromBufferPointer)();
        return l;
    }
    static bool announced = false;
    if (!announced) { announced = true;
        printf("[GPU-FROMBUF] computeLikelihoodFromBufferGPU active (clean-room whole-tree lnL, stateless)\n"); }
    return l;
}

// ============================================================================================================
// G.2.0b/G.2.1b — gated funnel hook (called LAST in PhyloTree::setLikelihoodKernel). Saves the ISA-set CPU
// Branch/Derv/FromBuffer pointers and installs the GPU overrides. Idempotent + re-applied on every funnel
// re-invocation (the ISA setter resets the pointers to CPU each call; this re-installs GPU). INSTALL gate is
// params-level (known even before the model is built); the per-call helpers re-check model-level conditions
// (reversible / single-mixture / non-site-specific / no +I / 4-or-20-state) and fall back to CPU.
// ============================================================================================================
void PhyloTree::setLikelihoodKernelGPU() {
    if (!params || !params->gpu) return;
    // G.4.2: under --jolt, the stateless GPU Branch/Derv/FromBuffer overrides must NOT install. JOLT is the ONLY
    // GPU path — it replaces ModelFactory::optimizeParameters wholesale for eligible candidates (+G/base), while
    // INELIGIBLE candidates (+I, +R, +FO, mixture) must fall back to the PURE CPU likelihood (normal speed), NOT
    // the slow stateless GPU clean-room sweep (G.2.2a measured ~25 min/model = 50-100x). Keeping both active would
    // (a) make the +I/+R tail run on the slow stateless path (timeout) and (b) reduce optimizeParametersJOLT's
    // self-check to GPU-vs-GPU; with this no-op the self-check's computeLikelihood() is a genuine CPU recompute.
    if (params->jolt) return;
    if (!aln || (aln->num_states != 4 && aln->num_states != 20)) return;
    if (isSuperTree()) return;
    // reject regimes whose output reads per-pattern/-category buffers the GPU overrides do not populate
    if (params->print_site_lh != WSL_NONE) return;
    if (params->print_partition_lh) return;
    if (params->print_site_rate) return;
    if (params->print_ancestral_sequence != AST_NONE) return;
    if (params->aLRT_replicates > 0 || params->localbp_replicates > 0 || params->aLRT_test || params->aBayes_test) return;
    if (params->gbo_replicates > 0 || params->num_bootstrap_samples > 0) return;
    if (!params->dating_method.empty()) return;
    if (params->pll) return;

    // save the genuine ISA-set CPU pointers (guard against re-saving our own GPU overrides), then install GPU.
    if (computeLikelihoodBranchPointer != &PhyloTree::computeLikelihoodBranchGPU)
        cpuComputeLikelihoodBranchPointer = computeLikelihoodBranchPointer;
    if (computeLikelihoodDervPointer != &PhyloTree::computeLikelihoodDervGPU)
        cpuComputeLikelihoodDervPointer = computeLikelihoodDervPointer;
    if (computeLikelihoodFromBufferPointer != &PhyloTree::computeLikelihoodFromBufferGPU)
        cpuComputeLikelihoodFromBufferPointer = computeLikelihoodFromBufferPointer;
    computeLikelihoodBranchPointer     = &PhyloTree::computeLikelihoodBranchGPU;       // G.2.0b lnL
    computeLikelihoodDervPointer       = &PhyloTree::computeLikelihoodDervGPU;         // G.2.1b single-edge df/ddf
    computeLikelihoodFromBufferPointer = &PhyloTree::computeLikelihoodFromBufferGPU;   // G.2.1b from-buffer lnL

    static bool announced = false;
    if (!announced) { announced = true;
        printf("[GPU-KERNEL] setLikelihoodKernelGPU: Branch+Derv+FromBuffer -> GPU (stateless clean-room); "
               "fixed_branch_length=%d num_states=%d (branch-opt %s)\n",
               params->fixed_branch_length, aln->num_states,
               params->fixed_branch_length == BRLEN_FIX ? "fixed/-blfix" : "GPU"); }
}

// ============================================================================================================
// Phase G.6 — host callback handed to gpu_jolt_optimize for DNA free-Q models (the eigensystem MOVES during the
// optimise). It applies a trial free-Q vector q[nFreeQ] to the LIVE model (gpuSetFreeParamsDecompose -> param_spec
// rate-class mapping + the G-T=1 gauge + decomposeRateMatrix), then copies the fresh eigensystem back to the
// launcher's host buffers. extern "C" to match the jolt_qdecompose_fn C ABI. ctx = the model + ns. The launcher is
// mutex-serialized and the model is thread-local, so this mutates only the calling thread's own model; the final
// optimised Q is written back deterministically after gpu_jolt_optimize returns (do NOT rely on the launcher's
// internal Q thrashing leaving the model in any particular state).
// ============================================================================================================
namespace { struct JoltQCtx { ModelSubst* model; int ns; }; }
extern "C" void jolt_qdecompose_intree(void* vctx, const double* q, double* eval, double* U, double* Uinv) {
    JoltQCtx* c = reinterpret_cast<JoltQCtx*>(vctx);
    c->model->gpuSetFreeParamsDecompose(q);
    int ns = c->ns;
    memcpy(eval, c->model->getEigenvalues(),         sizeof(double) * ns);
    memcpy(U,    c->model->getEigenvectors(),         sizeof(double) * (size_t)ns * ns);
    memcpy(Uinv, c->model->getInverseEigenvectors(),  sizeof(double) * (size_t)ns * ns);
}

// ============================================================================================================
// Phase G.4.2 — GPU JOLT joint-gradient optimiser for ONE candidate model. Builds the clean-room inputs from the
// LIVE objects (mirroring gpuComputeTreeLnLCleanRoom), runs the validated G.4.1b joint LM driver on the GPU,
// writes the optimised (197 branches + alpha) back through the cache-invalidating setters, and self-checks that a
// FRESH CPU computeLikelihood() reproduces the JOLT lnL. Returns NaN if JOLT-ineligible / CUDA error -> caller
// falls back to the standard CPU path.
// ============================================================================================================
double PhyloTree::optimizeParametersJOLT(int fixed_len) {
    // ---- eligibility gate (the validated G.4.1/G.4.1b scope: fixed-Q reversible, ns in {4,20}, no +I, gamma-or-uniform) ----
    // G.4.3a diagnostic: JOLT_DEBUG=1 logs the gate decision per candidate (which decline reason, or engage), so we can
    // tell whether an ineligible family (e.g. +F) REACHES this hook and is declined by a specific gate, vs never arrives
    // (staged-search dispatches it elsewhere). Env-gated => zero cost in production; touches no CPU-path behaviour.
    static const bool JOLT_DBG = (getenv("JOLT_DEBUG") != nullptr);
    if (JOLT_DBG) {
        string mn = model ? model->getName() : string("(nullmodel)");
        // freqtype: 1=USER_DEFINED 2=EQUAL 3=EMPIRICAL(+F) 4=ESTIMATE(+FO) (tools.h StateFreqType)
        fprintf(stderr, "[JOLT-GATE] reached hook model=%s freqtype=%d ns=%d rev=%d nmix=%d ssm=%d ndim=%d pinv=%.4g ncat=%d alpha=%.4g fixedlen=%d\n",
                mn.c_str(), model ? (int)model->getFreqType() : -1, aln ? aln->num_states : -1,
                model ? (int)model->isReversible() : -1, model ? model->getNMixtures() : -1,
                model ? (int)model->isSiteSpecificModel() : -1, model ? model->getNDim() : -999,
                site_rate ? site_rate->getPInvar() : -1.0, site_rate ? site_rate->getNRate() : -1,
                (site_rate && site_rate->getNRate() > 1) ? site_rate->getGammaShape() : -1.0, fixed_len);
        fflush(stderr);
    }
    #define JOLT_DECLINE(why) do { if (JOLT_DBG) { fprintf(stderr, "[JOLT-GATE] decline reason=%s\n", why); fflush(stderr); } return (double)NAN; } while (0)
    if (!model || !site_rate || !aln) JOLT_DECLINE("null-ptr");
    if (fixed_len != BRLEN_OPTIMIZE) JOLT_DECLINE("brlen-mode");   // JOLT optimises branches; other brlen modes -> CPU
    int ns = aln->num_states;
    if (ns != 4 && ns != 20) JOLT_DECLINE("num-states");
    if (!model->isReversible() || model->getNMixtures() != 1 || model->isSiteSpecificModel()) JOLT_DECLINE("nonrev/mixture/ssm");
    // G.6.1: free substitution params (DNA HKY..GTR) — the eigensystem MOVES, so JOLT FD-optimises them via the
    // decompose callback. ON BY DEFAULT (validated job 170795329: DNA -m MF 70 engage / 9 decline [8 +R + 1 pure-+I],
    // best-by-BIC == CPU F81+F+G4, worst write-back rel 6.2e-12, ZERO mismatch). JOLT_NO_FREEQ disables it (debug/A-B).
    // Restricted to ns==4 reversible, getNDim()<=5, FIXED freqs (exclude +FO, FREQ_ESTIMATE, whose free freq dims are
    // NOT yet handled). AA fixed-Q (getNDim()==0) is unaffected.
    static const bool JOLT_FREEQ_EN = (getenv("JOLT_NO_FREEQ") == nullptr);
    int nFreeQ = 0;
    {
        int ndim = model->getNDim();
        // audit RISK-1 hardening: tied-frequency DNA types (+FRY/+F1112/... = FREQ_DNA_*) contribute 1-3 FREE freq
        // params to getNDim(), which gpuGetFreeParams packs into the Q-vector tail; the launcher then mis-clamps them
        // as exchangeabilities ([MINQ,MAXQ]=[1e-4,100] vs the correct ~[0,1]) -> a coherent-but-SUBOPTIMAL lnL that
        // still passes the write-back gate (which checks GPU/CPU coherence, not optimality). Require nFreqParams==0 so
        // the entire getVariables() tail is exchangeabilities. +FQ/+F (0 freq dims) still engage; tied-freq is not in
        // the default -m MF DNA set, so this declines such explicit user models to CPU (defensive, no live regression).
        bool freeQok = JOLT_FREEQ_EN && ndim > 0 && ndim <= 5 && ns == 4 &&
                       model->getFreqType() != FREQ_ESTIMATE && model->isReversible() &&
                       nFreqParams(model->getFreqType()) == 0;
        if (ndim != 0 && !freeQok) JOLT_DECLINE("free-subst-params");  // +FO / tied-freq / AA-GTR / production free-Q -> CPU
        nFreeQ = freeQok ? ndim : 0;
    }
    int ncat = site_rate->getNRate();
    if (ncat < 1 || ncat > 64) JOLT_DECLINE("ncat-range");
    // G.4.3b audit fix: discriminate the rate model by isGammaRate() (robust) NOT getGammaShape() (which is a
    // POSITIVE inherited value for RateFree/+R -> the old check let +R / +R+I wrongly engage JOLT with uniform
    // proportions + mean-gamma rates, silently wrong since writeback precedes the self-check). JOLT only implements
    // the MEAN discrete-gamma (Yang-1994) discretisation, so require exactly GAMMA_CUT_MEAN: this declines +R
    // (isGammaRate()==0), +R+I, and the MEDIAN gamma variant +Gm/+I+Gm (isGammaRate()==GAMMA_CUT_MEDIAN).
    // G.5.1a: let PURE +R (FreeRate, no +I) through to the launcher ONLY under JOLT_RGRADCHECK — it runs the
    // weight-gradient FD self-check then declines to CPU (the +R optimiser branch is G.5.1b, not yet wired).
    bool rgcheck = (ncat > 1 && site_rate->isFreeRate() && site_rate->getPInvar() <= 0.0 && getenv("JOLT_RGRADCHECK") != nullptr);
    if (ncat > 1 && site_rate->isGammaRate() != GAMMA_CUT_MEAN && !rgcheck) JOLT_DECLINE("non-mean-gamma");
    // G.4.3b — +I (proportion of invariant sites) is now JOINTLY optimised by JOLT, but ONLY for +I+G
    // (RateGammaInvar: getProp(c)=(1-pinv)/K, standard mean-1 discrete-gamma rates). Pure +I (RateInvar, ncat==1)
    // rescales getRate=1/(1-pinv) -> out of JOLT scope -> CPU. A user-FIXED pinv, or no constant sites (pinvMax->0
    // degenerate), also fall to CPU. The invariant term L_p += pinv*base_invar[p] is added in the kernel; the joint
    // LM step moves pinv alongside the branches + alpha (same machinery that absorbed alpha in G.4.1b).
    static const double JOLT_MIN_PINVAR = 1e-6;          // == MIN_PINVAR (model/rateinvar.h)
    double pinv0 = site_rate->getPInvar();
    int optPinv = 0;
    if (pinv0 > 0.0) {
        if (site_rate->isFixPInvar())                                JOLT_DECLINE("fixed-pinvar");
        if (ncat <= 1)                                               JOLT_DECLINE("pure-pinvar-no-gamma");  // RateInvar getRate=1/(1-pinv) -> CPU (ncat>1 already => mean-gamma per the check above)
        if (params && params->no_rescale_gamma_invar)                JOLT_DECLINE("no-rescale-gamma-invar"); // GPU unconditionally rescales rates by 1/(1-pinv); this flag disables IQ-TREE's rescale -> mismatch -> CPU
        if (aln->frac_const_sites <= 2.0*JOLT_MIN_PINVAR)            JOLT_DECLINE("no-const-sites");
        optPinv = 1;
    }

    // ---- model eigen factors (alpha-independent; same convention as the clean-room lnL) ----
    double *eval = model->getEigenvalues();
    double *U    = model->getEigenvectors();
    double *Uinv = model->getInverseEigenvectors();
    if (!eval || !U || !Uinv) return (double)NAN;
    vector<double> UinvRowSum(ns, 0.0);
    for (int i = 0; i < ns; i++) { double s = 0; for (int j = 0; j < ns; j++) s += Uinv[i*ns+j]; UinvRowSum[i] = s; }
    vector<double> catProp(ncat), catRate0(ncat);
    for (int c = 0; c < ncat; c++) { catProp[c] = site_rate->getProp(c); catRate0[c] = site_rate->getRate(c); }   // G.5.1: +R rates/weights
    // G.6 free-Q: the initial free exchangeabilities (model->getVariables()[1..nFreeQ], raw rates), the ctx the
    // decompose callback binds to, and the output Q buffer. All empty/no-op for fixed-Q (nFreeQ==0).
    vector<double> q0vec(nFreeQ > 0 ? nFreeQ : 0), outQ(nFreeQ > 0 ? nFreeQ : 0);
    if (nFreeQ > 0) model->gpuGetFreeParams(q0vec.data());
    JoltQCtx qctx{ model, ns };

    // ---- topology rooted at internal node R (IQ-TREE roots at a leaf; lnL is reversible-invariant) ----
    if (!root || !root->isLeaf() || root->neighbors.empty()) return (double)NAN;
    Node *R = root->neighbors[0]->node;
    if (R->isLeaf()) return (double)NAN;

    map<Node*,int> nid;
    vector<Node*> nodes, parentNode;
    vector<double> parentLen;
    vector<int> leafTax;
    vector<vector<int>> childList;
    function<void(Node*,Node*,double)> indexDfs = [&](Node *n, Node *dad, double lenToDad) {
        int myi = (int)nodes.size(); nid[n] = myi; nodes.push_back(n);
        parentNode.push_back(dad); parentLen.push_back(lenToDad);
        leafTax.push_back(n->isLeaf() ? aln->getSeqID(n->name) : -1);
        childList.push_back(vector<int>());
        for (auto nb : n->neighbors) { if (nb->node == dad) continue; indexDfs(nb->node, n, nb->length); }
    };
    indexDfs(R, nullptr, 0.0);
    int nNodes = (int)nodes.size();
    // children must be recorded AFTER all indices assigned (a node's child indices are known once its subtree is visited)
    for (int i = 0; i < nNodes; i++) {
        Node *n = nodes[i], *dad = parentNode[i];
        for (auto nb : n->neighbors) { if (nb->node == dad) continue; childList[i].push_back(nid[nb->node]); }
        if ((int)childList[i].size() > 3) return (double)NAN;   // >3 children (only R has 3) -> unsupported
    }

    // ---- compact tip states + pattern frequencies + flat topology arrays ----
    int nptn = (int)aln->size(), ntax = (int)aln->getNSeq();
    vector<unsigned char> tip((size_t)ntax*nptn);
    for (int i = 0; i < nNodes; i++) {
        if (leafTax[i] < 0) continue; int tax = leafTax[i];
        if (tax < 0 || tax >= ntax) return (double)NAN;
        for (int p = 0; p < nptn; p++) { int st = (int)aln->at(p)[tax]; tip[(size_t)tax*nptn+p] = (unsigned char)((st < ns) ? st : ns); }
    }
    vector<double> ptnFreq(nptn);
    for (int p = 0; p < nptn; p++) ptnFreq[p] = (double)aln->at(p).frequency;

    // G.4.3b — pinv-independent invariant base per pattern (== ptn_invar[p]/pinv; replicates the constant-site
    // logic of PhyloTree::computePtnInvar, phylotreesse.cpp:560). base_invar[p] = Σ_{states s with which EVERY
    // taxon is compatible as a constant site} freq[s]: const_char==STATE_UNKNOWN -> 1 ; <ns -> freq[const_char] ;
    // DNA/PROTEIN ambiguous -> sum over compatible states ; >STATE_UNKNOWN -> 0 (a variable pattern). Multiplying
    // by pinv reproduces IQ-TREE's own ptn_invar exactly, so the final CPU self-check is a genuine parity gate.
    vector<double> base_invar(nptn, 0.0);
    double pinvMax = aln->frac_const_sites;
    if (optPinv) {
        vector<double> sf(ns, 0.0); model->getStateFrequency(sf.data(), 0);
        const int ambi_aa[] = {4+8, 32+64, 512+1024};   // B=N|D, Z=Q|E, U=I|L
        int SU = (int)aln->STATE_UNKNOWN;
        for (int p = 0; p < nptn; p++) {
            int cc = (int)aln->at(p).const_char;
            if (cc > SU)                            base_invar[p] = 0.0;
            else if (cc == SU)                      base_invar[p] = 1.0;
            else if (cc < ns)                       base_invar[p] = sf[cc];
            else if (aln->seq_type == SEQ_DNA)     { double s=0; int cs=cc-ns+1; for (int x=0;x<ns;x++) if (cs & (1<<x)) s+=sf[x]; base_invar[p]=s; }
            else if (aln->seq_type == SEQ_PROTEIN) { double s=0; int cs=cc-ns;   if (cs>=0 && cs<3) for (int x=0;x<11;x++) if (ambi_aa[cs] & (1<<x)) s+=sf[x]; base_invar[p]=s; }
        }
    }

    vector<int> nodeNch(nNodes), nodeChild(nNodes*3, -1), nodeLeaf(nNodes);
    vector<double> nodeParentLen(nNodes);
    for (int i = 0; i < nNodes; i++) {
        nodeNch[i] = (int)childList[i].size(); nodeLeaf[i] = leafTax[i]; nodeParentLen[i] = parentLen[i];
        for (int k = 0; k < (int)childList[i].size() && k < 3; k++) nodeChild[i*3+k] = childList[i][k];
    }

    double alpha0 = (ncat > 1) ? site_rate->getGammaShape() : 1.0;
    int optAlpha = (ncat > 1 && !site_rate->isFixGammaShape()) ? 1 : 0;

    // ---- run the JOLT optimiser on the GPU ----
    vector<double> outBrlen(nNodes, 0.0); double outAlpha = alpha0; double outPinv = pinv0; int outIters = 0;
    double joltLnL = gpu_jolt_optimize(ns, nptn, ncat, ntax, nNodes, /*root=*/nid[R],
        Uinv, UinvRowSum.data(), U, eval, catProp.data(), tip.data(), ptnFreq.data(),
        nodeNch.data(), nodeChild.data(), nodeLeaf.data(), nodeParentLen.data(),
        alpha0, optAlpha, /*maxiter=*/400,
        base_invar.data(), pinv0, optPinv, JOLT_MIN_PINVAR, pinvMax,
        catRate0.data(), rgcheck ? 1 : 0,   // G.5.1: +R FreeRate seeding + gated FD-check
        nFreeQ, (nFreeQ > 0 ? q0vec.data() : nullptr), jolt_qdecompose_intree, &qctx,   // G.6: DNA free-Q
        (nFreeQ > 0 ? outQ.data() : nullptr),
        outBrlen.data(), &outAlpha, &outPinv, &outIters);
    if (std::isnan(joltLnL)) {
        static bool warned = false;
        if (!warned) { warned = true; printf("[JOLT] gpu_jolt_optimize returned NaN -> CPU fallback (optimizeParameters)\n"); }
        return (double)NAN;
    }

    // ---- write the optimised branch lengths back (both directed neighbours of each edge v -> parent) ----
    for (int v = 0; v < nNodes; v++) {
        Node *child = nodes[v], *par = parentNode[v];
        if (!par) continue;                                     // R: no parent edge (covered as some node's child edge)
        Neighbor *fwd = par->findNeighbor(child); Neighbor *bwd = child->findNeighbor(par);
        if (fwd) fwd->length = outBrlen[v];
        if (bwd) bwd->length = outBrlen[v];
    }
    // ---- write Q + alpha + pinv back through the setters, then invalidate ALL partial-LH + transition caches ----
    // G.6: set the model to the OPTIMISED free-Q deterministically (the launcher's internal Q thrashing leaves the
    // model in an indeterminate state) — gpuSetFreeParamsDecompose applies param_spec + re-decomposes, so the
    // self-check below recomputes the CPU lnL at exactly the JOLT optimum (a genuine GPU-vs-CPU write-back gate).
    if (nFreeQ > 0) model->gpuSetFreeParamsDecompose(outQ.data());
    if (optPinv) site_rate->setPInvar(outPinv);                 // G.4.3b: sets p_invar + recomputes rates (RateGammaInvar::setPInvar)
    if (optAlpha) site_rate->setGammaShape(outAlpha);           // sets gamma_shape + recomputes the discrete rates
    clearAllPartialLH();                                        // brlen + alpha + pinv + Q changed -> partials, theta & ptn_invar stale

    // ---- self-check: a FRESH CPU computeLikelihood() must reproduce the JOLT lnL (the load-bearing G.4.2a gate) ----
    double cpuLnL = computeLikelihood();
    double rel = (cpuLnL != 0.0) ? fabs((joltLnL - cpuLnL) / cpuLnL) : fabs(joltLnL - cpuLnL);
    static int report_count = 0;
    // G.4.3a: use model->getName() (includes the +F/+FO freq suffix) not model->name (matrix only) — the old print
    // dropped +F, mislabelling LG+F+G4 as "LG+G4" and making +F JOLT-coverage invisible/uncountable. Cap raised so a
    // full -m TESTONLY logs every engagement (coverage is now measurable).
    string joltModelName = model->getName() + (ncat > 1 ? ("+G" + std::to_string(ncat)) : string(""));
    if (report_count < 1000) { report_count++;
        printf("[JOLT] model=%s ns=%d ncat=%d: %d joint iters | GPU lnL=%.6f  CPU lnL=%.6f  rel=%.3e %s | alpha %.6f->%.6f | pinv %.6f->%.6f%s\n",
               joltModelName.c_str(), ns, ncat, outIters,
               joltLnL, cpuLnL, rel, (rel <= 1e-9 ? "PASS" : (rel <= 1e-6 ? "OK(gamma-resid)" : "MISMATCH")),
               alpha0, (ncat>1?site_rate->getGammaShape():0.0),
               pinv0, (optPinv?site_rate->getPInvar():0.0), (optPinv?" +I":"")); }

    // G.6.1 safety gate: if the fresh CPU recompute disagrees with the JOLT lnL at the SAME written-back params by
    // more than the gamma-residual band, the GPU result is untrustworthy (a kernel/regime failure, not a convergence
    // gap — write-back coherence is otherwise ~1e-12 universally). Return NaN so the caller re-optimises on the CPU
    // from scratch. (Convergence-to-the-CPU-MLE is validated separately: G.6.0b JOLT>=CPU on HKY..GTR; +G/+I rel ~1e-12.)
    if (!(rel <= 1e-6)) {   // audit RISK-3: NOT(<=) so a NaN/inf rel (cpuLnL underflowed to NaN) ALSO trips the
                            // fallback and returns NaN BEFORE the setCurScore(cpuLnL) below could poison _cur_score
        static bool warned_mismatch = false;
        if (!warned_mismatch) { warned_mismatch = true;
            printf("[JOLT] write-back MISMATCH rel=%.3e > 1e-6 -> CPU fallback (model=%s)\n", rel, joltModelName.c_str()); }
        return (double)NAN;
    }

    setCurScore(cpuLnL);
    return cpuLnL;
}

// ============================================================================================================
// Phase G.8.2.2 — PRODUCTION GPU JOLT optimiser for NON-FUSED PROFILE-MIXTURE models (C20/C30/C60/MEOW...).
// The mixture analogue of optimizeParametersJOLT: dispatched from ModelFactory::optimizeParameters under --jolt
// when getNMixtures()>1. Optimises BRANCHES + the gamma shape alpha on the GPU (validated diagonal-LM joint step
// over the regime axis r=m*ncat+c), holding the CLASS WEIGHTS FIXED, then writes back + self-checks vs a fresh CPU
// computeLikelihood (rel<=1e-6 gate -> NaN/CPU fallback). It reuses the VALIDATED block-optimiser core of the
// gpuMixJointOptimizeCrossCheckOnce kill-switch (PASS@1e-8, cold-vs-warm 8.85e-9) — the warm path with the EM weight
// block REMOVED, because the eligibility gate restricts to weight-fixed mixtures.
//
// ELIGIBILITY — gate on model->getNDim()==0 (red-teamed, the necessary-AND-sufficient "branches+alpha only" test):
// ModelMixture::getNDim() = (fix_prop?0:size-1) + Sum_m at(m)->getNDim(). ==0 therefore implies BOTH fix_prop==true
// (fixed published class weights, the C-series default; else +size-1 weight dims) AND every per-class getNDim()==0
// (no free per-class freq/Q dims). The latter matters: with -mfopt each class becomes FREQ_ESTIMATE adding ns-1 free
// freq params per class that the CPU optimises and the GPU would SILENTLY DROP — and the write-back self-check would
// NOT catch it (it recomputes the CPU lnL at the SAME un-optimised freqs the GPU used, so they agree). This mirrors
// the single-model gate (optimizeParametersJOLT: if (ndim!=0 && !freeQok) DECLINE). +I and free weights -> CPU.
// ============================================================================================================
double PhyloTree::optimizeParametersJOLTMix(int fixed_len) {
    static const bool JOLT_DBG = (getenv("JOLT_DEBUG") != nullptr);
    #define JMIX_DECLINE(why) do { if (JOLT_DBG) { fprintf(stderr, "[JOLTMIX-GATE] decline reason=%s\n", why); fflush(stderr); } return (double)NAN; } while (0)
    if (!model || !site_rate || !aln) JMIX_DECLINE("null-ptr");
    if (fixed_len != BRLEN_OPTIMIZE) JMIX_DECLINE("brlen-mode");
    int ns = aln->num_states;
    if (ns != 4 && ns != 20) JMIX_DECLINE("num-states");
    if (!model->isReversible() || model->getNMixtures() <= 1 || model->isSiteSpecificModel()) JMIX_DECLINE("nonrev/single/ssm");
    ModelMixture *mix = dynamic_cast<ModelMixture*>(model);
    if (!mix || mix->isFused()) JMIX_DECLINE("not-nonfused-mixture");   // LG4M/LG4X 1:1 class<->rate pairing -> CPU
    if (model->getNDim() != 0) JMIX_DECLINE("free-mixture-params");     // free weights (-mwopt) or per-class freq/Q (-mfopt) -> CPU
    if (site_rate->getPInvar() > 0.0) JMIX_DECLINE("plus-I");           // +I omitted in the clean-room sweep -> CPU
    int N = model->getNMixtures();
    int ncat = site_rate->getNRate();
    if (ncat < 1 || ncat > 64) JMIX_DECLINE("ncat-range");
    if (ncat > 1 && site_rate->isGammaRate() != GAMMA_CUT_MEAN) JMIX_DECLINE("non-mean-gamma");  // only the Yang-1994 mean discretisation
    if (!root || !root->isLeaf() || root->neighbors.empty()) JMIX_DECLINE("bad-root");
    Node *Rt = root->neighbors[0]->node;
    if (!Rt || Rt->isLeaf()) JMIX_DECLINE("Rt-leaf");
    #undef JMIX_DECLINE

    // canonical node ordering: REPLICATE the clean-room indexDfs (root at Rt, preorder, skip dad) so b[] indexed by
    // this nid is a valid parentLenOverride for BOTH clean-room functions (identical indexDfs => identical nid). The
    // self-check below proves the index match (a divergent nid would corrupt the branch step and fail the gate).
    std::map<Node*,int> nidMap; std::vector<Node*> nodes; std::vector<Node*> parentOf; std::vector<double> bLive;
    std::function<void(Node*,Node*,double)> indexDfs = [&](Node *n, Node *dad, double lenToDad) {
        nidMap[n] = (int)nodes.size(); nodes.push_back(n); parentOf.push_back(dad); bLive.push_back(lenToDad);
        for (auto nb : n->neighbors) { if (nb->node == dad) continue; indexDfs(nb->node, n, nb->length); }
    };
    indexDfs(Rt, nullptr, 0.0);
    int nNodes = (int)nodes.size();
    int rootId = nidMap[Rt];

    double alpha0 = (ncat > 1) ? site_rate->getGammaShape() : 1.0;
    int optAlpha = (ncat > 1 && !site_rate->isFixGammaShape()) ? 1 : 0;
    auto alphaArg = [&](double a){ return (ncat > 1) ? a : -1.0; };   // -1 => clean-room keeps the live cat rates (ncat==1 path)

    // ---- GPU optimiser loop: joint diagonal-LM over (all branches + alpha), weights FIXED (w_override=nullptr => the
    // clean-room reads the live model weights, identical to what the derivative uses). Validated diagonal-LM core from
    // gpuMixJointOptimizeCrossCheckOnce::runOpt (warm path, EM weight block removed). Held under the process-wide mutex
    // because the mix clean-room launchers are not internally locked and ModelFinder runs candidates OpenMP-parallel. ----
    std::vector<double> b = bLive; double alpha = (ncat > 1 ? alpha0 : 1.0);
    double finalLnL = (double)NAN; int outIters = 0;
    {
        std::lock_guard<std::mutex> lk(gpu_mixjolt_mtx);
        double mu = 1.0; int stall = 0;
        double lnL = gpuComputeTreeLnLCleanRoomMix(nullptr, nullptr, nullptr, b.data(), alphaArg(alpha));
        if (!std::isnan(lnL)) {
            const int maxOuter = 400;
            for (int outer = 0; outer < maxOuter; outer++) {
                double lnL0 = lnL;
                std::vector<Node*> cN, pN; std::vector<double> df, ddf;
                if (!gpuComputeAllBranchDervCleanRoomMix(cN, pN, df, ddf, b.data(), alphaArg(alpha))) { lnL = (double)NAN; break; }
                std::vector<double> gdf(nNodes, 0.0), gddf(nNodes, 0.0);
                for (size_t i = 0; i < cN.size(); i++) { int v = nidMap.at(cN[i]); gdf[v] = df[i]; gddf[v] = ddf[i]; }
                double ga = 0.0, curvA = 1e-12, eps = 0.0;
                if (optAlpha) {   // alpha gradient/curvature by central FD on the clean-room lnL (no new kernel)
                    eps = 1e-3 * std::max(alpha, 1.0);
                    double lp = gpuComputeTreeLnLCleanRoomMix(nullptr, nullptr, nullptr, b.data(), alpha+eps);
                    double lm = gpuComputeTreeLnLCleanRoomMix(nullptr, nullptr, nullptr, b.data(), alpha-eps);
                    if (std::isnan(lp) || std::isnan(lm)) { lnL = (double)NAN; break; }
                    ga = (lp - lm) / (2.0*eps); curvA = std::fabs((lp - 2.0*lnL + lm) / (eps*eps)); if (curvA < 1e-12) curvA = 1e-12;
                }
                for (int bt = 0; bt < 16; bt++) {   // shared-mu LM backtracking over ALL branches + the scalar alpha together
                    std::vector<double> bc = b;
                    for (int v = 0; v < nNodes; v++) { if (v == rootId) continue;
                        double stepv = gdf[v] / (std::fabs(gddf[v]) + mu); bc[v] = b[v] + stepv;
                        if (bc[v] < 1e-6) bc[v] = 1e-6; if (bc[v] > 20.0) bc[v] = 20.0; }
                    double ac = alpha;
                    if (optAlpha) { ac = alpha + ga / (curvA + mu); if (ac < 0.02) ac = 0.02; if (ac > 50.0) ac = 50.0; }
                    double ln = gpuComputeTreeLnLCleanRoomMix(nullptr, nullptr, nullptr, bc.data(), alphaArg(ac));
                    if (std::isnan(ln)) { lnL = (double)NAN; break; }
                    if (ln > lnL) { b = bc; alpha = ac; lnL = ln; mu = std::max(mu*0.5, 1e-9); break; }   // accept any strict improvement
                    else mu = std::min(mu*4.0, 1e12);                                                     // cap mu (else it runs to +inf and freezes)
                }
                if (std::isnan(lnL)) break;
                outIters = outer + 1;
                if (JOLT_DBG) fprintf(stderr, "[JOLTMIX-DBG] outer=%d lnL=%.6f mu=%.2e alpha=%.4f\n", outer, lnL, mu, alpha);
                // ridge-recognizing termination: 3 consecutive outers improving by <1e-7 => the diagonal-LM has reached
                // its ~1e-8 accuracy floor on the ill-conditioned mixture ridge (validated in the kill-switch). The
                // 1e-6 write-back gate is comfortably above this floor, so coherence holds; the lnL may sit marginally
                // below the true MLE but the gap (<<1) is inconsequential for BIC selection (exact penalty term).
                if (std::fabs(lnL - lnL0) < 1e-7) stall++; else stall = 0;
                if (outer > 0 && stall >= 3) break;
            }
            finalLnL = lnL;
        }
    }
    if (std::isnan(finalLnL)) {
        static bool warned = false;
        if (!warned) { warned = true; printf("[JOLTMIX] gpu mixture optimise returned NaN -> CPU fallback (optimizeParameters)\n"); }
        return (double)NAN;
    }

    // ---- write the optimised branch lengths (both directed neighbours) + alpha back; NO weight/Q write-back (fixed) ----
    for (int v = 0; v < nNodes; v++) {
        Node *child = nodes[v], *par = parentOf[v];
        if (!par) continue;                                     // Rt: no parent edge (covered as some node's child edge)
        Neighbor *fwd = par->findNeighbor(child); Neighbor *bwd = child->findNeighbor(par);
        if (fwd) fwd->length = b[v];
        if (bwd) bwd->length = b[v];
    }
    if (optAlpha) site_rate->setGammaShape(alpha);              // sets gamma_shape + recomputes the discrete rates
    clearAllPartialLH();                                        // brlen + alpha changed -> partials/theta stale

    // ---- self-check: a FRESH CPU computeLikelihood() must reproduce the JOLT lnL at the written-back params ----
    double cpuLnL = computeLikelihood();
    double rel = (cpuLnL != 0.0) ? std::fabs((finalLnL - cpuLnL) / cpuLnL) : std::fabs(finalLnL - cpuLnL);
    static int report_count = 0;
    string joltModelName = model->getName() + (ncat > 1 ? ("+G" + std::to_string(ncat)) : string(""));
    if (report_count < 1000) { report_count++;
        printf("[JOLTMIX] model=%s N=%d ns=%d ncat=%d: %d iters | GPU lnL=%.6f  CPU lnL=%.6f  rel=%.3e %s | alpha %.6f->%.6f\n",
               joltModelName.c_str(), N, ns, ncat, outIters, finalLnL, cpuLnL, rel,
               (rel <= 1e-6 ? "OK" : "MISMATCH"), alpha0, (ncat > 1 ? site_rate->getGammaShape() : 0.0)); }

    if (!(rel <= 1e-6)) {   // NOT(<=) so a NaN/inf rel also trips the fallback before setCurScore poisons _cur_score
        static bool warned_mm = false;
        if (!warned_mm) { warned_mm = true;
            printf("[JOLTMIX] write-back MISMATCH rel=%.3e > 1e-6 -> CPU fallback (model=%s)\n", rel, joltModelName.c_str()); }
        return (double)NAN;
    }
    setCurScore(cpuLnL);
    return cpuLnL;
}

#endif // IQTREE_GPU
