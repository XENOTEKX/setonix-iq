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
#include "model/rateheterogeneity.h"
#include "model/rategamma.h"   // G.4.3b: GAMMA_CUT_MEAN — the robust mean-gamma discriminator (isGammaRate())
#include "alignment/alignment.h"
#include "tree/gpu/gpu_iqtree.h"
#include <vector>
#include <map>
#include <functional>
#include <cmath>
#include <cstdio>
using namespace std;

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
        bool freeQok = JOLT_FREEQ_EN && ndim > 0 && ndim <= 5 && ns == 4 &&
                       model->getFreqType() != FREQ_ESTIMATE && model->isReversible();
        if (ndim != 0 && !freeQok) JOLT_DECLINE("free-subst-params");  // +FO / AA-GTR / production free-Q -> CPU
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
    if (rel > 1e-6) {
        static bool warned_mismatch = false;
        if (!warned_mismatch) { warned_mismatch = true;
            printf("[JOLT] write-back MISMATCH rel=%.3e > 1e-6 -> CPU fallback (model=%s)\n", rel, joltModelName.c_str()); }
        return (double)NAN;
    }

    setCurScore(cpuLnL);
    return cpuLnL;
}

#endif // IQTREE_GPU
