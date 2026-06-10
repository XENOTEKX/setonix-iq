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
#include "model/modelsubst.h"
#include "model/rateheterogeneity.h"
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
// Phase G.4.2 — GPU JOLT joint-gradient optimiser for ONE candidate model. Builds the clean-room inputs from the
// LIVE objects (mirroring gpuComputeTreeLnLCleanRoom), runs the validated G.4.1b joint LM driver on the GPU,
// writes the optimised (197 branches + alpha) back through the cache-invalidating setters, and self-checks that a
// FRESH CPU computeLikelihood() reproduces the JOLT lnL. Returns NaN if JOLT-ineligible / CUDA error -> caller
// falls back to the standard CPU path.
// ============================================================================================================
double PhyloTree::optimizeParametersJOLT(int fixed_len) {
    // ---- eligibility gate (the validated G.4.1/G.4.1b scope: fixed-Q reversible, ns in {4,20}, no +I, gamma-or-uniform) ----
    if (!model || !site_rate || !aln) return (double)NAN;
    if (fixed_len != BRLEN_OPTIMIZE) return (double)NAN;        // JOLT optimises branches; other brlen modes -> CPU
    int ns = aln->num_states;
    if (ns != 4 && ns != 20) return (double)NAN;
    if (!model->isReversible() || model->getNMixtures() != 1 || model->isSiteSpecificModel()) return (double)NAN;
    if (model->getNDim() != 0) return (double)NAN;              // free substitution params (e.g. GTR/+FO) -> eigen would move -> CPU
    if (site_rate->getPInvar() > 0.0) return (double)NAN;       // +I not yet supported by JOLT -> CPU
    int ncat = site_rate->getNRate();
    if (ncat < 1 || ncat > 64) return (double)NAN;
    if (ncat > 1 && site_rate->getGammaShape() <= 0.0) return (double)NAN;  // multi-cat but not gamma (+R) -> CPU

    // ---- model eigen factors (alpha-independent; same convention as the clean-room lnL) ----
    double *eval = model->getEigenvalues();
    double *U    = model->getEigenvectors();
    double *Uinv = model->getInverseEigenvectors();
    if (!eval || !U || !Uinv) return (double)NAN;
    vector<double> UinvRowSum(ns, 0.0);
    for (int i = 0; i < ns; i++) { double s = 0; for (int j = 0; j < ns; j++) s += Uinv[i*ns+j]; UinvRowSum[i] = s; }
    vector<double> catProp(ncat);
    for (int c = 0; c < ncat; c++) catProp[c] = site_rate->getProp(c);

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
    vector<int> nodeNch(nNodes), nodeChild(nNodes*3, -1), nodeLeaf(nNodes);
    vector<double> nodeParentLen(nNodes);
    for (int i = 0; i < nNodes; i++) {
        nodeNch[i] = (int)childList[i].size(); nodeLeaf[i] = leafTax[i]; nodeParentLen[i] = parentLen[i];
        for (int k = 0; k < (int)childList[i].size() && k < 3; k++) nodeChild[i*3+k] = childList[i][k];
    }

    double alpha0 = (ncat > 1) ? site_rate->getGammaShape() : 1.0;
    int optAlpha = (ncat > 1 && !site_rate->isFixGammaShape()) ? 1 : 0;

    // ---- run the JOLT optimiser on the GPU ----
    vector<double> outBrlen(nNodes, 0.0); double outAlpha = alpha0; int outIters = 0;
    double joltLnL = gpu_jolt_optimize(ns, nptn, ncat, ntax, nNodes, /*root=*/nid[R],
        Uinv, UinvRowSum.data(), U, eval, catProp.data(), tip.data(), ptnFreq.data(),
        nodeNch.data(), nodeChild.data(), nodeLeaf.data(), nodeParentLen.data(),
        alpha0, optAlpha, /*maxiter=*/400, outBrlen.data(), &outAlpha, &outIters);
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
    // ---- write alpha back through the gamma setter, then invalidate ALL partial-LH + transition caches ----
    if (optAlpha) site_rate->setGammaShape(outAlpha);           // sets gamma_shape + recomputes the discrete rates
    clearAllPartialLH();                                        // brlen + alpha changed -> partials & theta stale (advisor: the cache-coherence watch item)

    // ---- self-check: a FRESH CPU computeLikelihood() must reproduce the JOLT lnL (the load-bearing G.4.2a gate) ----
    double cpuLnL = computeLikelihood();
    double rel = (cpuLnL != 0.0) ? fabs((joltLnL - cpuLnL) / cpuLnL) : fabs(joltLnL - cpuLnL);
    static int report_count = 0;
    if (report_count < 12) { report_count++;
        printf("[JOLT] model=%s+%s%d ns=%d ncat=%d: %d joint iters | GPU lnL=%.6f  CPU lnL=%.6f  rel=%.3e %s | alpha %.6f->%.6f\n",
               model->name.c_str(), (ncat>1?"G":""), (ncat>1?ncat:0), ns, ncat, outIters,
               joltLnL, cpuLnL, rel, (rel <= 1e-9 ? "PASS" : (rel <= 1e-6 ? "OK(gamma-resid)" : "MISMATCH")),
               alpha0, (ncat>1?site_rate->getGammaShape():0.0)); }

    setCurScore(cpuLnL);
    return cpuLnL;
}

#endif // IQTREE_GPU
