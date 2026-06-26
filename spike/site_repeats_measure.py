#!/usr/bin/env python3
# Site-repeats / distinct-subtree-config measurement on a REAL AA-100K tree.
# Question: how much could the partial-likelihood compute collapse via subtree repeats,
# and WHERE (near tips vs root)? This decides whether site-repeats is a real per-GPU-iteration
# speedup lever (-> makes the GPU+MPI hybrid a clear win) or a near-tip-only marginal one.
#
# Method (RAxML-NG / Kobert-Stamatakis-Flouri style): compress columns to nptn distinct patterns,
# then postorder: each node's "config id" at pattern p = the distinct combination of its children's
# config ids. distinct_configs[node] = #partials that actually need computing at that node; the rest
# are repeats reusable by index. Baseline (current) work = nInternal * nptn.
import sys, numpy as np
PHY=sys.argv[1] if len(sys.argv)>1 else "/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/AA/LG+I+G4/taxa_100/len_100000/tree_1/alignment_100000.phy"
TF =sys.argv[2] if len(sys.argv)>2 else "/scratch/rc29/as1708/treesearch-profiling/results/ts6aax_172361466/aa_100000_fused.treefile"
sys.setrecursionlimit(100000)

# --- read phylip (one line per taxon) ---
seqs={}
with open(PHY) as f:
    ntax,nsite=map(int,f.readline().split())
    for line in f:
        if not line.strip(): continue
        p=line.split(None,1)
        if len(p)<2: continue
        seqs[p[0]]=p[1].replace(" ","").replace("\n","")
names=list(seqs.keys()); assert len(names)==ntax, (len(names),ntax)
M=np.empty((ntax,nsite),dtype=np.uint8)
for i,nm in enumerate(names):
    M[i]=np.frombuffer(seqs[nm].encode('latin-1'),dtype=np.uint8)[:nsite]

# --- compress full columns -> distinct patterns (IQ-TREE nptn) ---
cols=np.ascontiguousarray(M.T)                      # nsite x ntax
void=cols.view([('',cols.dtype)]*ntax)
uniq,counts=np.unique(void,return_counts=True)
nptn=uniq.shape[0]
patM=uniq.view(cols.dtype).reshape(nptn,ntax).T.astype(np.int64)   # ntax x nptn leaf states
print(f"ntax={ntax} nsite={nsite} nptn(distinct columns)={nptn} ({100.0*nptn/nsite:.1f}% of sites distinct)")

# --- parse newick ---
s=open(TF).read().strip(); pos=[0]
class N:
    __slots__=('ch','nm')
    def __init__(s_): s_.ch=[]; s_.nm=None
def parse():
    n=N()
    if s[pos[0]]=='(':
        pos[0]+=1
        while True:
            n.ch.append(parse())
            if s[pos[0]]==',': pos[0]+=1; continue
            if s[pos[0]]==')': pos[0]+=1; break
        while pos[0]<len(s) and s[pos[0]] not in ',()': pos[0]+=1
    else:
        st=pos[0]
        while s[pos[0]] not in ':,()': pos[0]+=1
        n.nm=s[st:pos[0]]
        while pos[0]<len(s) and s[pos[0]] not in ',()': pos[0]+=1
    return n
root=parse()
idx={nm:i for i,nm in enumerate(names)}

# --- postorder: distinct subtree-config count per internal node ---
res=[]   # (subtree_size, distinct_configs)
def post(n):
    if not n.ch:
        return patM[idx[n.nm]], 1
    cids=[]; size=0
    for c in n.ch:
        cid,csz=post(c); cids.append(cid); size+=csz
    stack=np.ascontiguousarray(np.vstack(cids).T)              # nptn x k
    v=stack.view([('',stack.dtype)]*stack.shape[1])
    _,inv=np.unique(v,return_inverse=True)
    inv=np.asarray(inv).ravel()
    res.append((size,int(inv.max())+1))
    return inv.astype(np.int64), size
post(root)

# --- report ---
nInt=len(res)
tot_full=nInt*nptn
tot_rep=sum(d for _,d in res)
print(f"internal nodes={nInt}")
print(f"FULL partial work (current) = nInternal*nptn = {tot_full:,}")
print(f"SITE-REPEAT partial work    = sum(distinct)   = {tot_rep:,}")
print(f"=> OVERALL partial-compute reduction = {tot_full/tot_rep:.2f}x")
print("\n-- where the savings live (by subtree size, power-of-two buckets) --")
buck={}
for size,dist in res:
    b=1<<(size.bit_length()-1)
    e=buck.setdefault(b,[0,0,0]); e[0]+=1; e[1]+=dist; e[2]+=nptn
print(f"{'subtree taxa':>14} {'nodes':>6} {'mean distinct/nptn':>20} {'reduction':>10} {'%of full work':>14}")
for b in sorted(buck):
    cnt,ds,fs=buck[b]
    print(f"{b:>6}-{b*2-1:<7} {cnt:>6} {ds/fs:>20.3f} {fs/ds:>9.1f}x {100.0*ds/tot_rep:>13.1f}%")
# honest split: the root-ward half (where the gradient/upper work concentrates)
rootward=sorted(res,key=lambda x:-x[0])[:nInt//2]
rw_full=len(rootward)*nptn; rw_rep=sum(d for _,d in rootward)
print(f"\nroot-ward half (largest {len(rootward)} subtrees, where the all-branch gradient cost lives):")
print(f"  reduction there = {rw_full/rw_rep:.2f}x  (this is the number that matters for the JOLT all-branch reopt)")
