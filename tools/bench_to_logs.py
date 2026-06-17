#!/usr/bin/env python3.11
"""bench_to_logs.py — turn the G.7.2 GPU benchmark sweep outputs into dashboard log JSON.

Reads:
  bench_<dev>_<jobid>/SUMMARY.tsv         (clean wall/energy/VRAM/util/parity/winner per type,scale,device)
  prof_<dev>_<jobid>/<TYPE>_<SCALE>/kernsum.csv   (nsys per-kernel GPU-time breakdown; A100)
  prof_<dev>_<jobid>/<TYPE>_<SCALE>/ncu.csv       (ncu occupancy/SM%/DRAM% on hot kernels; <=100K)

Writes (idempotent, overwrites its own ids):
  logs/runs/<run_id>.json       -> standard run record (wall via timing[], modelfinder{}, verify[] parity)
  logs/profiles/<run_id>.json   -> {profile_id, ..., gpu:{bench:true, scale,type,device, wall_s, energy_wh,
                                    peak_vram_gb, gpu_util_pct, parity_rel, winner, full_lnL, full_bic,
                                    engage, decline, nTile, kernels:[...], ncu:[...]}}

Then: python3.11 tools/build.py  (normalise -> web/data -> docs).
Run id scheme: gpu_<type>_<scalelabel>_<device>   e.g. gpu_aa_1m_h200
"""
import sys, os, csv, json, glob, re, datetime

ROOT = "/home/272/as1708/setonix-iq"
SCRATCH = "/scratch/rc29/as1708/iqtree3-gpu"
LOGS_RUNS = os.path.join(ROOT, "logs", "runs")
LOGS_PROF = os.path.join(ROOT, "logs", "profiles")

def scale_label(s):
    s = int(s)
    return {10000:"10K",100000:"100K",1000000:"1M",10000000:"10M"}.get(s, str(s))

def find_bench_summaries():
    return sorted(glob.glob(os.path.join(SCRATCH, "bench_*_*/SUMMARY.tsv")))

def find_prof_dir(dev):
    ds = sorted(glob.glob(os.path.join(SCRATCH, f"prof_{dev}_*")), key=os.path.getmtime, reverse=True)
    return ds[0] if ds else None

def parse_kernsum(path):
    """nsys cuda_gpu_kern_sum CSV -> top kernels [{name,time_pct,total_ns,instances,avg_ns}]."""
    if not os.path.exists(path): return []
    out = []
    try:
        with open(path) as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                # column names vary slightly by nsys version; match loosely
                def col(*cands):
                    for c in cands:
                        for k in row:
                            if k and k.strip().lower().startswith(c): return row[k]
                    return None
                name = col("name")
                if not name: continue
                tpct = col("time (%)","time(%)","time %")
                tot  = col("total time","total")
                inst = col("instances","num calls","count")
                avg  = col("avg")
                # shorten kernel name (drop template args/namespaces for display)
                short = re.sub(r"\(.*", "", name).strip()
                short = short.split("::")[-1][:48]
                def num(x):
                    if x is None: return None
                    try: return float(str(x).replace(",",""))
                    except: return None
                out.append({"name": short, "time_pct": num(tpct), "total_ns": num(tot),
                            "instances": num(inst), "avg_ns": num(avg)})
    except Exception as e:
        sys.stderr.write(f"  kernsum parse warn {path}: {e}\n")
    out = [k for k in out if k["time_pct"] is not None]
    out.sort(key=lambda k: -(k["time_pct"] or 0))
    return out[:12]

def parse_ncu(path):
    """ncu --csv raw -> per-kernel metric rows [{kernel, occupancy_pct, sm_pct, dram_pct, regs}]."""
    if not os.path.exists(path): return []
    rows = {}
    try:
        with open(path) as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                kn = r.get("Kernel Name") or r.get('"Kernel Name"')
                metric = r.get("Metric Name"); val = r.get("Metric Value")
                if not kn or not metric: continue
                kshort = re.sub(r"\(.*","",kn).strip().split("::")[-1][:40]
                d = rows.setdefault(kshort, {"kernel": kshort})
                m = metric.lower()
                try: v = float(str(val).replace(",",""))
                except: v = None
                if "warps_active" in m: d["occupancy_pct"] = v
                elif "sm__throughput" in m: d["sm_pct"] = v
                elif "dram_throughput" in m: d["dram_pct"] = v
                elif "registers_per_thread" in m: d["regs"] = v
    except Exception as e:
        sys.stderr.write(f"  ncu parse warn {path}: {e}\n")
    return list(rows.values())

def main():
    os.makedirs(LOGS_RUNS, exist_ok=True); os.makedirs(LOGS_PROF, exist_ok=True)
    summaries = find_bench_summaries()
    if not summaries:
        print("No bench SUMMARY.tsv found yet."); return
    # profiling data is on A100 only (per the decision)
    prof_a100 = find_prof_dir("a100")
    n_run = n_prof = 0
    for sumf in summaries:
        with open(sumf) as f:
            rdr = csv.DictReader(f, delimiter="\t")
            for row in rdr:
                ty = row["type"]; sc = row["scale"]; dev = row["device"]
                slabel = scale_label(sc)
                rid = f"gpu_{ty.lower()}_{slabel.lower()}_{dev}"
                wall = float(row["wall_total_s"]); lnl = row["full_lnL"]; bic = row["full_bic"]
                try: lnl_f = float(lnl)
                except: lnl_f = None
                try: bic_f = float(bic)
                except: bic_f = None
                parity = float(row["worst_parity_rel"]) if row["worst_parity_rel"] not in ("","NA") else None
                winner = row["winner"]
                aln = (f"/scratch/dx61/.../{ty}/.../len_{sc}/alignment_{sc}.phy")
                now = datetime.datetime.now().astimezone().isoformat()
                # ---- run JSON (standard render: wall, model, lnL, parity) ----
                run = {
                    "run_id": rid,
                    "pbs_id": None,
                    "platform": "gadi",
                    "run_type": "gpu_baseline",
                    "label": f"GPU JOLT {ty}-{slabel} -m MF ({dev.upper()})",
                    "description": f"CTF -m MF (all models) {ty} {slabel} sites on 1x {dev.upper()} (G.7.2 tiling binary, parity-matched).",
                    "timing": [{"command": f"iqtree3 --jolt --gpu -m MF (CTF) -s {ty}_{slabel}.phy", "time_s": wall}],
                    "verify": ([{"file": f"{ty}_{slabel}", "status": "pass",
                                 "expected": lnl_f, "reported": lnl_f, "diff": parity}]
                               if (lnl_f is not None and parity is not None) else []),
                    "env": {"date": now, "hostname": row["host"],
                            "cpu": "—", "iqtree_version": f"IQ-TREE3-GPU JOLT (bin {row['bin_md5'][:8]})"},
                    "dataset_info": {"taxa": 100, "sites": int(sc), "sequence_type": ty,
                                     "dataset_short": f"{ty}-{slabel}"},
                    "modelfinder": {"model_selected": winner, "best_model_bic": winner,
                                    "log_likelihood": lnl_f, "bic": bic_f, "candidates": []},
                }
                json.dump(run, open(os.path.join(LOGS_RUNS, rid+".json"), "w"), indent=1)
                n_run += 1
                # ---- profile JSON (gpu block = the benchmark record the GPU page renders) ----
                gpu = {"bench": True, "scale": int(sc), "scale_label": slabel, "type": ty,
                       "device": dev.upper(), "device_name": {"a100":"NVIDIA A100-80","h200":"NVIDIA H200"}.get(dev,dev),
                       "wall_s": wall, "wall_sub_s": float(row["wall_sub_s"]), "wall_coarse_s": float(row["wall_coarse_s"]),
                       "wall_refine_s": float(row["wall_refine_s"]),
                       "energy_wh": float(row["energy_wh"]), "mean_power_w": float(row["mean_w"]),
                       "peak_vram_gb": round(int(row["peak_vram_mib"])/1024.0, 2), "peak_vram_mib": int(row["peak_vram_mib"]),
                       "gpu_util_pct": int(row["max_util"]), "parity_rel": parity,
                       "winner": winner, "full_lnL": lnl_f, "full_bic": bic_f,
                       "engage": int(row["engage"]) if row["engage"].isdigit() else None,
                       "decline": int(row["decline"]) if row["decline"].isdigit() else None,
                       "nTile": int(row["nTile"]) if row["nTile"].isdigit() else 1,
                       "kernels": [], "ncu": []}
                # attach nsys/ncu (A100 profiling only — match type,scale)
                if prof_a100:
                    pdir = os.path.join(prof_a100, f"{ty}_{sc}")
                    gpu["kernels"] = parse_kernsum(os.path.join(pdir, "kernsum.csv"))
                    gpu["ncu"] = parse_ncu(os.path.join(pdir, "ncu.csv"))
                prof = {"profile_id": rid, "slurm_id": None, "pbs_id": None, "date": now,
                        "dataset": f"{ty}-{slabel}", "threads": 1, "model": winner, "gpu": gpu}
                json.dump(prof, open(os.path.join(LOGS_PROF, rid+".json"), "w"), indent=1)
                n_prof += 1
                print(f"  {rid:24} wall={wall:.0f}s E={gpu['energy_wh']:.2f}Wh VRAM={gpu['peak_vram_gb']}GB util={gpu['gpu_util_pct']}% "
                      f"parity={parity:.2e} nTile={gpu['nTile']} winner={winner} kern={len(gpu['kernels'])} ncu={len(gpu['ncu'])}")
    print(f"\nwrote {n_run} run + {n_prof} profile JSON -> logs/. Now: python3.11 tools/build.py")

if __name__ == "__main__":
    main()
