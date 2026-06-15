#!/bin/bash
# probe_dcgm_ncu_h200.sh — tiny diagnostic: CAN a non-root user read true SM-active% on a Gadi GPU node?
# The DCGM sweep returned "DCGM available: 0" (profiling field 1002 probe failed) on both A100 and H200.
# This settles WHY, and whether ANY non-root path to true SM%/occupancy exists, without wasting a full sweep.
# Single GPU, ~10 min, no build. Submit: qsub gadi-ci/gpu-modelfinder/probe_dcgm_ncu_h200.sh
#PBS -N dcgm-probe
#PBS -P dx61
#PBS -q gpuhopper
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=48GB
#PBS -l walltime=00:15:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
echo "════════ node $(hostname) $(date -Iseconds) ════════"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo; echo "════════ [1] NVIDIA driver profiling restriction (THE definitive check) ════════"
echo "-- /proc/driver/nvidia/params (RmProfilingAdminOnly: 1  =>  non-root profiling BLOCKED) --"
grep -iE "profil" /proc/driver/nvidia/params 2>&1 || echo "  (no Profiling line / unreadable)"
echo "-- modprobe param NVreg_RestrictProfilingToAdminUsers --"
cat /sys/module/nvidia/parameters/NVreg_RestrictProfilingToAdminUsers 2>&1 || echo "  (modprobe param unreadable)"

echo; echo "════════ [2] DCGM binary / hostengine state ════════"
command -v dcgmi && dcgmi --version 2>&1 | head -2
command -v nv-hostengine && echo "nv-hostengine present"
echo "-- already-running standalone hostengine? --"
pgrep -a nv-hostengine 2>&1 || echo "  (none)"

echo; echo "════════ [3] DCGM embedded mode (dcgmi dmon auto-embeds; no standalone) ════════"
echo "-- non-profiling field 203 (DEV_GPU_UTIL) — should ALWAYS work --"
dcgmi dmon -e 203 -c 2 -d 1000 2>&1; echo "  exit=$?"
echo "-- profiling field 1002 (PROF_SM_ACTIVE) — the one the sweep failed on --"
dcgmi dmon -e 1002 -c 2 -d 1000 2>&1; echo "  exit=$?"

echo; echo "════════ [4] DCGM with an explicit standalone hostengine + profile unpause ════════"
nv-hostengine 2>&1 | head; sleep 3
echo "-- discovery --"; dcgmi discovery -l 2>&1 | head -5
echo "-- profile --pause (some sites need an explicit pause/resume cycle) --"; dcgmi profile --pause 2>&1 | head -3
echo "-- SM_ACTIVE/OCC/DRAM/FP64 watch via standalone HE --"
dcgmi dmon -e 1002,1003,1005,1006 -c 3 -d 1000 2>&1; echo "  exit=$?"
nv-hostengine --term 2>&1 | head -2

echo; echo "════════ [5] Nsight Compute (ncu) perf-counter permission probe ════════"
if command -v ncu >/dev/null 2>&1; then
  ncu --version 2>&1 | head -2
  cat > /tmp/_probe.cu <<'CU'
__global__ void k(float*x){int i=threadIdx.x; if(i<256) x[i]=x[i]*1.0001f+0.5f;}
int main(){float*d; cudaMalloc(&d,256*sizeof(float)); for(int i=0;i<2000;i++) k<<<32,256>>>(d); cudaDeviceSynchronize(); return 0;}
CU
  nvcc -arch=sm_90 /tmp/_probe.cu -o /tmp/_probe 2>&1 | head -3
  echo "-- ncu sm__throughput (ERR_NVGPUCTRPERM => perf counters admin-restricted) --"
  ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active \
      --target-processes all /tmp/_probe 2>&1 | tail -25; echo "  exit=$?"
else
  echo "  ncu NOT on PATH after module load cuda/12.5.1"
fi

echo; echo "════════ VERDICT GUIDE ════════"
echo "  [1] RmProfilingAdminOnly:1 + [3]/[4] 1002 fail + [5] ERR_NVGPUCTRPERM  => admin-locked, NO non-root SM%."
echo "  [1] =0 (or absent) AND ([3] or [4] or [5] yields numbers)            => SM% IS obtainable; fix the sweep probe."
echo "════════ DONE $(date -Iseconds) ════════"
