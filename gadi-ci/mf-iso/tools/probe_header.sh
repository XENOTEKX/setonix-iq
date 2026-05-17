#!/bin/bash
# probe_header.sh — emit hardware/software/binary probe before a benchmark.
#
# Source from a PBS run script:   . tools/probe_header.sh "${IQTREE}"
#
# Caller must export ${WORK_DIR} (output dir) before sourcing.
# Output lines are tagged with PROBE: prefixes so they can be grep'd
# from the run log offline.

probe_hw_sw() {
    local iqtree_bin="${1:-}"

    echo "PROBE: ===== hardware ====="
    echo "PROBE: hw_host=$(hostname)"
    echo "PROBE: hw_date=$(date -Iseconds)"
    echo "PROBE: hw_kernel=$(uname -srvm)"
    if [[ -r /etc/os-release ]]; then
        os_pretty=$(awk -F= '/^PRETTY_NAME=/ {gsub(/"/,"",$2); print $2}' /etc/os-release)
        echo "PROBE: hw_os=${os_pretty}"
    fi

    # CPU summary.
    if command -v lscpu >/dev/null 2>&1; then
        lscpu | sed 's/^/PROBE: hw_cpu_lscpu: /'
    fi

    # NUMA layout.
    if command -v numactl >/dev/null 2>&1; then
        numactl --hardware 2>&1 | sed 's/^/PROBE: hw_numa: /'
    fi

    # Memory snapshot at job start.
    if [[ -r /proc/meminfo ]]; then
        awk '/MemTotal|MemFree|MemAvailable|Buffers|Cached|SwapTotal|HugePages_Total|AnonHugePages/ {print $0}' /proc/meminfo | \
            sed 's/^/PROBE: hw_mem: /'
    fi

    # Hugepage / THP state — relevant for Phase 1 (madvise) work.
    if [[ -r /sys/kernel/mm/transparent_hugepage/enabled ]]; then
        thp="$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)"
        echo "PROBE: hw_thp_enabled=${thp}"
    fi
    if [[ -r /sys/kernel/mm/transparent_hugepage/defrag ]]; then
        thp_d="$(cat /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null)"
        echo "PROBE: hw_thp_defrag=${thp_d}"
    fi

    echo "PROBE: ===== software ====="
    # Modules currently loaded.
    if command -v module >/dev/null 2>&1; then
        module list 2>&1 | sed 's/^/PROBE: sw_modules: /'
    fi

    # Compilers as resolved on PATH.
    for tool in icpx icx clang++ clang gcc mpicxx mpicc mpirun; do
        if command -v "$tool" >/dev/null 2>&1; then
            p="$(command -v "$tool")"
            v="$("$tool" --version 2>&1 | head -1)"
            echo "PROBE: sw_${tool}: path=${p}  version=${v}"
        fi
    done

    # OpenMPI configuration (succinct).
    if command -v ompi_info >/dev/null 2>&1; then
        echo "PROBE: sw_ompi_info_full:"
        ompi_info 2>&1 | head -30 | sed 's/^/PROBE: sw_ompi_info: /'
    fi

    echo "PROBE: ===== binary ====="
    if [[ -n "${iqtree_bin}" && -x "${iqtree_bin}" ]]; then
        echo "PROBE: bin_path=${iqtree_bin}"
        echo "PROBE: bin_stat=$(stat -c '%y size=%s mode=%A owner=%U:%G' "${iqtree_bin}")"
        echo "PROBE: bin_md5=$(md5sum "${iqtree_bin}" | awk '{print $1}')"
        echo "PROBE: bin_file=$(file -L "${iqtree_bin}" 2>&1 | head -1)"

        # Linkage check (omp, mpi, libiomp5).
        ldd "${iqtree_bin}" 2>&1 | grep -iE 'omp|mpi|libc|libstdc' | head -20 | \
            sed 's/^/PROBE: bin_ldd: /'

        # IQ-TREE banner.
        "${iqtree_bin}" --version 2>&1 | head -3 | sed 's/^/PROBE: bin_version: /'

        # FCA / Phase 0.5 / Phase 0.6 / MF-TIME symbol and string presence —
        # confirms the right binary is being used.
        #
        # NOTE: 'nm' and standalone 'strings' use mmap() to read large ELF files.
        # On Lustre compute nodes mmap() of a large /scratch file can silently
        # return empty data even when sequential read() works correctly.  Piping
        # through 'cat' forces a sequential read(); 'strings' reading from stdin
        # (a pipe) cannot use mmap.  We pre-extract all printable strings into a
        # RAM-backed temp file (/dev/shm) once and grep from there.
        local _str_cache
        _str_cache=$(mktemp /dev/shm/.iq3sym.XXXXXX 2>/dev/null) || \
        _str_cache=$(mktemp)
        cat "${iqtree_bin}" 2>/dev/null | strings > "${_str_cache}" 2>/dev/null

        # Symbol check: search mangled-name substrings (unique within the ELF
        # .strtab) and display the demangled label for readability.
        # Mangling: filterRatesMPIEi, 11filterRatesEi, getNextModelEv, evaluateAll
        while IFS='|' read -r lbl srch; do
            if grep -q "${srch}" "${_str_cache}" 2>/dev/null; then
                echo "PROBE: bin_sym_present: ${lbl}"
            else
                echo "PROBE: bin_sym_MISSING: ${lbl}"
            fi
        done <<'SYMS'
CandidateModelSet::filterRatesMPI|filterRatesMPIEi
CandidateModelSet::filterRates|11filterRatesEi
CandidateModelSet::getNextModel|getNextModelEv
CandidateModelSet::evaluateAll|evaluateAll
SYMS

        # Critical string markers — these tell us whether MF-MPI-DIAG / MF-TIME
        # logging is compiled in.  Same _str_cache reused (Lustre-safe).
        for s in 'MF-MPI-DIAG' 'MF-TIME: rank' 'filterRatesMPI fired'; do
            if grep -q "${s}" "${_str_cache}" 2>/dev/null; then
                echo "PROBE: bin_str_present: ${s}"
            else
                echo "PROBE: bin_str_MISSING: ${s}"
            fi
        done

        rm -f "${_str_cache}"
    else
        echo "PROBE: bin_missing or not executable: ${iqtree_bin}"
    fi

    echo "PROBE: ===== source ====="
    local src_dir="${ISO_DIR:-/scratch/rc29/${USER:-as1708}/iqtree3-mf-iso}/src/iqtree3"
    if [[ -d "${src_dir}/.git" ]]; then
        ( cd "${src_dir}" && \
          echo "PROBE: src_branch=$(git branch --show-current 2>/dev/null)" && \
          echo "PROBE: src_head=$(git log --oneline -1 2>/dev/null)" && \
          echo "PROBE: src_describe=$(git describe --always --dirty 2>/dev/null)" )
    fi

    # PBS context.
    echo "PROBE: ===== PBS ====="
    echo "PROBE: pbs_jobid=${PBS_JOBID:-not_in_pbs}"
    echo "PROBE: pbs_queue=${PBS_QUEUE:-?}"
    echo "PROBE: pbs_ncpus=${PBS_NCPUS:-?}"
    echo "PROBE: pbs_node=${PBS_NODENUM:-?}"
    if [[ -s "${PBS_NODEFILE:-/dev/null}" ]]; then
        echo "PROBE: pbs_nodefile_summary:"
        awk '{c[$1]++} END {for (h in c) print "PROBE: pbs_nodefile:  "h" slots="c[h]}' "${PBS_NODEFILE}"
    fi

    echo "PROBE: ===== end probe ====="
    echo ""
}

# Helper to capture a snapshot of all environment relevant to the run.
probe_env() {
    echo "PROBE: ===== env (filtered) ====="
    env | grep -E '^(OMP_|KMP_|OMPI_|GOMP_|MV2_|I_MPI_|MKL_|TMP|PROJECT|USER|HOME|PATH)=' | \
        sort | sed 's/^/PROBE: env: /'
    echo "PROBE: ===== end env ====="
    echo ""
}
