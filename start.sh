#!/usr/bin/env bash
#
# Gadi-IQ Agent — Quick Start Script
#
# Gadi/NCI port of start.sh. Runs the PBS pipeline and pushes data to GitHub.
#
# Usage:
#   ./start.sh                 # Push data to GitHub (Action generates dashboard)
#   ./start.sh pipeline        # Run CI/CD pipeline + push data
#   ./start.sh profile [FILE]  # Run profiling + push data
#   ./start.sh deepprofile     # Submit PBS deep profile job
#   ./start.sh batch [T ...]   # Fan out mega profile across thread counts
#   ./start.sh generate        # Generate local dashboard preview
#   ./start.sh status          # Show PBS job status + NCI allocation
#
set -euo pipefail

AGENT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT="${PROJECT:-rc29}"
IQTREE_DIR="${IQTREE_DIR:-/scratch/${PROJECT}/${USER}/iqtree3}"
PIPELINE="${PIPELINE:-$IQTREE_DIR/gadi-ci/run_pipeline.sh}"
PROFILER="${PROFILER:-$IQTREE_DIR/gadi-ci/run_profiling.sh}"
DEEP_PROFILER="${DEEP_PROFILER:-$IQTREE_DIR/gadi-ci/run_mega_profile.sh}"
BATCH="${BATCH:-$IQTREE_DIR/gadi-ci/submit_mega_batch.sh}"
REPO_URL="$(cd "$AGENT_DIR" && git remote get-url origin 2>/dev/null || echo 'https://github.com/OWNER/setonix-iq.git')"

RED='\033[0;31m'; GREEN='\033[0;32m'; BLUE='\033[0;34m'
CYAN='\033[0;36m'; YELLOW='\033[1;33m'; NC='\033[0m'

banner() {
  echo -e "${CYAN}"
  echo "  ╔═══════════════════════════════════════════╗"
  echo "  ║         Gadi-IQ — IQ-TREE Dashboard       ║"
  echo "  ╚═══════════════════════════════════════════╝"
  echo -e "${NC}"
}

check_links() {
  mkdir -p "$AGENT_DIR/logs/runs" "$AGENT_DIR/logs/profiles"
}

sync_to_github() {
  cd "$AGENT_DIR"
  if [[ -d .git ]]; then
    echo -e "${BLUE}Pushing logs/ to GitHub (Actions will build + publish Pages)...${NC}"
    git add -A
    git commit -m "Update runs $(date '+%Y-%m-%d %H:%M')" 2>/dev/null || echo "No changes to commit"
    local branch; branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"
    git push origin "${branch}" 2>&1 \
      && echo -e "${GREEN}Pushed branch ${branch}. Pages will update after workflow finishes.${NC}" \
      || echo -e "${RED}Push failed — check auth${NC}"
  else
    echo -e "${YELLOW}Git not initialized. Run: git init && git remote add origin $REPO_URL${NC}"
  fi
}

cmd_start()    { banner; check_links; sync_to_github; }

cmd_pipeline() {
  banner
  if [[ ! -x "$PIPELINE" ]]; then
    echo -e "${RED}Pipeline script missing: $PIPELINE${NC}"
    echo -e "${YELLOW}Tip: rsync ./gadi-ci/ to ${IQTREE_DIR}/gadi-ci/ first.${NC}"
    exit 1
  fi
  echo -e "${BLUE}Running Gadi CI pipeline...${NC}"
  bash "$PIPELINE"
  check_links
  sync_to_github
}

cmd_profile() {
  banner
  local dataset="${3:-turtle.fa}"
  local threads="${4:-1}"
  echo -e "${BLUE}Running profiling on ${dataset} with ${threads} thread(s)...${NC}"
  bash "$PROFILER" "$dataset" "$threads"
  check_links
  sync_to_github
}

cmd_deepprofile() {
  banner
  local threads="${3:-48}"
  echo -e "${BLUE}Submitting deep profile PBS job (threads=${threads})...${NC}"
  qsub -v "THREADS=${threads},PROJECT=${PROJECT}" "$DEEP_PROFILER"
}

cmd_batch() {
  banner
  shift || true
  echo -e "${BLUE}Fanning out mega profile across thread counts: ${*:-default 4/8/16/24/48}${NC}"
  bash "$BATCH" "$@"
}

cmd_generate() {
  check_links
  echo -e "${GREEN}Building local dashboard preview to docs/ ...${NC}"
  python3 "$AGENT_DIR/tools/validate.py"
  python3 "$AGENT_DIR/tools/build.py"
  echo -e "${GREEN}Done. Open docs/index.html or run: python3 -m http.server -d docs${NC}"
}

cmd_status() {
  banner
  echo -e "${BLUE}=== PBS Job Status (user $USER) ===${NC}"
  qstat -u "$USER" 2>/dev/null || echo "No PBS access"
  echo ""
  if command -v nqstat >/dev/null 2>&1; then
    echo -e "${BLUE}=== nqstat ===${NC}"
    nqstat "$USER" 2>/dev/null | head -20 || true
    echo ""
  fi
  echo -e "${BLUE}=== NCI Allocation (project ${PROJECT}) ===${NC}"
  nci_account 2>/dev/null | head -30 || echo "nci_account not available"
  echo ""
  echo -e "${BLUE}=== Disk Quota ===${NC}"
  quota -s 2>/dev/null | head -6 || echo "N/A"
}

case "${1:-start}" in
  start|"")       cmd_start ;;
  pipeline)       cmd_pipeline ;;
  profile)        cmd_profile "$@" ;;
  deepprofile)    cmd_deepprofile "$@" ;;
  batch)          cmd_batch "$@" ;;
  generate)       cmd_generate ;;
  status)         cmd_status ;;
  sync)           sync_to_github ;;
  -h|--help)
    echo "Usage: $0 [start|pipeline|profile|deepprofile|batch|generate|status|sync]"
    ;;
  *)
    echo "Unknown command: $1"
    echo "Usage: $0 [start|pipeline|profile|deepprofile|batch|generate|status|sync]"
    exit 1
    ;;
esac
