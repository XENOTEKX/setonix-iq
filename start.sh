#!/usr/bin/env bash
#
# Setonix Agent - Quick Start Script
#
# Usage:
#   ./start.sh                 # Push data to GitHub (Action generates dashboard)
#   ./start.sh pipeline        # Run CI/CD pipeline + push data
#   ./start.sh profile [FILE]  # Run profiling + push data
#   ./start.sh generate        # Generate local dashboard preview
#   ./start.sh status          # Show allocation + job status
#
set -euo pipefail

AGENT_DIR="$(cd "$(dirname "$0")" && pwd)"
IQTREE_DIR="${IQTREE_DIR:-/scratch/${PAWSEY_PROJECT:-pawsey0000}/${USER}/iqtree3}"
PIPELINE="$IQTREE_DIR/setonix-ci/run_pipeline.sh"
PROFILER="$IQTREE_DIR/setonix-ci/run_profiling.sh"
DEEP_PROFILER="$IQTREE_DIR/setonix-ci/run_deep_profile.sh"
REPO_URL="$(cd "$AGENT_DIR" && git remote get-url origin 2>/dev/null || echo 'https://github.com/OWNER/setonix-iq.git')"

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; BLUE='\033[0;34m'
CYAN='\033[0;36m'; YELLOW='\033[1;33m'; NC='\033[0m'

banner() {
  echo -e "${CYAN}"
  echo "  ╔═══════════════════════════════════════════╗"
  echo "  ║       Setonix Agent - IQ-TREE Dashboard   ║"
  echo "  ╚═══════════════════════════════════════════╝"
  echo -e "${NC}"
}

check_links() {
  # The new dashboard pipeline reads from logs/runs/*.json and logs/profiles/*.json
  # (the run_pipeline.sh / run_profiling.sh scripts on Setonix write those JSON files
  # directly). No symlinks into scratch are required anymore — just make sure the
  # directories exist so git commits succeed.
  mkdir -p "$AGENT_DIR/logs/runs" "$AGENT_DIR/logs/profiles"
}

sync_to_github() {
  cd "$AGENT_DIR"
  if [[ -d .git ]]; then
    echo -e "${BLUE}Pushing logs/ to GitHub (Actions will build + publish Pages)...${NC}"
    git add -A
    git commit -m "Update runs $(date '+%Y-%m-%d %H:%M')" 2>/dev/null || echo "No changes to commit"
    git push origin main 2>&1 && echo -e "${GREEN}Pushed! GitHub Pages URL will update once the build workflow finishes.${NC}" || echo -e "${RED}Push failed — check auth${NC}"
  else
    echo -e "${YELLOW}Git not initialized. Run: git init && git remote add origin $REPO_URL${NC}"
  fi
}

cmd_start() {
  banner
  check_links
  echo -e "${GREEN}Pushing data to GitHub (Action will generate dashboard)...${NC}"
  sync_to_github
}

cmd_pipeline() {
  banner
  echo -e "${BLUE}Running CI/CD pipeline...${NC}"
  bash "$PIPELINE"
  echo ""
  check_links
  echo -e "${GREEN}Pipeline done. Pushing data...${NC}"
  sync_to_github
}

cmd_profile() {
  banner
  local dataset="${3:-$IQTREE_DIR/test_scripts/test_data/turtle.fa}"
  local threads="${4:-1}"
  echo -e "${BLUE}Running profiling on ${dataset} with ${threads} thread(s)...${NC}"
  bash "$PROFILER" "$dataset" "$threads"
  echo ""
  check_links
  echo -e "${GREEN}Profiling done. Pushing data...${NC}"
  sync_to_github
}

cmd_deepprofile() {
  banner
  local dataset="${3:-test_scripts/test_data/turtle.fa}"
  local threads="${4:-1}"
  local model="${5:-GTR+G4}"
  echo -e "${BLUE}Running deep profiling (CPU + GPU hardware/software)...${NC}"
  echo -e "${CYAN}Dataset: ${dataset} | Threads: ${threads} | Model: ${model}${NC}"
  bash "$DEEP_PROFILER" "$dataset" "$threads" "$model"
  echo ""
  check_links
  echo -e "${GREEN}Deep profiling done. Pushing data...${NC}"
  sync_to_github
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
  echo -e "${BLUE}=== Job Status ===${NC}"
  squeue -u "$USER" --format="%i %j %T %M %l %N" 2>/dev/null || echo "No SLURM access"
  echo ""
  echo -e "${BLUE}=== Allocation Balance ===${NC}"
  pawseyAccountBalance -p "${PAWSEY_PROJECT:-$USER}" 2>/dev/null || echo "N/A"
  echo ""
  pawseyAccountBalance -p "${PAWSEY_PROJECT:-$USER}-gpu" 2>/dev/null || echo "N/A"
  echo ""
  echo -e "${BLUE}=== Disk Quota ===${NC}"
  quota -s 2>/dev/null | head -6 || echo "N/A"
}

# ---- Main ----
case "${1:-start}" in
  start|"")       cmd_start ;;
  pipeline)       cmd_pipeline ;;
  profile)        cmd_profile "$@" ;;
  deepprofile)    cmd_deepprofile "$@" ;;
  generate)       cmd_generate ;;
  status)         cmd_status ;;
  sync)           sync_to_github ;;
  -h|--help)
    echo "Usage: $0 [start|pipeline|profile|deepprofile|generate|status|sync]"
    ;;
  *)
    echo "Unknown command: $1"
    echo "Usage: $0 [start|pipeline|profile|deepprofile|generate|status|sync]"
    exit 1
    ;;
esac
