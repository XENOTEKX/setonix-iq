#!/usr/bin/env bash
#
# Setonix Agent - Quick Start Script
#
# Usage:
#   ./start.sh                 # Generate dashboard + push to GitHub
#   ./start.sh pipeline        # Run CI/CD pipeline, generate + push
#   ./start.sh profile [FILE]  # Run profiling, generate + push
#   ./start.sh generate        # Just regenerate + push
#   ./start.sh status          # Show allocation + job status
#
set -euo pipefail

AGENT_DIR="$(cd "$(dirname "$0")" && pwd)"
IQTREE_DIR="/scratch/pawsey1351/asamuel/iqtree3"
PIPELINE="$IQTREE_DIR/setonix-ci/run_pipeline.sh"
PROFILER="$IQTREE_DIR/setonix-ci/run_profiling.sh"
REPO_URL="https://github.com/XENOTEKX/setonix-iq.git"

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
  local ok=true
  if [[ ! -L "$AGENT_DIR/website/results" ]] || [[ ! -d "$AGENT_DIR/website/results" ]]; then
    echo -e "${YELLOW}Setting up results symlink...${NC}"
    ln -sf "$IQTREE_DIR/setonix-ci/results" "$AGENT_DIR/website/results"
  fi
  if [[ ! -L "$AGENT_DIR/website/profiles" ]] || [[ ! -d "$AGENT_DIR/website/profiles" ]]; then
    echo -e "${YELLOW}Setting up profiles symlink...${NC}"
    ln -sf "$IQTREE_DIR/setonix-ci/profiles" "$AGENT_DIR/website/profiles"
  fi
  if [[ ! -L "$AGENT_DIR/website/assets/flamegraph.svg" ]]; then
    mkdir -p "$AGENT_DIR/website/assets"
    ln -sf "$IQTREE_DIR/setonix-ci/dashboard/flamegraph.svg" "$AGENT_DIR/website/assets/flamegraph.svg" 2>/dev/null || true
  fi
}

sync_to_github() {
  cd "$AGENT_DIR"
  if [[ -d .git ]]; then
    echo -e "${BLUE}Pushing to GitHub (including logs/)...${NC}"
    git add -A
    git commit -m "Update dashboard $(date '+%Y-%m-%d %H:%M')" 2>/dev/null || echo "No changes to commit"
    git push origin main 2>&1 && echo -e "${GREEN}Pushed! On Mac run: cd setonix-iq && git pull && open dashboard.html${NC}" || echo -e "${RED}Push failed — check auth${NC}"
  else
    echo -e "${YELLOW}Git not initialized. Run: git init && git remote add origin $REPO_URL${NC}"
  fi
}

cmd_start() {
  banner
  check_links
  echo -e "${GREEN}Generating dashboard...${NC}"
  python3 "$AGENT_DIR/serve.py"
  sync_to_github
}

cmd_pipeline() {
  banner
  echo -e "${BLUE}Running CI/CD pipeline...${NC}"
  bash "$PIPELINE"
  echo ""
  check_links
  echo -e "${GREEN}Pipeline done. Generating dashboard...${NC}"
  python3 "$AGENT_DIR/serve.py"
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
  echo -e "${GREEN}Profiling done. Generating dashboard...${NC}"
  python3 "$AGENT_DIR/serve.py"
  sync_to_github
}

cmd_generate() {
  check_links
  python3 "$AGENT_DIR/serve.py"
  sync_to_github
}

cmd_status() {
  banner
  echo -e "${BLUE}=== Job Status ===${NC}"
  squeue -u "$USER" --format="%i %j %T %M %l %N" 2>/dev/null || echo "No SLURM access"
  echo ""
  echo -e "${BLUE}=== Allocation Balance ===${NC}"
  pawseyAccountBalance -p pawsey1351 2>/dev/null || echo "N/A"
  echo ""
  pawseyAccountBalance -p pawsey1351-gpu 2>/dev/null || echo "N/A"
  echo ""
  echo -e "${BLUE}=== Disk Quota ===${NC}"
  quota -s 2>/dev/null | head -6 || echo "N/A"
}

# ---- Main ----
case "${1:-start}" in
  start|"")       cmd_start ;;
  pipeline)       cmd_pipeline ;;
  profile)        cmd_profile "$@" ;;
  generate)       cmd_generate ;;
  status)         cmd_status ;;
  sync)           sync_to_github ;;
  -h|--help)
    echo "Usage: $0 [start|pipeline|profile|generate|status|sync]"
    ;;
  *)
    echo "Unknown command: $1"
    echo "Usage: $0 [start|pipeline|profile|generate|status]"
    exit 1
    ;;
esac
