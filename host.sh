#!/usr/bin/env bash
#
# Setonix Agent — Dashboard Host (run on development server)
#
# Serves the IQ-TREE dashboard on a local HTTP port.
# Pulls latest data from GitHub and regenerates periodically via cron.
#
# Quick start:
#   1. Clone the repo:    git clone <repo-url> ~/setonix-iq
#   2. Start the server:  cd ~/setonix-iq && ./host.sh start
#   3. Install cron:      ./host.sh cron
#   4. Open in browser:   http://<hostname>:<PORT>
#
# Commands:
#   ./host.sh start    — Start HTTP server in background (screen)
#   ./host.sh stop     — Stop the HTTP server
#   ./host.sh refresh  — Pull latest data and regenerate dashboard
#   ./host.sh cron     — Install cron job (pull + regenerate every 5 min)
#   ./host.sh uncron   — Remove the cron job
#   ./host.sh status   — Show server and cron status
#   ./host.sh log      — Tail the refresh log
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${DASHBOARD_PORT:-8080}"
SCREEN_NAME="setonix-dashboard"
LOG_FILE="$SCRIPT_DIR/.host.log"
PID_FILE="$SCRIPT_DIR/.host.pid"
CRON_TAG="# setonix-dashboard-refresh"

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; BLUE='\033[0;34m'
CYAN='\033[0;36m'; YELLOW='\033[1;33m'; NC='\033[0m'

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"; }

cmd_refresh() {
  cd "$SCRIPT_DIR"
  log "Pulling latest..."
  git pull --quiet 2>>"$LOG_FILE" || { log "ERROR: git pull failed"; return 1; }

  log "Regenerating dashboard..."
  python3 serve.py >>"$LOG_FILE" 2>&1 || { log "ERROR: serve.py failed"; return 1; }

  log "Dashboard refreshed"
}

cmd_start() {
  # Check if already running
  if screen -list 2>/dev/null | grep -q "$SCREEN_NAME"; then
    echo -e "${YELLOW}Server already running. Use ./host.sh stop first.${NC}"
    echo -e "Access at: ${CYAN}http://$(hostname):${PORT}${NC}"
    return 0
  fi

  # Generate dashboard first
  echo -e "${BLUE}Generating dashboard...${NC}"
  cmd_refresh

  # Ensure docs/ exists
  if [[ ! -f "$SCRIPT_DIR/docs/index.html" ]]; then
    echo -e "${RED}Error: docs/index.html not found. Run serve.py first.${NC}"
    return 1
  fi

  # Start HTTP server in a screen session
  echo -e "${GREEN}Starting dashboard server on port ${PORT}...${NC}"
  screen -dmS "$SCREEN_NAME" bash -c "cd '$SCRIPT_DIR/docs' && python3 -m http.server $PORT 2>&1 | tee -a '$LOG_FILE'"

  # Wait a moment and verify
  sleep 1
  if screen -list 2>/dev/null | grep -q "$SCREEN_NAME"; then
    echo -e "${GREEN}Server running.${NC}"
    echo -e "Access at: ${CYAN}http://$(hostname):${PORT}${NC}"
    log "Server started on port $PORT"
  else
    echo -e "${RED}Server failed to start. Check $LOG_FILE${NC}"
    return 1
  fi
}

cmd_stop() {
  if screen -list 2>/dev/null | grep -q "$SCREEN_NAME"; then
    screen -S "$SCREEN_NAME" -X quit
    echo -e "${GREEN}Server stopped.${NC}"
    log "Server stopped"
  else
    echo -e "${YELLOW}Server not running.${NC}"
  fi
}

cmd_cron() {
  local cron_line="*/5 * * * * cd $SCRIPT_DIR && ./host.sh refresh $CRON_TAG"

  # Check if already installed
  if crontab -l 2>/dev/null | grep -qF "$CRON_TAG"; then
    echo -e "${YELLOW}Cron job already installed.${NC}"
    crontab -l 2>/dev/null | grep "$CRON_TAG"
    return 0
  fi

  # Add cron entry
  (crontab -l 2>/dev/null; echo "$cron_line") | crontab -
  echo -e "${GREEN}Cron job installed (refresh every 5 minutes):${NC}"
  echo "  $cron_line"
  log "Cron job installed"
}

cmd_uncron() {
  if crontab -l 2>/dev/null | grep -qF "$CRON_TAG"; then
    crontab -l 2>/dev/null | grep -vF "$CRON_TAG" | crontab -
    echo -e "${GREEN}Cron job removed.${NC}"
    log "Cron job removed"
  else
    echo -e "${YELLOW}No cron job found.${NC}"
  fi
}

cmd_status() {
  echo -e "${BLUE}=== Dashboard Server ===${NC}"
  if screen -list 2>/dev/null | grep -q "$SCREEN_NAME"; then
    echo -e "  Status: ${GREEN}Running${NC}"
    echo -e "  URL:    ${CYAN}http://$(hostname):${PORT}${NC}"
  else
    echo -e "  Status: ${RED}Stopped${NC}"
  fi

  echo ""
  echo -e "${BLUE}=== Cron Job ===${NC}"
  if crontab -l 2>/dev/null | grep -qF "$CRON_TAG"; then
    echo -e "  Status: ${GREEN}Installed${NC}"
    crontab -l 2>/dev/null | grep "$CRON_TAG" | sed 's/^/  /'
  else
    echo -e "  Status: ${YELLOW}Not installed${NC} (run ./host.sh cron)"
  fi

  echo ""
  echo -e "${BLUE}=== Last Refresh ===${NC}"
  if [[ -f "$LOG_FILE" ]]; then
    tail -3 "$LOG_FILE" | sed 's/^/  /'
  else
    echo "  No log yet"
  fi
}

cmd_log() {
  if [[ -f "$LOG_FILE" ]]; then
    tail -30 "$LOG_FILE"
  else
    echo "No log file yet."
  fi
}

# ---- Main ----
case "${1:-start}" in
  start)    cmd_start ;;
  stop)     cmd_stop ;;
  refresh)  cmd_refresh ;;
  cron)     cmd_cron ;;
  uncron)   cmd_uncron ;;
  status)   cmd_status ;;
  log)      cmd_log ;;
  -h|--help)
    echo "Usage: $0 [start|stop|refresh|cron|uncron|status|log]"
    echo ""
    echo "Environment:"
    echo "  DASHBOARD_PORT=8080   HTTP port (default: 8080)"
    ;;
  *)
    echo "Unknown command: $1"
    echo "Usage: $0 [start|stop|refresh|cron|uncron|status|log]"
    exit 1
    ;;
esac
