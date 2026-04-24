##
##  Gadi-IQ — Build & Profiling Makefile (NCI / Intel / PBS)
##
##  Targets:
##    make build             — RelWithDebInfo build of IQ-TREE on Gadi
##    make build-profiling   — build with -fno-omit-frame-pointer (perf unwinding)
##    make profile           — submit quick perf profiling, refresh dashboard
##    make deep-profile      — submit PBS deep profile (perf + VTune)
##    make dashboard         — validate + build docs/, push to GitHub
##    make test              — run pytest suite over logs/*.json
##    make status            — show PBS job + NCI allocation status
##    make harvest           — rsync scratch profile dirs into local mirror + enrich logs
##    make clean             — remove build directories
##

SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

# ── Paths ────────────────────────────────────────────────────────────────────
AGENT_DIR   := $(shell cd "$(dir $(abspath $(lastword $(MAKEFILE_LIST))))" && pwd)
PROJECT     ?= $(or $(PROJECT),rc29)
IQTREE_DIR  := $(or $(IQTREE_DIR),/scratch/$(PROJECT)/$(USER)/iqtree3)

BUILD_DIR         := $(IQTREE_DIR)/build
BUILD_PROFILING   := $(IQTREE_DIR)/build-profiling

DATASET   ?= $(IQTREE_DIR)/test_scripts/test_data/turtle.fa
THREADS   ?= 1
MODEL     ?= GTR+G4

# Optional: pre-load Intel VTune + LLVM-based compiler via Gadi's module system.
# Safe to leave empty on non-Gadi systems — make simply runs the commands.
MODULES   ?= intel-vtune/2024.2.0 intel-compiler-llvm/2024.2.0
MODCMD    := if command -v module >/dev/null 2>&1; then module load $(MODULES) 2>/dev/null || true; fi

# ── Compiler flags ────────────────────────────────────────────────────────────
# Sapphire Rapids is the target — build with -xSAPPHIRERAPIDS when icx is
# available, fall back to portable -O3 otherwise.
SR_FLAGS      := -O3 -xSAPPHIRERAPIDS -fno-omit-frame-pointer
CMAKE_STANDARD := \
	-DCMAKE_BUILD_TYPE=RelWithDebInfo

CMAKE_PROFILING := \
	-DCMAKE_BUILD_TYPE=RelWithDebInfo \
	-DCMAKE_CXX_FLAGS="$(SR_FLAGS) -g" \
	-DCMAKE_C_FLAGS="$(SR_FLAGS) -g"

# ── Helpers ───────────────────────────────────────────────────────────────────
BOLD  := \033[1m
GREEN := \033[0;32m
BLUE  := \033[0;34m
CYAN  := \033[0;36m
NC    := \033[0m

define header
	@printf '$(CYAN)$(BOLD)>> $(1)$(NC)\n'
endef

# ── Targets ───────────────────────────────────────────────────────────────────

.PHONY: help
help:
	@echo ""
	@printf '$(BOLD)Gadi-IQ — IQ-TREE Build & Profiling$(NC)\n'
	@echo ""
	@printf '  $(GREEN)make build$(NC)            Standard RelWithDebInfo build\n'
	@printf '  $(GREEN)make build-profiling$(NC)  Build with frame pointers (fixes perf [unknown])\n'
	@printf '  $(GREEN)make profile$(NC)          Quick perf profiling on Gadi (interactive PBS)\n'
	@printf '  $(GREEN)make deep-profile$(NC)     Submit deep profile PBS job (perf + VTune)\n'
	@printf '  $(GREEN)make batch$(NC)            Fan out mega profile across thread counts\n'
	@printf '  $(GREEN)make harvest$(NC)          Rsync wave data from Gadi scratch + enrich logs\n'
	@printf '  $(GREEN)make dashboard$(NC)        Validate + build docs/, push to GitHub\n'
	@printf '  $(GREEN)make test$(NC)             Run pytest suite over logs/*.json\n'
	@printf '  $(GREEN)make status$(NC)           Show PBS jobs + NCI allocation balance\n'
	@printf '  $(GREEN)make clean$(NC)            Remove build directories\n'
	@echo ""
	@printf '  Variables (override with make PROJECT=... DATASET=... THREADS=... MODEL=...):\n'
	@printf '    PROJECT  = $(PROJECT)\n'
	@printf '    DATASET  = $(DATASET)\n'
	@printf '    THREADS  = $(THREADS)\n'
	@printf '    MODEL    = $(MODEL)\n'
	@echo ""

# ── Build targets ─────────────────────────────────────────────────────────────

.PHONY: build
build:
	$(call header,Building IQ-TREE (RelWithDebInfo) on Gadi)
	@$(MODCMD); \
	mkdir -p "$(BUILD_DIR)"; \
	cd "$(BUILD_DIR)" && cmake .. $(CMAKE_STANDARD) && make -j$$(nproc)
	@printf '$(GREEN)Build complete: $(BUILD_DIR)/iqtree3$(NC)\n'

.PHONY: build-profiling
build-profiling:
	$(call header,Building IQ-TREE with -fno-omit-frame-pointer)
	@$(MODCMD); \
	mkdir -p "$(BUILD_PROFILING)"; \
	cd "$(BUILD_PROFILING)" && cmake .. $(CMAKE_PROFILING) && make -j$$(nproc)
	@printf '$(GREEN)Profiling build complete: $(BUILD_PROFILING)/iqtree3$(NC)\n'

.PHONY: clean
clean:
	$(call header,Cleaning build directories)
	rm -rf "$(BUILD_DIR)" "$(BUILD_PROFILING)"

# ── Profiling targets ─────────────────────────────────────────────────────────

.PHONY: profile
profile:
	$(call header,Running perf profiling on Gadi)
	bash "$(IQTREE_DIR)/gadi-ci/run_profiling.sh" "$(DATASET)" "$(THREADS)" "$(MODEL)"
	@$(MAKE) --no-print-directory dashboard

.PHONY: deep-profile
deep-profile:
	$(call header,Submitting deep profile PBS job)
	qsub -v "THREADS=$(THREADS),PROJECT=$(PROJECT)" \
	     "$(IQTREE_DIR)/gadi-ci/run_mega_profile.sh"

.PHONY: batch
batch:
	$(call header,Fanning out mega profile batch via PBS)
	bash "$(IQTREE_DIR)/gadi-ci/submit_mega_batch.sh"

.PHONY: harvest
harvest:
	$(call header,Harvesting scratch profile dirs)
	PROFILE_ROOT="$(IQTREE_DIR)/gadi-ci/profiles" \
	SCRATCH_DIR="$(IQTREE_DIR)" \
	BENCHMARKS_DIR="$(IQTREE_DIR)/benchmarks" \
	python3 "$(AGENT_DIR)/tools/harvest_scratch.py"

# ── Dashboard + tests ─────────────────────────────────────────────────────────

.PHONY: dashboard
dashboard:
	$(call header,Validating + building dashboard)
	python3 "$(AGENT_DIR)/tools/validate.py"
	python3 "$(AGENT_DIR)/tools/build.py"
	cd "$(AGENT_DIR)" && git add -A \
	    && (git commit -m "dashboard update $$(date '+%Y-%m-%d %H:%M')" || true) \
	    && git push origin "$$(git rev-parse --abbrev-ref HEAD)"

.PHONY: test
test:
	$(call header,Running pytest)
	cd "$(AGENT_DIR)" && python3 -m pytest tests/ -v

# ── Status ────────────────────────────────────────────────────────────────────

.PHONY: status
status:
	@printf '$(CYAN)=== PBS jobs ($$USER) ===$(NC)\n'
	@qstat -u "$$USER" 2>/dev/null || echo "PBS not available"
	@printf '\n$(CYAN)=== NCI allocation (project $(PROJECT)) ===$(NC)\n'
	@nci_account 2>/dev/null | head -25 || echo "nci_account not available"
	@printf '\n$(CYAN)=== Disk quota ===$(NC)\n'
	@quota -s 2>/dev/null | head -6 || true
