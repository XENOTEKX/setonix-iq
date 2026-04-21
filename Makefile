##
##  Setonix IQ-TREE — Build & Profiling Makefile
##
##  Targets:
##    make build             — standard RelWithDebInfo build of IQ-TREE
##    make build-profiling   — build with -fno-omit-frame-pointer (correct perf unwinding)
##    make profile           — run perf profiling, regenerate dashboard, push
##    make deep-profile      — run deep profiling (CPU + GPU hardware counters)
##    make dashboard         — regenerate dashboard HTML and push to GitHub
##    make status            — show HPC job and allocation status
##    make clean             — remove build directories
##
##  Usage on HPC (Setonix):
##    cd ~/setonix-iq && make build-profiling
##
##  Usage on macOS (dashboard only):
##    make dashboard
##

SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

# ── Paths ────────────────────────────────────────────────────────────────────
AGENT_DIR   := $(shell cd "$(dir $(abspath $(lastword $(MAKEFILE_LIST))))" && pwd)
IQTREE_DIR  := $(or $(IQTREE_DIR),/scratch/$(or $(PAWSEY_PROJECT),pawsey0000)/$(USER)/iqtree3)

BUILD_DIR         := $(IQTREE_DIR)/build
BUILD_PROFILING   := $(IQTREE_DIR)/build-profiling

DATASET   ?= $(IQTREE_DIR)/test_scripts/test_data/turtle.fa
THREADS   ?= 1
MODEL     ?= GTR+G4

# ── Compiler flags ────────────────────────────────────────────────────────────
# Standard: RelWithDebInfo gives -O2 -g — good for production profiling
CMAKE_STANDARD := \
	-DCMAKE_BUILD_TYPE=RelWithDebInfo

# Profiling: keep frame pointers so `perf -g` can unwind the call stack.
# Without this flag perf cannot trace through -O3 inlined frames and
# reports [unknown] for 70%+ of samples.
CMAKE_PROFILING := \
	-DCMAKE_BUILD_TYPE=RelWithDebInfo \
	-DCMAKE_CXX_FLAGS="-fno-omit-frame-pointer -g" \
	-DCMAKE_C_FLAGS="-fno-omit-frame-pointer -g"

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
	@printf '$(BOLD)Setonix IQ-TREE Build & Profiling$(NC)\n'
	@echo ""
	@printf '  $(GREEN)make build$(NC)            Standard build (RelWithDebInfo)\n'
	@printf '  $(GREEN)make build-profiling$(NC)  Build with frame pointers (fixes perf [unknown])\n'
	@printf '  $(GREEN)make profile$(NC)          Profile with perf, update dashboard\n'
	@printf '  $(GREEN)make deep-profile$(NC)     Deep profile (CPU + GPU hardware counters)\n'
	@printf '  $(GREEN)make dashboard$(NC)        Validate + build static site, push to GitHub\n'
	@printf '  $(GREEN)make test$(NC)             Run pytest suite over logs/*.json\n'
	@printf '  $(GREEN)make status$(NC)           Show HPC job and allocation status\n'
	@printf '  $(GREEN)make clean$(NC)            Remove build directories\n'
	@echo ""
	@printf '  Variables (override with make DATASET=... THREADS=... MODEL=...):\n'
	@printf '    DATASET  = $(DATASET)\n'
	@printf '    THREADS  = $(THREADS)\n'
	@printf '    MODEL    = $(MODEL)\n'
	@echo ""

# ── Build targets ─────────────────────────────────────────────────────────────

.PHONY: build
build:
	$(call header,Building IQ-TREE (standard RelWithDebInfo))
	mkdir -p "$(BUILD_DIR)"
	cd "$(BUILD_DIR)" && cmake .. $(CMAKE_STANDARD)
	cd "$(BUILD_DIR)" && make -j$$(nproc)
	@printf '$(GREEN)Build complete: $(BUILD_DIR)/iqtree3$(NC)\n'

.PHONY: build-profiling
build-profiling:
	$(call header,Building IQ-TREE with -fno-omit-frame-pointer)
	@echo "  Frame pointers preserved — perf call-graph unwinding will work correctly."
	@echo "  Binary: $(BUILD_PROFILING)/iqtree3"
	mkdir -p "$(BUILD_PROFILING)"
	cd "$(BUILD_PROFILING)" && cmake .. $(CMAKE_PROFILING)
	cd "$(BUILD_PROFILING)" && make -j$$(nproc)
	@printf '$(GREEN)Profiling build complete: $(BUILD_PROFILING)/iqtree3$(NC)\n'
	@echo ""
	@echo "  To verify frame pointers are present:"
	@echo "    objdump -d $(BUILD_PROFILING)/iqtree3 | grep -A2 'push.*rbp' | head -20"

.PHONY: clean
clean:
	$(call header,Cleaning build directories)
	rm -rf "$(BUILD_DIR)" "$(BUILD_PROFILING)"
	@printf '$(GREEN)Cleaned.$(NC)\n'

# ── Profiling targets ─────────────────────────────────────────────────────────

.PHONY: profile
profile:
	$(call header,Running perf profiling)
	bash "$(IQTREE_DIR)/setonix-ci/run_profiling.sh" "$(DATASET)" "$(THREADS)"
	@$(MAKE) --no-print-directory dashboard

.PHONY: deep-profile
deep-profile:
	$(call header,Running deep profiling (CPU + GPU hardware counters))
	bash "$(IQTREE_DIR)/setonix-ci/run_deep_profile.sh" "$(DATASET)" "$(THREADS)" "$(MODEL)"
	@$(MAKE) --no-print-directory dashboard

# ── Dashboard targets ─────────────────────────────────────────────────────────

.PHONY: dashboard
dashboard:
	$(call header,Validating + building static dashboard)
	python3 "$(AGENT_DIR)/tools/validate.py"
	python3 "$(AGENT_DIR)/tools/build.py"
	@$(MAKE) --no-print-directory push

.PHONY: test
test:
	$(call header,Running pytest)
	python3 -m pytest "$(AGENT_DIR)/tests" -v

.PHONY: push
push:
	$(call header,Pushing to GitHub (Pages will rebuild automatically))
	cd "$(AGENT_DIR)" && \
	  git add -A && \
	  git commit -m "Update dashboard $$(date '+%Y-%m-%d %H:%M')" 2>/dev/null || true && \
	  git push origin main && \
	  printf '$(GREEN)Pushed — GitHub Actions will publish to Pages.$(NC)\n'

# ── Status ────────────────────────────────────────────────────────────────────

.PHONY: status
status:
	bash "$(AGENT_DIR)/start.sh" status
