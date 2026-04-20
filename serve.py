#!/usr/bin/env python3
"""
Setonix Agent Dashboard Generator

Generates a self-contained HTML dashboard with all data embedded.
Output goes to docs/index.html for GitHub Pages deployment.

Why not a server? Setonix (Pawsey HPC) disables SSH TCP forwarding
on shared login nodes for security, so port forwarding doesn't work.

Usage:
  ./serve.py              # generate docs/index.html
"""

import json
import os
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEBSITE_DIR = os.path.join(SCRIPT_DIR, 'website')
API_DIR = os.path.join(WEBSITE_DIR, 'api')
DOCS_DIR = os.path.join(SCRIPT_DIR, 'docs')
LOGS_DIR = os.path.join(SCRIPT_DIR, 'logs')

sys.path.insert(0, API_DIR)
from data import get_all_runs, get_all_profiles


def generate():
    """Generate self-contained dashboard HTML with embedded data."""
    runs = get_all_runs()
    profiles = get_all_profiles()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Write JSON for reference
    json_path = os.path.join(API_DIR, 'runs.json')
    with open(json_path, 'w') as f:
        json.dump(runs, f, indent=2)

    # Export individual per-run JSON files to logs/runs/
    runs_log_dir = os.path.join(LOGS_DIR, 'runs')
    os.makedirs(runs_log_dir, exist_ok=True)
    for run in runs:
        run_path = os.path.join(runs_log_dir, '%s.json' % run['run_id'])
        with open(run_path, 'w') as f:
            json.dump(run, f, indent=2)

    # Export individual per-profile JSON files to logs/profiles/
    profiles_log_dir = os.path.join(LOGS_DIR, 'profiles')
    os.makedirs(profiles_log_dir, exist_ok=True)
    for prof in profiles:
        prof_path = os.path.join(profiles_log_dir, '%s.json' % prof['profile_id'])
        with open(prof_path, 'w') as f:
            json.dump(prof, f, indent=2)

    # Read the template
    template_path = os.path.join(WEBSITE_DIR, 'index.html')
    with open(template_path) as f:
        template = f.read()

    # Find the <script> block and replace the data loading
    # We'll inject DATA right at the start of the script and replace loadData
    data_json = json.dumps(runs)
    profiles_json = json.dumps(profiles)

    # Build the replacement script block
    old_script_start = '// ============ State ============'
    old_script_end = 'loadData();'

    start_idx = template.index(old_script_start)
    end_idx = template.rindex(old_script_end) + len(old_script_end)

    new_script = build_script(data_json, profiles_json, now)

    output_html = template[:start_idx] + new_script + template[end_idx:]

    os.makedirs(DOCS_DIR, exist_ok=True)
    output_path = os.path.join(DOCS_DIR, 'index.html')
    with open(output_path, 'w') as f:
        f.write(output_html)

    # Also write to root for convenience
    root_path = os.path.join(SCRIPT_DIR, 'dashboard.html')
    with open(root_path, 'w') as f:
        f.write(output_html)

    print("Generated: %s" % output_path)
    print("Also: %s" % root_path)
    print("Runs: %d | Profiles: %d | Generated: %s" % (len(runs), len(profiles), now))
    return output_path


def build_script(data_json, profiles_json, generated_time):
    """Build the complete JavaScript block with embedded data."""
    return '''// ============ State ============
var DATA = %s;
var PROFILES = %s;
var currentRunIdx = DATA.length - 1;
var currentProfileIdx = PROFILES.length - 1;
var currentDataset = null; // selected dataset for run selector
var _profViewingRun = null; // tracking which run is shown in profiling page
var charts = {};
var expandedRunId = null;
var hotspotSelection = {};
var microarchSelection = {};

// ============ Navigation ============
document.querySelectorAll('.sidebar nav a').forEach(function(a) {
  a.addEventListener('click', function(e) {
    e.preventDefault();
    showPage(a.dataset.page);
  });
});

function showPage(page) {
  document.querySelectorAll('.sidebar nav a').forEach(function(x) { x.classList.remove('active'); });
  var link = document.querySelector('.sidebar nav a[data-page="' + page + '"]');
  if (link) link.classList.add('active');
  document.querySelectorAll('.page').forEach(function(p) { p.classList.remove('active'); });
  var pageEl = document.getElementById('page-' + page);
  if (pageEl) pageEl.classList.add('active');
}

// ============ Clipboard Helper ============
function copyText(text, btn) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(function() {
      var orig = btn.textContent;
      btn.textContent = 'Copied!'; btn.classList.add('copied');
      setTimeout(function() { btn.textContent = orig; btn.classList.remove('copied'); }, 1500);
    }).catch(function() { fallbackCopy(text, btn); });
  } else { fallbackCopy(text, btn); }
}
function fallbackCopy(text, btn) {
  var ta = document.createElement('textarea');
  ta.value = text; ta.style.position = 'fixed'; ta.style.left = '-9999px';
  document.body.appendChild(ta); ta.select();
  try { document.execCommand('copy'); btn.textContent = 'Copied!'; btn.classList.add('copied');
    setTimeout(function() { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 1500);
  } catch(e) {}
  document.body.removeChild(ta);
}

function copyAllCmds(btn) {
  var run = DATA[currentRunIdx];
  var script = '#!/bin/bash\\n# Pipeline commands from Run ' + run.run_id + '\\n# ' + (run.env.date || '') + ' on ' + (run.env.hostname || '') + '\\n\\n' +
    run.timing.map(function(t) { return t.command; }).join('\\n') + '\\n';
  copyText(script, btn);
}

function copySingleCmd(text, btn) { copyText(text, btn); }

// ============ Formatting ============
function fmtTime(s) {
  if (s >= 3600) return (s / 3600).toFixed(1) + 'h';
  if (s >= 60) return (s / 60).toFixed(1) + 'm';
  return s.toFixed(2) + 's';
}
function escHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function escAttr(s) { return s.replace(/\\\\/g,'\\\\\\\\').replace(/'/g,"\\\\'").replace(/"/g,'&quot;'); }

// ============ Data Loading ============
function loadData() {
  if (DATA.length === 0) {
    document.getElementById('overviewSubtitle').textContent = 'No pipeline runs found. Run the pipeline first.';
    return;
  }
  currentRunIdx = DATA.length - 1;
  populateAllSelectors();
  initChartSelectors();
  renderHotspotSelector();
  renderMicroarchSelector();
  renderAll();
  renderRunsList();
  renderProfilesList();
  document.getElementById('serverDot').style.background = 'var(--green)';
  document.getElementById('serverStatus').textContent = 'Static (GitHub Pages)';
  document.getElementById('lastUpdate').textContent = 'Generated: %s';
}

function populateAllSelectors() {
  var selectors = ['testRunSelector', 'profRunSelector', 'gpuRunSelector', 'envRunSelector'];
  selectors.forEach(function(id) {
    var sel = document.getElementById(id);
    if (!sel) return;
    sel.innerHTML = DATA.map(function(r, i) {
      var label = r.label || r.run_id;
      return '<option value="' + i + '"' + (i === currentRunIdx ? ' selected' : '') + '>' + escHtml(label) + ' (' + fmtTime(r.summary.total_time) + ')</option>';
    }).join('');
  });
  // Update tab counts
  var runsCount = document.getElementById('runsTabCount');
  var profsCount = document.getElementById('profilesTabCount');
  if (runsCount) runsCount.textContent = DATA.length;
  if (profsCount) profsCount.textContent = PROFILES.length;
}

function switchRun(idx) {
  currentRunIdx = parseInt(idx);
  populateAllSelectors();
  renderAll();
}

function refreshData() { loadData(); }

// ============ Rendering ============
function renderAll() {
  var run = DATA[currentRunIdx];
  renderRunSelector();
  renderOverview(run);
  renderTests(run);
  renderProfiling(run);
  renderGPU(run);
  renderAllocation();
  renderEnvironment(run);
  renderQuickCmds(run);
}

// ============ Run Selector ============
function buildRunIndex() {
  // Group runs by dataset, mapping dataset->thread->DATA index
  var index = {};
  DATA.forEach(function(r, i) {
    r.timing.forEach(function(t) {
      var p = parseCmd(t.command);
      var ds = p.dataset;
      if (ds === 'unknown') ds = r.label || r.run_id;
      if (!index[ds]) index[ds] = {};
      index[ds][p.threads] = i;
    });
  });
  return index;
}

function renderRunSelector() {
  var tabsEl = document.getElementById('runDsTabs');
  var stopsEl = document.getElementById('runThreadStops');
  var infoEl = document.getElementById('runSelectorInfo');
  if (!tabsEl || !stopsEl) return;

  var runIndex = buildRunIndex();
  var dsKeys = Object.keys(runIndex).sort(function(a, b) {
    // Sort benchmark datasets first, then others
    var aIsBench = false, bIsBench = false;
    var aThreads = Object.keys(runIndex[a]);
    var bThreads = Object.keys(runIndex[b]);
    aThreads.forEach(function(t) { var r = DATA[runIndex[a][t]]; if (r.run_type === 'cpu_baseline' || r.run_type === 'gpu_benchmark') aIsBench = true; });
    bThreads.forEach(function(t) { var r = DATA[runIndex[b][t]]; if (r.run_type === 'cpu_baseline' || r.run_type === 'gpu_benchmark') bIsBench = true; });
    if (aIsBench && !bIsBench) return -1;
    if (!aIsBench && bIsBench) return 1;
    return a.localeCompare(b);
  });

  // Auto-select dataset if not set or invalid
  if (!currentDataset || dsKeys.indexOf(currentDataset) === -1) {
    var found = false;
    dsKeys.forEach(function(ds) {
      var threads = Object.keys(runIndex[ds]);
      threads.forEach(function(t) {
        if (runIndex[ds][t] === currentRunIdx) { currentDataset = ds; found = true; }
      });
    });
    if (!found) currentDataset = dsKeys[0];
  }

  // Render dataset tabs
  tabsEl.innerHTML = dsKeys.map(function(ds) {
    var short = ds.replace('.fa', '').replace('.phy', '');
    var count = Object.keys(runIndex[ds]).length;
    var isActive = ds === currentDataset;
    return '<div class="run-ds-tab' + (isActive ? ' active' : '') + '" onclick="selectDataset(\\''+escAttr(ds)+'\\')"><span>' + escHtml(short) + '</span><span class="ds-count">' + count + '</span></div>';
  }).join('');

  // Get thread configs for current dataset
  var dsRuns = runIndex[currentDataset] || {};
  var threadKeys = Object.keys(dsRuns).map(function(t) { return parseInt(t); }).sort(function(a, b) { return a - b; });

  if (threadKeys.length === 0) {
    stopsEl.innerHTML = '<div class="run-no-threads">No thread variants</div>';
    document.getElementById('trackFill').style.width = '0';
    return;
  }

  // Find active thread index
  var activeThreadIdx = 0;
  threadKeys.forEach(function(t, idx) {
    if (dsRuns[t] === currentRunIdx) activeThreadIdx = idx;
  });

  // Track fill percentage
  var fillPct = threadKeys.length > 1 ? (activeThreadIdx / (threadKeys.length - 1) * 100) : 0;
  document.getElementById('trackFill').style.width = fillPct + '%%';

  // Render thread stops
  stopsEl.innerHTML = threadKeys.map(function(t, idx) {
    var runIdx = dsRuns[t];
    var isActive = runIdx === currentRunIdx;
    return '<div class="run-thread-stop' + (isActive ? ' active' : '') + '" onclick="selectThread(' + t + ')">' +
      '<div class="ts-threads">' + t + 'T</div>' +
    '</div>';
  }).join('');

  // Info panel: show current run details
  var curRun = DATA[currentRunIdx];
  var curCmd = curRun.timing.length > 0 ? curRun.timing[0] : null;
  var curInfo = curCmd ? parseCmd(curCmd.command) : null;
  var model = '';
  if (curRun.modelfinder && curRun.modelfinder.model_selected) {
    model = curRun.modelfinder.model_selected.replace(' chosen according to BIC', '');
  } else if (curInfo) {
    model = curInfo.model;
  }
  infoEl.innerHTML =
    '<span>Model: <span class="rsi-value">' + escHtml(model || 'AUTO') + '</span></span>' +
    '<span>Time: <span class="rsi-accent">' + fmtTime(curRun.summary.total_time) + '</span></span>' +
    '<span>' + escHtml(curRun.env.hostname || '') + '</span>';
}

function selectDataset(ds) {
  currentDataset = ds;
  // Switch to the first thread in this dataset
  var runIndex = buildRunIndex();
  var dsRuns = runIndex[ds] || {};
  var threadKeys = Object.keys(dsRuns).map(function(t) { return parseInt(t); }).sort(function(a, b) { return a - b; });
  if (threadKeys.length > 0) {
    currentRunIdx = dsRuns[threadKeys[0]];
  }
  populateAllSelectors();
  renderAll();
}

function selectThread(t) {
  var runIndex = buildRunIndex();
  var dsRuns = runIndex[currentDataset] || {};
  if (dsRuns[t] != null) {
    currentRunIdx = dsRuns[t];
    populateAllSelectors();
    renderAll();
  }
}

function renderOverview(run) {
  var s = run.summary;
  var profile = (run.profile && run.profile.metrics) || {};

  // Show label/description if available, otherwise run_id
  var runLabel = run.label || run.run_id;
  var subtitleParts = [runLabel, run.env.date || 'N/A', run.env.hostname || 'N/A'];
  if (run.env.cpu) subtitleParts.push(run.env.cpu.replace(' 64-Core Processor', ''));
  document.getElementById('overviewSubtitle').textContent = subtitleParts.join(' | ');

  // Aggregate test results across ALL runs (many baseline runs have no verify data)
  var totalTests = 0, totalPass = 0, totalFail = 0;
  DATA.forEach(function(r) {
    var rs = r.summary;
    totalPass += rs.pass;
    totalFail += rs.fail;
    totalTests += rs.pass + rs.fail;
  });
  var allTestsPass = totalFail === 0;

  var badge = document.getElementById('statusBadge');
  if (totalTests === 0) {
    badge.textContent = 'BASELINES';
    badge.className = 'badge badge-pass';
  } else {
    badge.textContent = allTestsPass ? 'ALL PASS' : totalFail + ' FAILED';
    badge.className = 'badge ' + (allTestsPass ? 'badge-pass' : 'badge-fail');
  }

  // Count benchmark runs and datasets
  var benchmarkCount = 0;
  var datasets = {};
  DATA.forEach(function(r) {
    if (r.run_type !== 'cpu_baseline' && r.run_type !== 'gpu_benchmark') return;
    benchmarkCount++;
    r.timing.forEach(function(t) { var p = parseCmd(t.command); if (p.dataset !== 'unknown') datasets[p.dataset] = true; });
  });
  var datasetCount = Object.keys(datasets).length;

  // Calculate best speedup across benchmark runs (exclude tests)
  var bestSpeedup = 0;
  var bestSpeedupLabel = '';
  var speedupBaselines = {};
  DATA.forEach(function(r) {
    if (r.run_type !== 'cpu_baseline' && r.run_type !== 'gpu_benchmark') return;
    r.timing.forEach(function(t) {
      var p = parseCmd(t.command);
      if (p.threads === 1) {
        var key = p.dataset;
        if (!speedupBaselines[key] || t.time_s < speedupBaselines[key]) speedupBaselines[key] = t.time_s;
      }
    });
  });
  DATA.forEach(function(r) {
    if (r.run_type !== 'cpu_baseline' && r.run_type !== 'gpu_benchmark') return;
    r.timing.forEach(function(t) {
      var p = parseCmd(t.command);
      if (p.threads > 1) {
        var key = p.dataset;
        var base = speedupBaselines[key];
        if (base) {
          var sp = base / t.time_s;
          if (sp > bestSpeedup) { bestSpeedup = sp; bestSpeedupLabel = p.dataset.replace('.fa','').replace('.phy','') + ' @ ' + p.threads + 'T'; }
        }
      }
    });
  });

  // Find best and worst IPC across benchmark runs
  var worstIPC = Infinity, worstIPCLabel = '', bestIPC = 0, bestIPCLabel = '';
  DATA.forEach(function(r) {
    if (r.run_type !== 'cpu_baseline' && r.run_type !== 'gpu_benchmark') return;
    if (r.profile && r.profile.metrics && r.profile.metrics.IPC != null) {
      var ipc = r.profile.metrics.IPC;
      var ds = (r.profile.dataset || '?').replace('.fa','').replace('.phy','');
      var threads = r.profile.threads || 1;
      var label = ds + ' T' + threads;
      if (ipc < worstIPC) { worstIPC = ipc; worstIPCLabel = label; }
      if (ipc > bestIPC) { bestIPC = ipc; bestIPCLabel = label; }
    }
  });

  document.getElementById('statsGrid').innerHTML =
    '<div class="stat-card">' +
      '<div class="stat-icon" style="background:var(--accent-glow);color:var(--accent);">&#x25B6;</div>' +
      '<div class="label">Benchmark Runs</div>' +
      '<div class="value" style="color:var(--accent)">' + benchmarkCount + '</div>' +
      '<div class="change">' + datasetCount + ' dataset' + (datasetCount > 1 ? 's' : '') + ' | ' + (totalTests > 0 ? totalPass + '/' + totalTests + ' tests pass' : 'Baseline runs') + '</div>' +
    '</div>' +
    '<div class="stat-card">' +
      '<div class="stat-icon" style="background:rgba(234,179,8,0.1);color:var(--yellow);">&#x26A1;</div>' +
      '<div class="label">Best Speedup</div>' +
      '<div class="value" style="color:var(--yellow)">' + (bestSpeedup > 0 ? bestSpeedup.toFixed(2) + '\\u00d7' : 'N/A') + '</div>' +
      '<div class="change">' + (bestSpeedupLabel || 'Need multi-thread baselines') + '</div>' +
    '</div>' +
    '<div class="stat-card">' +
      '<div class="stat-icon" style="background:rgba(52,211,153,0.1);color:var(--green);">&#x1F4CA;</div>' +
      '<div class="label">Best IPC</div>' +
      '<div class="value" style="color:var(--green)">' + (bestIPC > 0 ? bestIPC.toFixed(3) : 'N/A') + '</div>' +
      '<div class="change">' + (bestIPCLabel || 'No perf data') + '</div>' +
    '</div>' +
    '<div class="stat-card">' +
      '<div class="stat-icon" style="background:rgba(248,113,113,0.1);color:var(--red);">&#x26A0;</div>' +
      '<div class="label">Worst IPC</div>' +
      '<div class="value" style="color:var(--red)">' + (worstIPC < Infinity ? worstIPC.toFixed(3) : 'N/A') + '</div>' +
      '<div class="change">' + (worstIPCLabel || 'No perf data') + '</div>' +
    '</div>';

  renderLeaderboard(run);
  renderHotspotChart();
  renderMicroarchChart();
  renderScalingChart();
  renderOverviewConfig(run);
}

function renderOverviewConfig(run) {
  var card = document.getElementById('overviewConfigCard');
  var content = document.getElementById('overviewConfigContent');
  if (!card || !content) return;

  var env = run.env || {};
  var mf = run.modelfinder || {};

  // Parse the first (or main) command to extract dataset/model/threads
  var mainCmd = run.timing.length > 0 ? run.timing[run.timing.length - 1] : null;
  var cmdInfo = mainCmd ? parseCmd(mainCmd.command) : null;

  // Try to find a matching deep profile for richer data
  var matchedProfile = null;
  if (cmdInfo) {
    PROFILES.forEach(function(p) {
      var pds = (p.dataset || '').split('/').pop();
      if (pds === cmdInfo.dataset || (pds && cmdInfo.dataset && pds.indexOf(cmdInfo.dataset.replace('.fa','').replace('.phy','')) >= 0)) {
        matchedProfile = p;
      }
    });
  }

  var aln = matchedProfile ? (matchedProfile.alignment || {}) : {};
  var sys = matchedProfile ? (matchedProfile.system || {}) : {};
  var hasProfile = matchedProfile != null;

  // Build display values — prefer profile data, fall back to run data
  var dataset = cmdInfo ? cmdInfo.dataset : 'N/A';
  var taxa = hasProfile && aln.taxa ? aln.taxa : (cmdInfo ? cmdInfo.taxa : '?');
  var sites = hasProfile && aln.sites ? aln.sites : (cmdInfo ? cmdInfo.sites : '?');
  var model = mf.model_selected || (hasProfile && aln.substitution_model ? aln.substitution_model : (cmdInfo ? cmdInfo.model : 'AUTO'));
  var threads = cmdInfo ? cmdInfo.threads : (env.cores || '?');
  var wallTime = run.summary.total_time;

  // System info
  var cpu = sys.cpu || env.cpu || 'N/A';
  var cores = sys.cores || env.cores || '?';
  var gcc = sys.gcc || env.gcc || 'N/A';
  var rocm = sys.rocm || env.rocm || 'N/A';
  var hostname = sys.hostname || env.hostname || 'N/A';

  function fmtFileSize(bytes) {
    if (!bytes) return 'N/A';
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
  }

  card.style.display = '';

  var html =
    '<div class="config-grid">' +
      '<div class="config-section">' +
        '<div class="section-label" style="margin-top:0;">Alignment</div>' +
        '<div class="config-items">' +
          '<div class="config-item"><span class="ci-label">Dataset</span><span class="ci-value" style="color:var(--accent)">' + escHtml(dataset) + '</span></div>' +
          '<div class="config-item"><span class="ci-label">Taxa</span><span class="ci-value" style="color:var(--accent2);font-weight:700;font-size:1.1rem;">' + taxa + '</span></div>' +
          '<div class="config-item"><span class="ci-label">Sites</span><span class="ci-value" style="color:var(--accent2);font-weight:700;font-size:1.1rem;">' + (typeof sites === 'number' ? sites.toLocaleString() : sites) + '</span></div>' +
          (hasProfile ? '<div class="config-item"><span class="ci-label">Data Type</span><span class="ci-value">' + escHtml(aln.data_type || 'N/A') + '</span></div>' : '') +
          (hasProfile ? '<div class="config-item"><span class="ci-label">File Size</span><span class="ci-value">' + fmtFileSize(aln.file_size_bytes) + '</span></div>' : '') +
          (hasProfile && aln.site_patterns ? '<div class="config-item"><span class="ci-label">Site Patterns</span><span class="ci-value">' + aln.site_patterns.toLocaleString() + '</span></div>' : '') +
          (hasProfile && aln.informative_sites ? '<div class="config-item"><span class="ci-label">Informative Sites</span><span class="ci-value">' + aln.informative_sites.toLocaleString() + '</span></div>' : '') +
          (hasProfile && aln.constant_sites != null ? '<div class="config-item"><span class="ci-label">Constant Sites</span><span class="ci-value">' + aln.constant_sites + ' (' + (aln.constant_sites_pct || 0) + '%%)</span></div>' : '') +
          (hasProfile && aln.free_parameters ? '<div class="config-item"><span class="ci-label">Free Parameters</span><span class="ci-value">' + aln.free_parameters + '</span></div>' : '') +
        '</div>' +
      '</div>' +
      '<div class="config-section">' +
        '<div class="section-label" style="margin-top:0;">Model &amp; Results</div>' +
        '<div class="config-items">' +
          '<div class="config-item"><span class="ci-label">Model</span><span class="ci-value" style="color:var(--accent);font-weight:600;">' + escHtml(model) + '</span></div>' +
          (hasProfile && aln.rate_model ? '<div class="config-item"><span class="ci-label">Rate Heterogeneity</span><span class="ci-value">' + escHtml(aln.rate_model) + '</span></div>' : '') +
          (hasProfile && aln.gamma_alpha ? '<div class="config-item"><span class="ci-label">Gamma \\u03b1</span><span class="ci-value">' + aln.gamma_alpha + '</span></div>' : '') +
          '<div class="config-item"><span class="ci-label">Threads</span><span class="ci-value">' + threads + '</span></div>' +
          (hasProfile && aln.log_likelihood != null ? '<div class="config-item"><span class="ci-label">Log-Likelihood</span><span class="ci-value" style="color:var(--green);font-family:monospace;">' + aln.log_likelihood.toLocaleString() + '</span></div>' : '') +
          (hasProfile && aln.bic != null ? '<div class="config-item"><span class="ci-label">BIC</span><span class="ci-value" style="font-family:monospace;">' + aln.bic.toLocaleString() + '</span></div>' : '') +
          (hasProfile && aln.tree_length ? '<div class="config-item"><span class="ci-label">Tree Length</span><span class="ci-value">' + aln.tree_length + '</span></div>' : '') +
          '<div class="config-item"><span class="ci-label">Wall Time</span><span class="ci-value">' + fmtTime(wallTime) + '</span></div>' +
        '</div>' +
      '</div>' +
      '<div class="config-section">' +
        '<div class="section-label" style="margin-top:0;">System</div>' +
        '<div class="config-items">' +
          '<div class="config-item"><span class="ci-label">CPU</span><span class="ci-value">' + escHtml(cpu.replace(' 64-Core Processor', '')) + '</span></div>' +
          '<div class="config-item"><span class="ci-label">Cores</span><span class="ci-value">' + cores + (sys.threads_per_core ? ' (' + sys.threads_per_core + 'T/core)' : '') + '</span></div>' +
          (sys.mem_total_gb ? '<div class="config-item"><span class="ci-label">Memory</span><span class="ci-value">' + sys.mem_total_gb + ' GB</span></div>' : '') +
          (sys.l3_cache ? '<div class="config-item"><span class="ci-label">L3 Cache</span><span class="ci-value">' + escHtml(sys.l3_cache) + '</span></div>' : '') +
          '<div class="config-item"><span class="ci-label">GPU</span><span class="ci-value">' + escHtml(sys.gpu || env.gpu || 'N/A') + '</span></div>' +
          '<div class="config-item"><span class="ci-label">ROCm</span><span class="ci-value">' + escHtml(rocm) + '</span></div>' +
          '<div class="config-item"><span class="ci-label">GCC</span><span class="ci-value">' + escHtml(gcc) + '</span></div>' +
          (sys.numa_nodes ? '<div class="config-item"><span class="ci-label">NUMA Nodes</span><span class="ci-value">' + sys.numa_nodes + '</span></div>' : '') +
          '<div class="config-item"><span class="ci-label">Host</span><span class="ci-value">' + escHtml(hostname) + '</span></div>' +
        '</div>' +
      '</div>' +
    '</div>';

  // Command line
  var cmdLine = mainCmd ? mainCmd.command : '';
  if (cmdLine) {
    html += '<div class="config-cmd">' +
      '<span style="color:var(--text3);margin-right:8px;">$</span>' + escHtml(cmdLine) +
      '<button class="btn-sm" onclick="copySingleCmd(\\'' + escAttr(cmdLine) + '\\', this)">Copy</button>' +
    '</div>';
  }

  content.innerHTML = html;
  content.setAttribute('data-run-config', JSON.stringify({
    dataset: dataset, model: model, threads: threads, wallTime: wallTime,
    env: env, alignment: aln, system: sys, command: cmdLine,
    run_id: run.run_id, date: env.date
  }));
}

function copyOverviewConfig(btn) {
  var el = document.getElementById('overviewConfigContent');
  var data = JSON.parse(el.getAttribute('data-run-config') || '{}');
  var aln = data.alignment || {};
  var sys = data.system || {};
  var env = data.env || {};
  var lines = [
    '=== IQ-TREE Run Configuration ===',
    'Run ID: ' + (data.run_id || 'N/A'),
    'Date: ' + (data.date || 'N/A'),
    '',
    '--- Alignment ---',
    'Dataset: ' + (data.dataset || 'N/A'),
    'Taxa: ' + (aln.taxa || 'N/A'),
    'Sites: ' + (aln.sites || 'N/A'),
    aln.data_type ? 'Data Type: ' + aln.data_type : null,
    aln.site_patterns ? 'Site Patterns: ' + aln.site_patterns : null,
    aln.informative_sites ? 'Informative Sites: ' + aln.informative_sites : null,
    aln.constant_sites != null ? 'Constant Sites: ' + aln.constant_sites + ' (' + (aln.constant_sites_pct || 0) + '%%)' : null,
    '',
    '--- Model & Results ---',
    'Model: ' + (data.model || 'AUTO'),
    'Threads: ' + (data.threads || 'N/A'),
    aln.log_likelihood ? 'Log-Likelihood: ' + aln.log_likelihood : null,
    aln.bic ? 'BIC Score: ' + aln.bic : null,
    aln.tree_length ? 'Tree Length: ' + aln.tree_length : null,
    'Wall Time: ' + (data.wallTime != null ? data.wallTime + 's' : 'N/A'),
    '',
    '--- System ---',
    'CPU: ' + (sys.cpu || env.cpu || 'N/A'),
    'Cores: ' + (sys.cores || env.cores || 'N/A'),
    sys.mem_total_gb ? 'Memory: ' + sys.mem_total_gb + ' GB' : null,
    'GPU: ' + (sys.gpu || env.gpu || 'N/A'),
    'ROCm: ' + (sys.rocm || env.rocm || 'N/A'),
    'GCC: ' + (sys.gcc || env.gcc || 'N/A'),
    '',
    '--- Command ---',
    data.command || ''
  ].filter(function(l) { return l !== null; });
  navigator.clipboard.writeText(lines.join('\\n'));
  btn.textContent = 'Copied!';
  setTimeout(function() {
    btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy Config';
  }, 1500);
}

// ============ IQ-TREE Performance Metrics ============

// Known dataset sizes — extended as more benchmarks are added
var DATASET_INFO = {
  'turtle.fa': { taxa: 16, sites: 1998, sizeMB: 0.04 },
  'medium_dna.fa': { taxa: 50, sites: 5000, sizeMB: 0.5 },
  'medium_dna.phy': { taxa: 50, sites: 5000, sizeMB: 0.5 },
  'large_dna.phy': { taxa: 100, sites: 10000, sizeMB: 2.0 },
  'large_modelfinder.fa': { taxa: 100, sites: 10000, sizeMB: 2.0 },
  'xlarge_dna.fa': { taxa: 200, sites: 50000, sizeMB: 19.1 },
  'stress_dna.phy': { taxa: 200, sites: 20000, sizeMB: 7.6 }
};

// Model block sizes: nstates x ncat
var MODEL_BLOCKS = {
  'JC': 4, 'HKY': 4, 'GTR': 4, 'GTR+G4': 16, 'GTR+G8': 32,
  'HKY+G4': 16, 'JC+G4': 16, 'GTR+I+G4': 16, 'GTR+F+I+G4': 16,
  'WAG': 20, 'LG': 20, 'WAG+G4': 80, 'LG+G4': 80,
  'default': 16
};

function parseCmd(cmd) {
  var ds = 'unknown'; var threads = 1; var model = 'AUTO'; var gpu = false;
  var sMatch = cmd.match(/-s\\s+(\\S+)/);
  if (sMatch) ds = sMatch[1].replace(/.*\\//, '');
  var tMatch = cmd.match(/-T\\s+(\\d+)/);
  if (tMatch) threads = parseInt(tMatch[1]);
  var mMatch = cmd.match(/-m\\s+(\\S+)/);
  if (mMatch) model = mMatch[1];
  if (cmd.indexOf('--gpu') !== -1 || cmd.indexOf('-gpu') !== -1) gpu = true;
  var info = DATASET_INFO[ds] || { taxa: '?', sites: '?', sizeMB: null };
  var blockKey = model.replace(/\\+F/g, '').replace(/\\+I/g, '');
  var block = MODEL_BLOCKS[blockKey] || MODEL_BLOCKS['default'];
  return { dataset: ds, taxa: info.taxa, sites: info.sites, sizeMB: info.sizeMB, threads: threads, model: model, block: block, gpu: gpu };
}

var leaderboardSort = 'time';
var leaderboardDatasetFilter = 'all';

function sortLeaderboard(by) {
  leaderboardSort = by;
  document.getElementById('lbSortTime').className = 'btn' + (by === 'time' ? ' active' : '');
  document.getElementById('lbSortSpeedup').className = 'btn' + (by === 'speedup' ? ' active' : '');
  renderLeaderboard(DATA[currentRunIdx]);
}

function filterLeaderboardDataset(ds) {
  leaderboardDatasetFilter = ds;
  renderLeaderboard(DATA[currentRunIdx]);
}

function renderLeaderboard(run) {
  // Build entries from benchmark runs only (exclude test/integration runs)
  var entries = [];

  DATA.forEach(function(r) {
    // Skip non-benchmark runs (integration pipeline tests with turtle.fa)
    var isTest = r.run_type !== 'cpu_baseline' && r.run_type !== 'gpu_benchmark';
    if (isTest) return;

    r.timing.forEach(function(t) {
      var p = parseCmd(t.command);
      // Get log-likelihood from modelfinder description or verify data
      var loglik = null;
      if (r.modelfinder && r.modelfinder.log_likelihood) {
        loglik = r.modelfinder.log_likelihood;
      }
      r.verify.forEach(function(v) {
        if (t.command.indexOf(v.file.replace('.iqtree', '')) !== -1 && v.reported) loglik = v.reported;
      });
      // Try extracting from description field (e.g. "logLik=-123456.789")
      if (!loglik && r.description) {
        var llMatch = r.description.match(/logLik[=:]\\s*(-?[\\d.]+)/i);
        if (llMatch) loglik = parseFloat(llMatch[1]);
      }
      // Get the model selected from modelfinder if model was AUTO
      var displayModel = p.model;
      if (displayModel === 'AUTO' && r.modelfinder && r.modelfinder.model_selected) {
        displayModel = r.modelfinder.model_selected.replace(' chosen according to BIC', '');
      }

      // Get IPC from run profile if available
      var ipc = (r.profile && r.profile.metrics && r.profile.metrics.IPC) ? r.profile.metrics.IPC : null;

      entries.push({
        dataset: p.dataset, taxa: p.taxa, sites: p.sites, sizeMB: p.sizeMB,
        model: displayModel, block: p.block, threads: p.threads,
        gpu: p.gpu ? 'Yes' : '\\u2014',
        time: t.time_s, loglik: loglik, runId: r.run_id, cmd: t.command,
        label: r.label || r.run_id, ipc: ipc
      });
    });
  });

  // Calculate speedups — compare same dataset runs (1T baseline vs multi-T)
  var baselines = {};
  entries.forEach(function(e) {
    if (e.threads === 1) {
      var key = e.dataset;
      if (!baselines[key] || e.time < baselines[key]) baselines[key] = e.time;
    }
  });
  entries.forEach(function(e) {
    var key = e.dataset;
    var base = baselines[key];
    if (base) {
      e.speedup = (base / e.time).toFixed(2);
      e.baselineTime = base;
    } else {
      e.speedup = '1.00';
      e.baselineTime = e.time;
    }
  });

  // Find max speedup per dataset for bar scaling
  var maxSpeedups = {};
  entries.forEach(function(e) {
    var sp = parseFloat(e.speedup);
    if (!maxSpeedups[e.dataset] || sp > maxSpeedups[e.dataset]) maxSpeedups[e.dataset] = sp;
  });

  // Apply dataset filter
  var filtered = entries;
  if (leaderboardDatasetFilter !== 'all') {
    filtered = entries.filter(function(e) { return e.dataset === leaderboardDatasetFilter; });
  }

  // Sort
  if (leaderboardSort === 'speedup') {
    filtered.sort(function(a, b) { return parseFloat(b.speedup || 0) - parseFloat(a.speedup || 0); });
  } else {
    filtered.sort(function(a, b) { return a.time - b.time; });
  }

  // Build dataset filter buttons
  var dsSet = {};
  entries.forEach(function(e) { dsSet[e.dataset] = true; });
  var dsKeys = Object.keys(dsSet).sort();
  var filterHtml = '<button class="btn' + (leaderboardDatasetFilter === 'all' ? ' active' : '') + '" onclick="filterLeaderboardDataset(\\'all\\')">All</button>';
  dsKeys.forEach(function(ds) {
    var shortName = ds.replace('.fa', '').replace('.phy', '');
    var info = DATASET_INFO[ds];
    var sizeStr = info && info.sizeMB ? ' (' + info.sizeMB + ' MB)' : '';
    filterHtml += '<button class="btn' + (leaderboardDatasetFilter === ds ? ' active' : '') + '" onclick="filterLeaderboardDataset(\\'' + escAttr(ds) + '\\')">' + escHtml(shortName) + sizeStr + '</button>';
  });
  var filterEl = document.getElementById('lbDatasetFilters');
  if (filterEl) filterEl.innerHTML = filterHtml;

  // Update count
  var countEl = document.getElementById('lbCount');
  if (countEl) countEl.textContent = filtered.length + ' benchmark run' + (filtered.length !== 1 ? 's' : '');

  var tbody = document.querySelector('#leaderboardTable tbody');
  tbody.innerHTML = filtered.map(function(e, i) {
    var rankClass = i === 0 ? 'gold' : i === 1 ? 'silver' : i === 2 ? 'bronze' : '';
    var spVal = parseFloat(e.speedup) || 0;
    var spClass = spVal >= 3 ? 'speedup-good' : spVal >= 1.5 ? 'speedup-warn' : 'speedup-bad';
    var maxSp = maxSpeedups[e.dataset] || 1;
    var barPct = Math.min(100, (spVal / Math.max(maxSp, e.threads)) * 100);
    var barColor = spVal >= e.threads * 0.5 ? 'var(--green)' : spVal >= e.threads * 0.25 ? 'var(--yellow)' : 'var(--red)';
    if (e.threads === 1) barColor = 'var(--accent)';
    var sizeStr = e.sizeMB != null ? e.sizeMB.toFixed(1) : '\\u2014';
    var loglikStr = e.loglik != null ? (typeof e.loglik === 'number' ? e.loglik.toLocaleString() : e.loglik) : '\\u2014';
    var loglikStyle = e.loglik != null ? 'color:var(--green);font-family:monospace;' : 'color:var(--text3);';
    var ipcStr = e.ipc != null ? e.ipc.toFixed(3) : '\\u2014';
    var ipcColor = e.ipc != null ? (e.ipc >= 2 ? 'var(--green)' : e.ipc >= 1 ? 'var(--yellow)' : 'var(--red)') : 'var(--text3)';
    return '<tr title="' + escAttr(e.cmd) + '">' +
      '<td class="lb-rank ' + rankClass + '">' + (i + 1) + '</td>' +
      '<td class="lb-dataset">' + escHtml(e.dataset) + '<span style="color:var(--text3);font-size:0.68rem;margin-left:6px;">' + sizeStr + ' MB</span></td>' +
      '<td>' + e.taxa + '</td>' +
      '<td>' + (typeof e.sites === 'number' ? e.sites.toLocaleString() : e.sites) + '</td>' +
      '<td class="lb-model">' + escHtml(e.model) + '</td>' +
      '<td>' + e.block + '</td>' +
      '<td>' + e.threads + '</td>' +
      '<td>' + e.gpu + '</td>' +
      '<td style="font-weight:700;">' + fmtTime(e.time) + '</td>' +
      '<td><span class="speedup-badge ' + spClass + '">' + e.speedup + '\\u00d7</span></td>' +
      '<td><div class="eff-bar"><div class="fill" style="width:' + barPct.toFixed(0) + '%%;background:' + barColor + '"></div></div></td>' +
      '<td style="font-weight:600;color:' + ipcColor + ';">' + ipcStr + '</td>' +
      '<td style="' + loglikStyle + '">' + loglikStr + '</td>' +
    '</tr>';
  }).join('');
}

// ============ Chart Source Builders ============
function buildHotspotSources() {
  var sources = [];
  PROFILES.forEach(function(p, i) {
    var hs = (p.cpu || {}).hotspots || [];
    if (hs.length === 0) return;
    var dsName = (p.dataset || '?').split('/').pop().replace('.fa', '').replace('.phy', '') + ' T' + (p.threads || 1);
    sources.push({ key: 'p' + i, label: dsName, profile: p });
  });
  return sources;
}

function buildMicroarchSources() {
  var sources = [];
  PROFILES.forEach(function(p, i) {
    var d = (p.cpu || {}).derived || {};
    if (!d.IPC && !d['cache-miss-rate']) return;
    var dsName = (p.dataset || '?').split('/').pop().replace('.fa', '').replace('.phy', '') + ' T' + (p.threads || 1);
    sources.push({ key: 'p' + i, label: dsName, metrics: d, type: 'deep' });
  });
  DATA.forEach(function(r, i) {
    if (!r.profile || !r.profile.metrics || !r.profile.metrics.IPC) return;
    var m = r.profile.metrics;
    var rDs = (r.profile.dataset || '').replace('.fa', '').replace('.phy', '');
    var dsName = rDs + ' T' + (r.profile.threads || 1) + ' (perf)';
    sources.push({ key: 'd' + i, label: dsName, metrics: m, type: 'run' });
  });
  return sources;
}

function initChartSelectors() {
  buildHotspotSources().forEach(function(s) {
    if (hotspotSelection[s.key] === undefined) hotspotSelection[s.key] = true;
  });
  buildMicroarchSources().forEach(function(s) {
    if (microarchSelection[s.key] === undefined) microarchSelection[s.key] = true;
  });
}

function renderHotspotSelector() {
  var el = document.getElementById('hotspotCheckList');
  if (!el) return;
  var sources = buildHotspotSources();
  var searchEl = document.getElementById('hotspotSearch');
  var search = searchEl ? searchEl.value.toLowerCase() : '';
  var html = '';
  sources.forEach(function(s) {
    if (search && s.label.toLowerCase().indexOf(search) === -1) return;
    var sel = hotspotSelection[s.key] !== false;
    html += '<div class="cs-item' + (sel ? ' selected' : '') + '" onclick="toggleHotspot(\\'' + s.key + '\\')">' + escHtml(s.label) + '</div>';
  });
  if (!html) html = '<span class="cs-empty">No matching profiles</span>';
  el.innerHTML = html;
}

function toggleHotspot(key) {
  hotspotSelection[key] = !hotspotSelection[key];
  renderHotspotSelector();
  renderHotspotChart();
}

function hotspotSelectAll() {
  buildHotspotSources().forEach(function(s) { hotspotSelection[s.key] = true; });
  renderHotspotSelector();
  renderHotspotChart();
}

function hotspotSelectNone() {
  buildHotspotSources().forEach(function(s) { hotspotSelection[s.key] = false; });
  renderHotspotSelector();
  renderHotspotChart();
}

function renderMicroarchSelector() {
  var el = document.getElementById('microarchCheckList');
  if (!el) return;
  var sources = buildMicroarchSources();
  var searchEl = document.getElementById('microarchSearch');
  var search = searchEl ? searchEl.value.toLowerCase() : '';
  var html = '';
  sources.forEach(function(s) {
    if (search && s.label.toLowerCase().indexOf(search) === -1) return;
    var sel = microarchSelection[s.key] !== false;
    html += '<div class="cs-item' + (sel ? ' selected' : '') + '" onclick="toggleMicroarch(\\'' + s.key + '\\')">' + escHtml(s.label) + '</div>';
  });
  if (!html) html = '<span class="cs-empty">No matching runs</span>';
  el.innerHTML = html;
}

function toggleMicroarch(key) {
  microarchSelection[key] = !microarchSelection[key];
  renderMicroarchSelector();
  renderMicroarchChart();
}

function microarchSelectAll() {
  buildMicroarchSources().forEach(function(s) { microarchSelection[s.key] = true; });
  renderMicroarchSelector();
  renderMicroarchChart();
}

function microarchSelectNone() {
  buildMicroarchSources().forEach(function(s) { microarchSelection[s.key] = false; });
  renderMicroarchSelector();
  renderMicroarchChart();
}

// ============ Hotspot Chart ============
function renderHotspotChart() {
  if (charts.hotspot) charts.hotspot.destroy();
  var ctx = document.getElementById('hotspotChart');
  if (!ctx) return;

  var kernelNames = ['DervSIMD', 'PartialLH', 'BufferSIMD', 'FromBuffer', 'libgomp', 'parsimony', 'Other'];
  var kernelColors = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#6b7280', '#06b6d4', '#3b82f6'];
  var labels = [];
  var datasets = kernelNames.map(function(name, idx) {
    return { label: name, backgroundColor: kernelColors[idx], data: [] };
  });

  // Use only sources selected in the selector
  var sources = buildHotspotSources().filter(function(s) { return hotspotSelection[s.key] !== false; });

  if (sources.length === 0) {
    ctx.parentElement.innerHTML = '<div class="no-data-msg">No hotspot data. Run deep profiling on Setonix.</div>';
    return;
  }

  sources.forEach(function(s) {
    var p = s.profile;
    var hs = (p.cpu || {}).hotspots || [];
    labels.push(s.label);
    var buckets = { DervSIMD: 0, PartialLH: 0, BufferSIMD: 0, FromBuffer: 0, libgomp: 0, parsimony: 0, Other: 0 };
    hs.forEach(function(h) {
      var fn = h['function'] || '';
      if (fn.indexOf('DervSIMD') !== -1) buckets.DervSIMD += h.percent;
      else if (fn.indexOf('PartialLikelihood') !== -1 || fn.indexOf('PartialLH') !== -1) buckets.PartialLH += h.percent;
      else if (fn.indexOf('BufferSIMD') !== -1 && fn.indexOf('FromBuffer') === -1) buckets.BufferSIMD += h.percent;
      else if (fn.indexOf('FromBuffer') !== -1) buckets.FromBuffer += h.percent;
      else if (fn.indexOf('libgomp') !== -1 || h.module === 'libgomp.so.1.0.0') buckets.libgomp += h.percent;
      else if (fn.indexOf('arsimony') !== -1) buckets.parsimony += h.percent;
      else buckets.Other += h.percent;
    });
    var total = 0; for (var k in buckets) total += buckets[k];
    if (total < 100) buckets.Other += (100 - total);
    datasets[0].data.push(buckets.DervSIMD);
    datasets[1].data.push(buckets.PartialLH);
    datasets[2].data.push(buckets.BufferSIMD);
    datasets[3].data.push(buckets.FromBuffer);
    datasets[4].data.push(buckets.libgomp);
    datasets[5].data.push(buckets.parsimony);
    datasets[6].data.push(buckets.Other);
  });

  charts.hotspot = new Chart(ctx.getContext('2d'), {
    type: 'bar',
    data: { labels: labels, datasets: datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: 'bottom', labels: { color: '#94a3b8', boxWidth: 12, padding: 10, font: { size: 10 } } },
        tooltip: { backgroundColor: '#1a2332', borderColor: '#2a3444', borderWidth: 1, cornerRadius: 8,
          callbacks: { label: function(c) { return c.dataset.label + ': ' + c.raw.toFixed(1) + '%%'; } }
        }
      },
      scales: {
        x: { stacked: true, grid: { display: false }, ticks: { color: '#94a3b8', font: { size: 11 } } },
        y: { stacked: true, max: 100, grid: { color: '#1e293b' }, ticks: { color: '#64748b', callback: function(v) { return v + '%%'; } },
          title: { display: true, text: '%% of CPU Time', color: '#64748b' } }
      }
    }
  });
}

function renderMicroarchChart() {
  if (charts.microarch) charts.microarch.destroy();
  var ctx = document.getElementById('microarchChart');
  if (!ctx) return;

  var radarLabels = ['IPC', 'Cache Hit %%', 'Branch Acc %%', 'FE Efficiency %%', 'L1D Hit %%', 'dTLB Hit %%'];
  var radarDatasets = [];
  var colors = ['#3b82f6', '#8b5cf6', '#22c55e', '#f97316', '#ef4444', '#06b6d4', '#eab308', '#ec4899', '#14b8a6', '#a78bfa'];

  // Use only sources selected in the selector
  var sources = buildMicroarchSources().filter(function(s) { return microarchSelection[s.key] !== false; });

  if (sources.length === 0) {
    ctx.parentElement.innerHTML = '<div class="no-data-msg">No profile data. Select runs above or run benchmarks on Setonix.</div>';
    return;
  }

  sources.forEach(function(src, i) {
    var d = src.metrics;
    var ipcNorm = Math.min((d.IPC || 0) / 4.0 * 100, 100);
    var cacheHit = 100 - (d['cache-miss-rate'] || 0);
    var branchAcc = 100 - (d['branch-miss-rate'] || 0);
    var feEff = 100 - (d['frontend-stall-rate'] || 0);
    var l1dHit = 100 - (d['L1-dcache-miss-rate'] || 0);
    var dtlbHit = 100 - (d['dTLB-miss-rate'] || 0);
    radarDatasets.push({
      label: src.label,
      data: [ipcNorm, cacheHit, branchAcc, feEff, l1dHit, dtlbHit],
      borderColor: colors[i %% colors.length],
      backgroundColor: colors[i %% colors.length] + '15',
      pointBackgroundColor: colors[i %% colors.length],
      pointRadius: 3, borderWidth: 2
    });
  });

  charts.microarch = new Chart(ctx.getContext('2d'), {
    type: 'radar',
    data: { labels: radarLabels, datasets: radarDatasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'bottom', labels: { color: '#94a3b8', boxWidth: 12 } } },
      scales: {
        r: {
          min: 0, max: 100,
          grid: { color: '#1e293b' }, angleLines: { color: '#1e293b' },
          pointLabels: { color: '#94a3b8', font: { size: 10 } },
          ticks: { color: '#64748b', backdropColor: 'transparent', stepSize: 25,
            callback: function(v) { return v + '%%'; } }
        }
      }
    }
  });
}

function renderScalingChart() {
  if (charts.scaling) charts.scaling.destroy();
  var ctx = document.getElementById('scalingChart');
  if (!ctx) return;

  // Build thread-scaling data from commands across runs
  // Group by dataset: find same dataset at different thread counts
  var byDataset = {};
  DATA.forEach(function(r) {
    r.timing.forEach(function(t) {
      var p = parseCmd(t.command);
      if (p.taxa === '?') return;
      var key = p.dataset;
      if (!byDataset[key]) byDataset[key] = {};
      var tKey = p.threads;
      if (!byDataset[key][tKey] || t.time_s < byDataset[key][tKey]) {
        byDataset[key][tKey] = t.time_s;
      }
    });
  });

  // Also use profile data
  PROFILES.forEach(function(pr) {
    var ds = (pr.dataset || '').split('/').pop();
    var threads = pr.threads || 1;
    var taskClock = ((pr.cpu || {}).counters || {})['task-clock'];
    if (ds && taskClock) {
      var wallEst = taskClock / 1000 / threads;
      if (!byDataset[ds]) byDataset[ds] = {};
      if (!byDataset[ds][threads] || wallEst < byDataset[ds][threads]) {
        byDataset[ds][threads] = wallEst;
      }
    }
  });

  var dsKeys = Object.keys(byDataset);
  if (dsKeys.length === 0) {
    ctx.parentElement.innerHTML = '<div class="no-data-msg">Need multi-thread runs of the same dataset for scaling analysis.</div>';
    return;
  }

  var colors = ['#3b82f6', '#8b5cf6', '#22c55e', '#f97316', '#ef4444'];
  var allThreads = [];
  dsKeys.forEach(function(k) {
    Object.keys(byDataset[k]).forEach(function(t) {
      t = parseInt(t);
      if (allThreads.indexOf(t) === -1) allThreads.push(t);
    });
  });
  allThreads.sort(function(a, b) { return a - b; });

  var datasets = [];
  dsKeys.forEach(function(ds, i) {
    var base = byDataset[ds][1] || byDataset[ds][allThreads[0]];
    var data = allThreads.map(function(t) {
      if (!byDataset[ds][t] || !base) return null;
      return parseFloat((base / byDataset[ds][t]).toFixed(2));
    });
    datasets.push({
      label: ds.replace('.fa', '').replace('.phy', ''),
      data: data, borderColor: colors[i %% colors.length],
      backgroundColor: colors[i %% colors.length] + '22',
      fill: false, tension: 0.2, pointRadius: 5,
      pointBackgroundColor: colors[i %% colors.length],
      pointBorderColor: '#1a2332', pointBorderWidth: 2
    });
  });

  // Add ideal line
  var maxT = allThreads[allThreads.length - 1];
  datasets.push({
    label: 'Ideal',
    data: allThreads.map(function(t) { return t; }),
    borderColor: '#374151', borderDash: [5, 5],
    fill: false, pointRadius: 0, borderWidth: 1
  });

  charts.scaling = new Chart(ctx.getContext('2d'), {
    type: 'line',
    data: { labels: allThreads.map(function(t) { return t + 'T'; }), datasets: datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: 'bottom', labels: { color: '#94a3b8', boxWidth: 12 } },
        tooltip: { backgroundColor: '#1a2332', borderColor: '#2a3444', borderWidth: 1, cornerRadius: 8,
          callbacks: { label: function(c) { return c.dataset.label + ': ' + c.raw + '\\u00d7 speedup'; } }
        }
      },
      scales: {
        x: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' },
          title: { display: true, text: 'Threads', color: '#64748b' } },
        y: { grid: { color: '#1e293b' }, ticks: { color: '#64748b' }, beginAtZero: true,
          title: { display: true, text: 'Speedup (\\u00d7)', color: '#64748b' } }
      }
    }
  });
}

function renderQuickCmds(run) {
  var el = document.getElementById('quickCmds');
  el.innerHTML = run.timing.map(function(t, i) {
    return '<div class="cmd-block">' +
      '<span class="cmd-num">' + (i + 1) + '</span>' +
      '<span class="cmd-text">' + escHtml(t.command) + '</span>' +
      '<span class="cmd-time">' + t.time_s.toFixed(3) + 's</span>' +
      '<button class="cmd-copy" onclick="copySingleCmd(\\'' + escAttr(t.command) + '\\', this)">Copy</button>' +
    '</div>';
  }).join('');
}

// ============ All Runs List ============
function renderRunsList() {
  renderAllRunsStats();
  filterRuns();
}

function renderAllRunsStats() {
  var total = DATA.length;
  var runsWithTests = DATA.filter(function(r) { return r.summary.pass + r.summary.fail > 0; }).length;
  var baselines = DATA.filter(function(r) { return r.run_type === 'cpu_baseline'; }).length;
  var failing = DATA.filter(function(r) { return r.summary.fail > 0; }).length;
  var bestTime = Math.min.apply(null, DATA.map(function(r) { return r.summary.total_time; }));
  var avgTime = fmtTime(DATA.reduce(function(a, r) { return a + r.summary.total_time; }, 0) / total);
  var ipcs = DATA.map(function(r) { return (r.profile && r.profile.metrics && r.profile.metrics.IPC) || 0; });
  var bestIPC = Math.max.apply(null, ipcs);
  var totalCmds = DATA.reduce(function(a, r) { return a + r.timing.length; }, 0);

  document.getElementById('allRunsStats').innerHTML =
    '<div class="stat-card">' +
      '<div class="label">Total Runs</div>' +
      '<div class="value">' + total + '</div>' +
      '<div class="change">' + totalCmds + ' total commands</div>' +
    '</div>' +
    '<div class="stat-card">' +
      '<div class="label">Run Types</div>' +
      '<div class="value" style="color:var(--green)">' + baselines + ' / ' + runsWithTests + '</div>' +
      '<div class="change">' + baselines + ' baselines, ' + runsWithTests + ' with tests' + (failing > 0 ? ', ' + failing + ' failing' : '') + '</div>' +
    '</div>' +
    '<div class="stat-card">' +
      '<div class="label">Fastest Run</div>' +
      '<div class="value" style="color:var(--accent)">' + fmtTime(bestTime) + '</div>' +
      '<div class="change">Avg: ' + avgTime + '</div>' +
    '</div>' +
    '<div class="stat-card">' +
      '<div class="label">Peak IPC</div>' +
      '<div class="value" style="color:var(--accent2)">' + (bestIPC > 0 ? bestIPC.toFixed(3) : 'N/A') + '</div>' +
      '<div class="change">Instructions per cycle</div>' +
    '</div>' +
    '<div class="stat-card">' +
      '<div class="label">Hosts</div>' +
      '<div class="value">' + uniqueHosts().length + '</div>' +
      '<div class="change">' + (uniqueHosts().join(', ') || 'N/A') + '</div>' +
    '</div>';
}

function uniqueHosts() {
  var seen = {};
  var result = [];
  DATA.forEach(function(r) {
    var h = r.env.hostname;
    if (h && !seen[h]) { seen[h] = true; result.push(h); }
  });
  return result;
}

function filterRuns() {
  var query = (document.getElementById('runSearch') ? document.getElementById('runSearch').value : '').toLowerCase();
  var status = document.getElementById('runStatusFilter') ? document.getElementById('runStatusFilter').value : 'all';
  var sort = document.getElementById('runSortBy') ? document.getElementById('runSortBy').value : 'date-desc';

  var runs = DATA.map(function(r, i) { return { run: r, idx: i }; });

  if (query) {
    runs = runs.filter(function(item) {
      var r = item.run;
      var haystack = [
        r.run_id, r.label || '', r.description || '', r.run_type || '',
        r.env.hostname || '', r.env.date || '', r.env.cpu || '', r.env.gcc || '', r.env.rocm || ''
      ].concat(r.timing.map(function(t) { return t.command; })).join(' ').toLowerCase();
      return haystack.indexOf(query) !== -1;
    });
  }

  if (status === 'pass') runs = runs.filter(function(item) { return item.run.summary.all_pass; });
  if (status === 'fail') runs = runs.filter(function(item) { return !item.run.summary.all_pass; });

  runs.sort(function(a, b) {
    switch(sort) {
      case 'date-asc': return (a.run.env.date || '').localeCompare(b.run.env.date || '');
      case 'time-asc': return a.run.summary.total_time - b.run.summary.total_time;
      case 'time-desc': return b.run.summary.total_time - a.run.summary.total_time;
      case 'ipc-desc':
        var aIPC = (a.run.profile && a.run.profile.metrics && a.run.profile.metrics.IPC) || 0;
        var bIPC = (b.run.profile && b.run.profile.metrics && b.run.profile.metrics.IPC) || 0;
        return bIPC - aIPC;
      default: return (b.run.env.date || '').localeCompare(a.run.env.date || '');
    }
  });

  document.getElementById('runCount').textContent = 'Showing ' + runs.length + ' of ' + DATA.length + ' runs';

  var listEl = document.getElementById('runsList');
  listEl.innerHTML = runs.map(function(item, rank) {
    var r = item.run;
    var s = r.summary;
    var m = (r.profile && r.profile.metrics) || {};
    var rankClass = rank === 0 ? 'gold' : rank === 1 ? 'silver' : rank === 2 ? 'bronze' : '';
    var isExpanded = expandedRunId === r.run_id;
    var cpuShort = (r.env.cpu || '').replace(' 64-Core Processor', '');
    var runLabel = r.label || r.run_id;
    var hasTests = s.pass + s.fail > 0;
    var testBadge = hasTests
      ? '<span class="badge ' + (s.all_pass ? 'badge-pass' : 'badge-fail') + '">' + s.pass + '/' + (s.pass + s.fail) + '</span>'
      : '<span class="badge" style="background:var(--bg-tertiary);color:var(--text3);">Baseline</span>';

    return '<div class="run-row ' + (isExpanded ? 'active' : '') + '" id="run-' + r.run_id + '">' +
      '<div class="run-row-summary" onclick="toggleRunDetail(\\'' + r.run_id + '\\', ' + item.idx + ')">' +
        '<div class="rank ' + rankClass + '">#' + (rank + 1) + '</div>' +
        '<div class="run-info">' +
          '<div class="run-id">' + escHtml(runLabel) + '</div>' +
          '<div class="run-meta">' + (r.env.date || 'N/A') + ' &middot; ' + (r.env.hostname || 'N/A') + ' &middot; ' + (r.env.cores || '?') + ' cores &middot; ' + cpuShort + '</div>' +
          (r.description ? '<div class="run-meta" style="color:var(--text3);font-size:0.7rem;margin-top:2px;">' + escHtml(r.description.substring(0, 80)) + '</div>' : '') +
        '</div>' +
        '<div class="run-time">' + fmtTime(s.total_time) + '</div>' +
        '<div class="run-tests">' + testBadge + '</div>' +
        '<div class="run-ipc">' + (m.IPC || '\\u2014') + '</div>' +
        '<div class="run-cores">' + (r.env.cores || '?') + 'C</div>' +
        '<button class="run-detail-btn" onclick="event.stopPropagation(); toggleRunDetail(\\'' + r.run_id + '\\', ' + item.idx + ')">' + (isExpanded ? 'Hide' : 'Details') + '</button>' +
      '</div>' +
      '<div class="run-detail ' + (isExpanded ? 'open' : '') + '" id="detail-' + r.run_id + '">' +
        (isExpanded ? buildRunDetail(r) : '') +
      '</div>' +
    '</div>';
  }).join('');
}

function switchRunsTab(tab, btn) {
  document.querySelectorAll('#runsTabBar .tab-btn').forEach(function(b) { b.classList.remove('active'); });
  btn.classList.add('active');
  document.getElementById('runsTabContent').style.display = tab === 'runs' ? '' : 'none';
  document.getElementById('profilesTabContent').style.display = tab === 'profiles' ? '' : 'none';
}

function renderProfilesList() {
  var statsEl = document.getElementById('profilesStats');
  var listEl = document.getElementById('profilesList');
  if (!statsEl || !listEl) return;

  if (PROFILES.length === 0) {
    statsEl.innerHTML = '';
    listEl.innerHTML = '<div class="no-data-msg">No deep profiling sessions yet. Run <code>./start.sh deepprofile</code> on Setonix.</div>';
    return;
  }

  var bestIPC = 0;
  var totalHotspots = 0;
  var avgFEStall = 0;
  var gpuProfiles = 0;

  PROFILES.forEach(function(p) {
    var d = (p.cpu || {}).derived || {};
    var ipc = d.IPC || 0;
    if (ipc > bestIPC) bestIPC = ipc;
    totalHotspots += ((p.cpu || {}).hotspots || []).length;
    avgFEStall += d['frontend-stall-rate'] || 0;
    var hw = (p.gpu || {}).hardware || {};
    if (hw.temperature_c != null) gpuProfiles++;
  });
  avgFEStall = PROFILES.length > 0 ? (avgFEStall / PROFILES.length).toFixed(2) : 0;

  statsEl.innerHTML =
    '<div class="stat-card"><div class="label">Total Profiles</div><div class="value">' + PROFILES.length + '</div><div class="change">' + gpuProfiles + ' with GPU data</div></div>' +
    '<div class="stat-card"><div class="label">Best IPC</div><div class="value" style="color:var(--accent2)">' + (bestIPC > 0 ? bestIPC.toFixed(3) : 'N/A') + '</div><div class="change">Instructions per cycle</div></div>' +
    '<div class="stat-card"><div class="label">Avg FE Stalls</div><div class="value" style="color:' + (avgFEStall > 10 ? 'var(--red)' : 'var(--green)') + '">' + avgFEStall + '%%</div><div class="change">Frontend stall rate</div></div>' +
    '<div class="stat-card"><div class="label">Hotspot Funcs</div><div class="value">' + totalHotspots + '</div><div class="change">Across all profiles</div></div>' +
    '<div class="stat-card"><div class="label">Datasets</div><div class="value">' + uniqueDatasets().length + '</div><div class="change">' + (uniqueDatasets().join(', ').substring(0, 40) || 'N/A') + '</div></div>';

  listEl.innerHTML = PROFILES.map(function(p, i) {
    var d = (p.cpu || {}).derived || {};
    var hotspots = (p.cpu || {}).hotspots || [];
    var topFunc = hotspots.length > 0 ? hotspots[0] : null;
    var hw = (p.gpu || {}).hardware || {};
    var gpuInfo = hw.temperature_c != null ? hw.temperature_c + '\\u00b0C' : 'No GPU';
    var dateShort = (p.date || '').substring(0, 16).replace('T', ' ');
    var sys = p.system || {};

    return '<div class="profile-row" onclick="showPage(\\'profiling\\'); switchProfile(' + i + ');">' +
      '<div class="rank">#' + (i + 1) + '</div>' +
      '<div class="prof-info">' +
        '<div class="prof-id">' + p.profile_id + '</div>' +
        '<div class="prof-meta">' + dateShort + ' \\u00b7 ' + escHtml(sys.hostname || 'N/A') + ' \\u00b7 T' + (p.threads || '?') + ' \\u00b7 ' + escHtml(p.model || 'AUTO') + '</div>' +
      '</div>' +
      '<div class="prof-dataset">' + escHtml((p.dataset || 'N/A').split('/').pop()) + '</div>' +
      '<div class="prof-ipc">' + (d.IPC || '\\u2014') + '</div>' +
      '<div class="prof-hotspot" title="' + (topFunc ? escHtml(topFunc['function']) : '') + '">' + (topFunc ? topFunc.percent.toFixed(1) + '%% ' + escHtml(topFunc['function'].substring(0, 35)) : '\\u2014') + '</div>' +
      '<div class="prof-gpu">' + gpuInfo + '</div>' +
    '</div>';
  }).join('');
}

function uniqueDatasets() {
  var seen = {};
  var result = [];
  PROFILES.forEach(function(p) {
    var d = (p.dataset || '').split('/').pop();
    if (d && !seen[d]) { seen[d] = true; result.push(d); }
  });
  return result;
}

function toggleRunDetail(runId, runIdx) {
  if (expandedRunId === runId) {
    expandedRunId = null;
  } else {
    expandedRunId = runId;
  }
  filterRuns();
}

function buildRunDetail(run) {
  var s = run.summary;
  var m = (run.profile && run.profile.metrics) || {};
  var env = run.env || {};

  var cmdsHtml = run.timing.map(function(t, i) {
    return '<div class="cmd-block">' +
      '<span class="cmd-num">' + (i + 1) + '</span>' +
      '<span class="cmd-text">' + escHtml(t.command) + '</span>' +
      '<span class="cmd-time">' + t.time_s.toFixed(3) + 's</span>' +
      '<button class="cmd-copy" onclick="event.stopPropagation(); copySingleCmd(\\'' + escAttr(t.command) + '\\', this)">Copy</button>' +
    '</div>';
  }).join('');

  var verifyHtml;
  if (run.verify.length > 0) {
    verifyHtml = '<table class="verify-mini">' +
      '<thead><tr><th>Status</th><th>File</th><th>Expected</th><th>Reported</th><th>Diff</th></tr></thead>' +
      '<tbody>' + run.verify.map(function(v) {
        return '<tr>' +
          '<td>' + (v.status === 'pass' ? '\\u2705' : '\\u274c') + '</td>' +
          '<td style="font-family:monospace;font-size:0.8rem">' + v.file + '</td>' +
          '<td>' + v.expected + '</td>' +
          '<td>' + v.reported + '</td>' +
          '<td style="color:' + (v.diff < 0.01 ? 'var(--green)' : 'var(--yellow)') + '">' + v.diff + '</td>' +
        '</tr>';
      }).join('') + '</tbody></table>';
  } else {
    verifyHtml = '<p style="color:var(--text3)">No verification data</p>';
  }

  var envKeys = Object.keys(env);
  var envHtml = envKeys.map(function(k) {
    return '<div class="detail-kv">' +
      '<div class="dk-label">' + k + '</div>' +
      '<div class="dk-value">' + env[k] + '</div>' +
    '</div>';
  }).join('');

  var profileHtml;
  if (m.IPC) {
    profileHtml = '<div class="detail-grid">' +
      '<div class="detail-kv"><div class="dk-label">IPC</div><div class="dk-value" style="color:var(--accent)">' + m.IPC + '</div></div>' +
      '<div class="detail-kv"><div class="dk-label">Cache Miss Rate</div><div class="dk-value" style="color:var(--yellow)">' + (m['cache-miss-rate'] || 'N/A') + '%%</div></div>' +
      '<div class="detail-kv"><div class="dk-label">L1 D-Cache Miss</div><div class="dk-value" style="color:#f97316">' + (m['L1-dcache-miss-rate'] || 'N/A') + '%%</div></div>' +
      '<div class="detail-kv"><div class="dk-label">Branch Mispredict</div><div class="dk-value" style="color:var(--green)">' + (m['branch-miss-rate'] || 'N/A') + '%%</div></div>' +
      '<div class="detail-kv"><div class="dk-label">Instructions</div><div class="dk-value" style="color:var(--accent2)">' + (m.instructions ? (m.instructions / 1e9).toFixed(1) + 'B' : 'N/A') + '</div></div>' +
      '<div class="detail-kv"><div class="dk-label">Cycles</div><div class="dk-value">' + (m.cycles ? (m.cycles / 1e9).toFixed(2) + 'B' : 'N/A') + '</div></div>' +
    '</div>';
  } else {
    profileHtml = '<p style="color:var(--text3)">No profiling data</p>';
  }

  return '<div class="detail-section">' +
      '<h3>Commands (' + run.timing.length + ')</h3>' +
      '<div style="margin-bottom:8px">' +
        '<button class="copy-all-btn" onclick="event.stopPropagation(); copyRunCmds(\\'' + run.run_id + '\\', this)">' +
          '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>' +
          'Copy All as Script' +
        '</button>' +
      '</div>' +
      cmdsHtml +
    '</div>' +
    '<div class="detail-section">' +
      '<h3>Verification (' + s.pass + ' pass, ' + s.fail + ' fail)</h3>' +
      verifyHtml +
    '</div>' +
    '<div class="detail-section">' +
      '<h3>Performance Counters</h3>' +
      profileHtml +
    '</div>' +
    '<div class="detail-section">' +
      '<h3>Environment</h3>' +
      '<div class="detail-grid">' + envHtml + '</div>' +
    '</div>';
}

function copyRunCmds(runId, btn) {
  var run = null;
  for (var i = 0; i < DATA.length; i++) {
    if (DATA[i].run_id === runId) { run = DATA[i]; break; }
  }
  if (!run) return;
  var script = '#!/bin/bash\\n# Pipeline commands from Run ' + run.run_id + '\\n# ' + (run.env.date || '') + ' on ' + (run.env.hostname || '') + '\\n\\n' +
    run.timing.map(function(t) { return t.command; }).join('\\n') + '\\n';
  copyText(script, btn);
}

// ============ Tests ============
function renderTests(run) {
  var s = run.summary;
  var badge = document.getElementById('testsBadge');
  var hasTests = s.pass + s.fail > 0;

  if (hasTests) {
    badge.textContent = s.pass + '/' + (s.pass + s.fail) + ' passing';
    badge.className = 'badge ' + (s.all_pass ? 'badge-pass' : 'badge-fail');
  } else {
    badge.textContent = 'No Tests';
    badge.className = 'badge';
  }

  var tbody = document.querySelector('#testsTable tbody');
  if (run.verify.length > 0) {
    tbody.innerHTML = run.verify.map(function(v) {
      return '<tr data-status="' + v.status + '">' +
        '<td class="status-icon">' + (v.status === 'pass' ? '\\u2705' : '\\u274c') + '</td>' +
        '<td style="font-family:monospace;font-size:0.8rem">' + v.file + '</td>' +
        '<td>' + v.expected + '</td><td>' + v.reported + '</td>' +
        '<td style="color:' + (v.diff < 0.01 ? 'var(--green)' : 'var(--yellow)') + '">' + v.diff + '</td></tr>';
    }).join('');
  } else {
    tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text3);padding:24px;">' +
      'No verification data for this run. ' +
      (run.run_type === 'cpu_baseline' ? 'Baseline runs measure performance only.' : 'Run the test pipeline to generate verification results.') +
      '</td></tr>';
  }

  var totalTime = run.timing.reduce(function(a, t) { return a + t.time_s; }, 0);
  var ttbody = document.querySelector('#timingTable tbody');
  ttbody.innerHTML = run.timing.map(function(t, i) {
    var pct = ((t.time_s / totalTime) * 100).toFixed(1);
    return '<tr><td>' + (i+1) + '</td>' +
      '<td style="font-family:monospace;font-size:0.8rem">' + escHtml(t.command) + '</td>' +
      '<td>' + t.time_s.toFixed(3) + '</td><td>' + pct + '%%</td>' +
      '<td><div style="width:' + pct + '%%;height:6px;background:var(--accent);border-radius:3px;min-width:2px"></div></td>' +
      '<td><button class="cmd-copy" onclick="copySingleCmd(\\'' + escAttr(t.command) + '\\', this)">Copy</button></td></tr>';
  }).join('');
}

function filterTests(status, btn) {
  document.querySelectorAll('#page-tests .actions .btn').forEach(function(b) { b.classList.remove('active'); });
  btn.classList.add('active');
  document.querySelectorAll('#testsTable tbody tr').forEach(function(tr) {
    tr.style.display = (status === 'all' || tr.dataset.status === status) ? '' : 'none';
  });
}

// ============ Profiling ============
function switchProfile(idx) {
  currentProfileIdx = parseInt(idx);
  renderProfiling(DATA[currentRunIdx]);
}

function toggleCmds() {
  var el = document.getElementById('profilingCmds');
  el.style.display = el.style.display === 'none' ? '' : 'none';
}

function getActiveProfile() {
  if (PROFILES.length > 0 && currentProfileIdx >= 0 && currentProfileIdx < PROFILES.length)
    return PROFILES[currentProfileIdx];
  return null;
}

function renderProfileMetric(val, lbl, color) {
  return '<div class="profile-metric"><div class="val" style="color:' + color + '">' + val + '</div><div class="lbl">' + lbl + '</div></div>';
}

function renderProfiling(run) {
  var sel = document.getElementById('profRunSelector');

  // Build combined list: deep profiles + baseline runs with perf data
  var options = '';
  if (PROFILES.length > 0) {
    options += '<optgroup label="Deep Profiles">';
    PROFILES.forEach(function(p, i) {
      options += '<option value="deep-' + i + '"' + (currentProfileIdx === i && !window._profViewingRun ? ' selected' : '') + '>' +
        p.profile_id + ' (' + p.dataset + ', T' + p.threads + ')</option>';
    });
    options += '</optgroup>';
  }

  // Baseline runs with perf counter data
  var runsWithPerf = [];
  DATA.forEach(function(r, i) {
    if (r.profile && r.profile.metrics && r.profile.metrics.IPC) {
      runsWithPerf.push(i);
    }
  });
  if (runsWithPerf.length > 0) {
    options += '<optgroup label="Benchmark Runs (perf counters)">';
    runsWithPerf.forEach(function(i) {
      var r = DATA[i];
      var pf = r.profile;
      var ds = (pf.dataset || '?').replace('.fa','').replace('.phy','');
      options += '<option value="run-' + i + '"' + (window._profViewingRun === i ? ' selected' : '') + '>' +
        ds + ' T' + (pf.threads || 1) + ' — IPC ' + (pf.metrics.IPC || '?') + '</option>';
    });
    options += '</optgroup>';
  }

  sel.innerHTML = options;
  sel.onchange = function() {
    var v = this.value;
    if (v.indexOf('deep-') === 0) {
      window._profViewingRun = null;
      switchProfile(v.replace('deep-', ''));
    } else if (v.indexOf('run-') === 0) {
      var ri = parseInt(v.replace('run-', ''));
      window._profViewingRun = ri;
      currentRunIdx = ri;
      renderProfileConfig(null);
      renderBasicProfile(DATA[ri]);
    }
  };

  var prof = getActiveProfile();
  if (prof && !window._profViewingRun) {
    renderProfileConfig(prof);
    renderDeepProfile(prof);
  } else {
    renderProfileConfig(null);
    renderBasicProfile(run);
  }
}

function renderProfileConfig(prof) {
  var card = document.getElementById('profConfigCard');
  var content = document.getElementById('profConfigContent');
  if (!prof) { if (card) card.style.display = 'none'; return; }
  if (card) card.style.display = '';
  var sys = prof.system || {};
  var aln = prof.alignment || {};
  var iqtreeCmd = 'iqtree3 -s ' + escHtml(prof.dataset || '?') +
    (prof.model ? ' -m ' + escHtml(prof.model) : '') +
    ' -T ' + (prof.threads || 1) +
    ' --prefix output';

  function fmtFileSize(bytes) {
    if (!bytes) return 'N/A';
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
  }

  var html =
    '<div class="section-label" style="margin-top:0;">Alignment</div>' +
    '<div class="detail-grid" style="grid-template-columns:repeat(4,1fr);">' +
      '<div class="detail-kv"><div class="dk-label">Taxa (Sequences)</div><div class="dk-value" style="color:var(--accent2);font-size:1.2rem;">' + (aln.taxa || 'N/A') + '</div></div>' +
      '<div class="detail-kv"><div class="dk-label">Sites (Columns)</div><div class="dk-value" style="color:var(--accent2);font-size:1.2rem;">' + (aln.sites ? aln.sites.toLocaleString() : 'N/A') + '</div></div>' +
      '<div class="detail-kv"><div class="dk-label">Data Type</div><div class="dk-value">' + escHtml(aln.data_type || 'N/A') + '</div></div>' +
      '<div class="detail-kv"><div class="dk-label">File Size</div><div class="dk-value">' + fmtFileSize(aln.file_size_bytes) + '</div></div>' +
    '</div>' +
    '<div class="detail-grid" style="grid-template-columns:repeat(4,1fr);margin-top:8px;">' +
      '<div class="detail-kv"><div class="dk-label">Site Patterns</div><div class="dk-value">' + (aln.site_patterns ? aln.site_patterns.toLocaleString() : 'N/A') + '</div></div>' +
      '<div class="detail-kv"><div class="dk-label">Informative Sites</div><div class="dk-value">' + (aln.informative_sites ? aln.informative_sites.toLocaleString() : 'N/A') + '</div></div>' +
      '<div class="detail-kv"><div class="dk-label">Constant Sites</div><div class="dk-value">' + (aln.constant_sites != null ? aln.constant_sites + ' (' + (aln.constant_sites_pct || 0) + '%%)' : 'N/A') + '</div></div>' +
      '<div class="detail-kv"><div class="dk-label">Free Parameters</div><div class="dk-value">' + (aln.free_parameters || 'N/A') + '</div></div>' +
    '</div>' +

    '<div class="section-label">Model &amp; Results</div>' +
    '<div class="detail-grid" style="grid-template-columns:repeat(4,1fr);">' +
      '<div class="detail-kv"><div class="dk-label">Substitution Model</div><div class="dk-value" style="color:var(--accent);font-weight:600;">' + escHtml(aln.substitution_model || prof.model || 'AUTO') + '</div></div>' +
      '<div class="detail-kv"><div class="dk-label">Rate Heterogeneity</div><div class="dk-value">' + escHtml(aln.rate_model || 'N/A') + '</div></div>' +
      '<div class="detail-kv"><div class="dk-label">Gamma Shape \\u03b1</div><div class="dk-value">' + (aln.gamma_alpha || 'N/A') + '</div></div>' +
      '<div class="detail-kv"><div class="dk-label">Threads</div><div class="dk-value">' + (prof.threads || 'N/A') + '</div></div>' +
    '</div>' +
    '<div class="detail-grid" style="grid-template-columns:repeat(4,1fr);margin-top:8px;">' +
      '<div class="detail-kv"><div class="dk-label">Log-Likelihood</div><div class="dk-value" style="color:var(--green);font-family:monospace;">' + (aln.log_likelihood != null ? aln.log_likelihood.toLocaleString() : 'N/A') + '</div></div>' +
      '<div class="detail-kv"><div class="dk-label">BIC Score</div><div class="dk-value" style="font-family:monospace;">' + (aln.bic != null ? aln.bic.toLocaleString() : 'N/A') + '</div></div>' +
      '<div class="detail-kv"><div class="dk-label">Tree Length</div><div class="dk-value">' + (aln.tree_length || 'N/A') + '</div></div>' +
      '<div class="detail-kv"><div class="dk-label">Wall Time</div><div class="dk-value">' + (aln.wall_time_sec != null ? aln.wall_time_sec.toFixed(1) + 's' : 'N/A') + '</div></div>' +
    '</div>' +

    '<div class="section-label">System</div>' +
    '<div class="detail-grid" style="grid-template-columns:repeat(4,1fr);">' +
      '<div class="detail-kv"><div class="dk-label">CPU</div><div class="dk-value">' + escHtml(sys.cpu || 'N/A') + '</div></div>' +
      '<div class="detail-kv"><div class="dk-label">Cores</div><div class="dk-value">' + (sys.cores || 'N/A') + (sys.threads_per_core ? ' (' + sys.threads_per_core + 'T/core)' : '') + '</div></div>' +
      '<div class="detail-kv"><div class="dk-label">Memory</div><div class="dk-value">' + (sys.mem_total_gb ? sys.mem_total_gb + ' GB' : 'N/A') + '</div></div>' +
      '<div class="detail-kv"><div class="dk-label">L3 Cache</div><div class="dk-value">' + escHtml(sys.l3_cache || 'N/A') + '</div></div>' +
    '</div>' +
    '<div class="detail-grid" style="grid-template-columns:repeat(4,1fr);margin-top:8px;">' +
      '<div class="detail-kv"><div class="dk-label">GPU</div><div class="dk-value">' + escHtml(sys.gpu || 'N/A') + '</div></div>' +
      '<div class="detail-kv"><div class="dk-label">ROCm</div><div class="dk-value">' + escHtml(sys.rocm || 'N/A') + '</div></div>' +
      '<div class="detail-kv"><div class="dk-label">GCC</div><div class="dk-value">' + escHtml(sys.gcc || 'N/A') + '</div></div>' +
      '<div class="detail-kv"><div class="dk-label">NUMA Nodes</div><div class="dk-value">' + (sys.numa_nodes || 'N/A') + '</div></div>' +
    '</div>' +

    '<div style="margin-top:12px;padding:10px;background:var(--bg-tertiary);border-radius:6px;font-family:monospace;font-size:0.8rem;color:var(--text-secondary);display:flex;align-items:center;justify-content:space-between;">' +
      '<div><span style="color:var(--text-muted);margin-right:8px;">$</span>' + escHtml(iqtreeCmd) + '</div>' +
      '<button class="btn-sm" onclick="navigator.clipboard.writeText(\\'' + iqtreeCmd.replace(/'/g, "\\\\'") + '\\');this.textContent=\\'Copied!\\';var b=this;setTimeout(function(){b.textContent=\\'Copy Cmd\\';},1200)">Copy Cmd</button>' +
    '</div>';

  content.innerHTML = html;
  content.setAttribute('data-config', JSON.stringify({
    dataset: prof.dataset, model: prof.model, threads: prof.threads,
    alignment: aln, system: sys, iqtree_cmd: iqtreeCmd,
    profile_id: prof.profile_id, date: prof.date
  }));
}

function copyProfileConfig(btn) {
  var el = document.getElementById('profConfigContent');
  var data = JSON.parse(el.getAttribute('data-config') || '{}');
  var aln = data.alignment || {};
  var lines = [
    '=== IQ-TREE Run Configuration ===',
    'Profile ID: ' + (data.profile_id || 'N/A'),
    'Date: ' + (data.date || 'N/A'),
    '',
    '--- Alignment ---',
    'Dataset: ' + (data.dataset || 'N/A'),
    'Taxa: ' + (aln.taxa || 'N/A'),
    'Sites: ' + (aln.sites || 'N/A'),
    'Data Type: ' + (aln.data_type || 'N/A'),
    'Site Patterns: ' + (aln.site_patterns || 'N/A'),
    'Informative Sites: ' + (aln.informative_sites || 'N/A'),
    'Constant Sites: ' + (aln.constant_sites != null ? aln.constant_sites + ' (' + (aln.constant_sites_pct || 0) + '%%)' : 'N/A'),
    '',
    '--- Model & Results ---',
    'Model: ' + (aln.substitution_model || data.model || 'AUTO'),
    'Rate Heterogeneity: ' + (aln.rate_model || 'N/A'),
    'Gamma Alpha: ' + (aln.gamma_alpha || 'N/A'),
    'Threads: ' + (data.threads || 'N/A'),
    'Log-Likelihood: ' + (aln.log_likelihood || 'N/A'),
    'BIC Score: ' + (aln.bic || 'N/A'),
    'Tree Length: ' + (aln.tree_length || 'N/A'),
    'Wall Time: ' + (aln.wall_time_sec != null ? aln.wall_time_sec + 's' : 'N/A'),
    '',
    '--- System ---',
    'CPU: ' + ((data.system || {}).cpu || 'N/A'),
    'Cores: ' + ((data.system || {}).cores || 'N/A'),
    'Memory: ' + ((data.system || {}).mem_total_gb ? (data.system || {}).mem_total_gb + ' GB' : 'N/A'),
    'GPU: ' + ((data.system || {}).gpu || 'N/A'),
    'ROCm: ' + ((data.system || {}).rocm || 'N/A'),
    'GCC: ' + ((data.system || {}).gcc || 'N/A'),
    '',
    '--- Command ---',
    data.iqtree_cmd || ''
  ];
  navigator.clipboard.writeText(lines.join('\\n'));
  btn.textContent = 'Copied!';
  setTimeout(function() {
    btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy Config';
  }, 1500);
}

function renderDeepProfile(prof) {
  var cpu = prof.cpu || {};
  var c = cpu.counters || {};
  var d = cpu.derived || {};
  var hotspots = cpu.hotspots || [];
  var gpu = prof.gpu || {};
  var hw = gpu.hardware || {};
  var cmds = prof.commands || {};

  var cmdEl = document.getElementById('profilingCmds');
  var cmdItems = [
    { label: 'Modules', cmd: cmds.modules || '' },
    { label: 'perf stat', cmd: cmds.perf_stat || '' },
    { label: 'perf record', cmd: cmds.perf_record || '' },
    { label: 'rocprofv3 kernel', cmd: cmds.rocprofv3_kernel || '' },
    { label: 'rocprofv3 HIP', cmd: cmds.rocprofv3_hip || '' },
    { label: 'rocprofv3 memory', cmd: cmds.rocprofv3_memory || '' },
    { label: 'rocm-smi', cmd: cmds.rocm_smi || '' },
  ];
  cmdEl.innerHTML = cmdItems.filter(function(it) { return it.cmd; }).map(function(it) {
    return '<div class="cmd-item"><span class="cmd-label">' + it.label + '</span><code>' + escHtml(it.cmd) + '</code>' +
      '<button class="btn-sm" onclick="copySingleCmd(\\'' + escAttr(it.cmd) + '\\', this)">Copy</button></div>';
  }).join('');
  document.getElementById('profilingCmdsCard').style.display = '';

  var grid1 = document.getElementById('profileGrid');
  grid1.innerHTML = [
    renderProfileMetric(d.IPC || 'N/A', 'IPC', '#3b82f6'),
    renderProfileMetric(d['frontend-stall-rate'] != null ? d['frontend-stall-rate'] + '%%' : 'N/A', 'Frontend Stalls', '#ef4444'),
    renderProfileMetric(d['backend-stall-rate'] != null ? d['backend-stall-rate'] + '%%' : 'N/A', 'Backend Stalls', '#f97316'),
    renderProfileMetric(c.instructions ? (c.instructions / 1e9).toFixed(1) + 'B' : 'N/A', 'Instructions', '#8b5cf6'),
    renderProfileMetric(c.cycles ? (c.cycles / 1e9).toFixed(1) + 'B' : 'N/A', 'Cycles', '#06b6d4'),
  ].join('');

  var grid2 = document.getElementById('profileGrid2');
  grid2.innerHTML = [
    renderProfileMetric(d['cache-miss-rate'] != null ? d['cache-miss-rate'] + '%%' : 'N/A', 'Cache Miss Rate', '#eab308'),
    renderProfileMetric(d['L1-dcache-miss-rate'] != null ? d['L1-dcache-miss-rate'] + '%%' : 'N/A', 'L1 D-Cache Miss', '#f97316'),
    renderProfileMetric(d['LLC-miss-rate'] != null ? d['LLC-miss-rate'] + '%%' : 'N/A', 'LLC Miss Rate', '#ec4899'),
    renderProfileMetric(d['dTLB-miss-rate'] != null ? d['dTLB-miss-rate'] + '%%' : 'N/A', 'dTLB Miss Rate', '#14b8a6'),
    renderProfileMetric(d['branch-miss-rate'] != null ? d['branch-miss-rate'] + '%%' : 'N/A', 'Branch Mispredict', '#22c55e'),
  ].join('');

  if (charts.counter) charts.counter.destroy();
  var counterCtx = document.getElementById('counterChart').getContext('2d');
  var counterKeys = ['cycles','instructions','cache-references','cache-misses','branch-instructions','branch-misses',
    'L1-dcache-loads','L1-dcache-load-misses','LLC-loads','LLC-load-misses','dTLB-loads','dTLB-load-misses'];
  charts.counter = new Chart(counterCtx, {
    type: 'bar',
    data: {
      labels: counterKeys.map(function(k) { return k.replace('L1-dcache-','L1-').replace('-instructions','-instr').replace('-references','-refs').replace('-load-misses','-miss').replace('-loads',''); }),
      datasets: [{ data: counterKeys.map(function(k) { return c[k] || 0; }), backgroundColor: '#3b82f688', borderColor: '#3b82f6', borderWidth: 1, borderRadius: 4 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { color: '#94a3b8', maxRotation: 45, font: { size: 9 } } },
        y: { grid: { color: '#1e293b' }, ticks: { color: '#64748b' }, type: 'logarithmic' }
      }
    }
  });

  if (charts.stall) charts.stall.destroy();
  var stallCtx = document.getElementById('stallChart').getContext('2d');
  var feStall = d['frontend-stall-rate'] || 0;
  var beStall = d['backend-stall-rate'] || 0;
  var retiring = Math.max(0, 100 - feStall - beStall);
  charts.stall = new Chart(stallCtx, {
    type: 'doughnut',
    data: {
      labels: ['Frontend Stalls', 'Backend Stalls', 'Retiring/Useful'],
      datasets: [{ data: [feStall, beStall, retiring],
        backgroundColor: ['#ef4444', '#f97316', '#22c55e'],
        borderColor: '#0f1923', borderWidth: 2 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false, cutout: '55%%',
      plugins: { legend: { position: 'bottom', labels: { color: '#94a3b8', padding: 12 } } }
    }
  });

  var hsEl = document.getElementById('hotspotContainer');
  if (hotspots.length > 0) {
    hsEl.innerHTML = hotspots.map(function(h) {
      var barColor = h.percent > 30 ? '#ef4444' : h.percent > 10 ? '#f97316' : '#3b82f6';
      return '<div class="hotspot-row">' +
        '<div class="hotspot-pct" style="color:' + barColor + '">' + h.percent.toFixed(1) + '%%</div>' +
        '<div class="hotspot-bar-wrap"><div class="hotspot-bar" style="width:' + Math.min(h.percent * 2, 100) + '%%;background:' + barColor + '"></div></div>' +
        '<div class="hotspot-func" title="' + escHtml(h['function']) + '">' + escHtml(h['function']) + '</div>' +
        '<div class="hotspot-module">' + escHtml(h.module || '') + '</div></div>';
    }).join('');
  } else {
    hsEl.innerHTML = '<div class="no-data-msg">No hotspot data. Run deep profiling with perf record.</div>';
  }

  var gpuGrid = document.getElementById('profileGpuGrid');
  if (Object.keys(hw).length > 0) {
    gpuGrid.innerHTML = [
      renderProfileMetric(hw.temperature_c != null ? hw.temperature_c + '\\u00b0C' : 'N/A', 'Temperature', '#22c55e'),
      renderProfileMetric(hw.power_w != null ? hw.power_w + 'W' : 'N/A', 'Power', '#eab308'),
      renderProfileMetric(hw.utilization_pct != null ? hw.utilization_pct + '%%' : 'N/A', 'GPU Util', '#3b82f6'),
      renderProfileMetric(hw.vram_used_mb != null ? (hw.vram_used_mb / 1024).toFixed(1) + 'GB' : 'N/A', 'VRAM Used', '#8b5cf6'),
      renderProfileMetric(hw.clock_mhz != null ? hw.clock_mhz + 'MHz' : 'N/A', 'GPU Clock', '#06b6d4'),
    ].join('');
  } else {
    gpuGrid.innerHTML = '<div class="no-data-msg" style="grid-column:1/-1;">No GPU hardware metrics. Run deep profiling on a GPU node.</div>';
  }

  var traceEl = document.getElementById('gpuTraceContainer');
  var kernels = gpu.kernels || [];
  var hipCalls = gpu.hip_calls || [];
  var memCopies = gpu.memory_copies || [];

  if (kernels.length > 0) {
    traceEl.innerHTML = '<h3 style="margin:12px 16px 8px;font-size:0.85rem;color:var(--text3);">GPU Kernel Dispatches</h3>' +
      '<table class="gpu-trace-table"><thead><tr><th>Kernel</th><th>Duration</th><th>Grid Size</th><th>Block Size</th></tr></thead><tbody>' +
      kernels.map(function(k) {
        return '<tr><td style="font-family:monospace;font-size:0.75rem;">' + escHtml(k.kernel_name) + '</td>' +
          '<td>' + (k.duration_ns / 1e6).toFixed(3) + ' ms</td><td>' + k.grid_size + '</td><td>' + k.block_size + '</td></tr>';
      }).join('') + '</tbody></table>';
  } else if (hipCalls.length > 0) {
    traceEl.innerHTML = '<h3 style="margin:12px 16px 8px;font-size:0.85rem;color:var(--text3);">HIP API Calls</h3>' +
      '<table class="gpu-trace-table"><thead><tr><th>Function</th><th>Duration</th><th>Correlation ID</th></tr></thead><tbody>' +
      hipCalls.slice(0, 50).map(function(h) {
        return '<tr><td style="font-family:monospace;font-size:0.75rem;">' + escHtml(h['function']) + '</td>' +
          '<td>' + (h.duration_ns / 1e6).toFixed(3) + ' ms</td><td>' + h.correlation_id + '</td></tr>';
      }).join('') + '</tbody></table>';
  } else {
    traceEl.innerHTML = '<div class="no-data-msg">No GPU kernel dispatches detected. GPU traces will appear after HIP kernels are integrated.</div>';
  }

  if (memCopies.length > 0) {
    traceEl.innerHTML += '<h3 style="margin:20px 16px 8px;font-size:0.85rem;color:var(--text3);">Memory Copies</h3>' +
      '<table class="gpu-trace-table"><thead><tr><th>Direction</th><th>Size</th><th>Duration</th></tr></thead><tbody>' +
      memCopies.map(function(mc) {
        var sizeStr = mc.size_bytes > 1e6 ? (mc.size_bytes / 1e6).toFixed(1) + ' MB' : (mc.size_bytes / 1e3).toFixed(1) + ' KB';
        return '<tr><td>' + mc.direction + '</td><td>' + sizeStr + '</td><td>' + (mc.duration_ns / 1e6).toFixed(3) + ' ms</td></tr>';
      }).join('') + '</tbody></table>';
  }

  if (charts.ipcTrend) charts.ipcTrend.destroy();
  var ipcCtx = document.getElementById('ipcTrendChart').getContext('2d');
  var ipcSource = PROFILES.length > 0 ? PROFILES : DATA;
  charts.ipcTrend = new Chart(ipcCtx, {
    type: 'line',
    data: {
      labels: ipcSource.map(function(r) { return (r.profile_id || r.run_id || '').substring(0, 10); }),
      datasets: [{
        label: 'IPC',
        data: ipcSource.map(function(r) {
          if (r.cpu && r.cpu.derived) return r.cpu.derived.IPC || null;
          return (r.profile && r.profile.metrics && r.profile.metrics.IPC) || null;
        }),
        borderColor: '#8b5cf6', backgroundColor: 'rgba(139,92,246,0.1)',
        fill: true, tension: 0.3, pointRadius: 6, pointBackgroundColor: '#8b5cf6',
        pointBorderColor: '#1a2332', pointBorderWidth: 2, spanGaps: true
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { backgroundColor: '#1a2332', borderColor: '#2a3444', borderWidth: 1, cornerRadius: 8 } },
      scales: {
        x: { grid: { color: '#1e293b' }, ticks: { color: '#64748b' } },
        y: { grid: { color: '#1e293b' }, ticks: { color: '#64748b' }, beginAtZero: true }
      }
    }
  });

  if (charts.stallTrend) charts.stallTrend.destroy();
  var stallTrendCtx = document.getElementById('stallTrendChart').getContext('2d');
  if (PROFILES.length > 0) {
    charts.stallTrend = new Chart(stallTrendCtx, {
      type: 'line',
      data: {
        labels: PROFILES.map(function(p) { return (p.profile_id || '').substring(0, 10); }),
        datasets: [
          { label: 'Frontend Stalls %%', data: PROFILES.map(function(p) { return (p.cpu && p.cpu.derived) ? p.cpu.derived['frontend-stall-rate'] : null; }),
            borderColor: '#ef4444', fill: false, tension: 0.3, pointRadius: 5, spanGaps: true },
          { label: 'Backend Stalls %%', data: PROFILES.map(function(p) { return (p.cpu && p.cpu.derived) ? p.cpu.derived['backend-stall-rate'] : null; }),
            borderColor: '#f97316', fill: false, tension: 0.3, pointRadius: 5, spanGaps: true }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom', labels: { color: '#94a3b8' } } },
        scales: {
          x: { grid: { color: '#1e293b' }, ticks: { color: '#64748b' } },
          y: { grid: { color: '#1e293b' }, ticks: { color: '#64748b' }, beginAtZero: true, max: 100 }
        }
      }
    });
  }

  // Render flamegraph and call stack
  var rawStacks = cpu.folded_stacks || [];
  var hotspots = cpu.hotspots || [];
  var cleanedStacks = cleanFoldedStacks(rawStacks);
  var unknownRate = computeUnknownRate(cleanedStacks);
  renderFlamegraph(cleanedStacks, hotspots, unknownRate);
  renderCallStack(cleanedStacks, hotspots, unknownRate);
}

// ============ Stack Cleaning ============
// perf on stripped C++ binaries produces artifacts:
// - "void", "double", "non-virtual", "virtual" are demangled return types, not functions
// - consecutive [unknown] frames are unresolved addresses that should be collapsed
// - truncated C++ signatures like "std::__cxx11::basic_string<char," need cleanup

var CPP_RETURN_TYPES = {'void':1, 'double':1, 'int':1, 'float':1, 'bool':1, 'char':1, 'unsigned':1, 'long':1, 'short':1, 'virtual':1, 'non-virtual':1};

function cleanFrameName(name) {
  // Truncated C++ templates: add closing bracket
  if (name.indexOf('<') >= 0 && name.indexOf('>') < 0) {
    name = name.replace(/,\\s*$/, '') + '...>';
  }
  // Truncated C++ parameter lists
  if (name.indexOf('(') >= 0 && name.indexOf(')') < 0) {
    name = name.replace(/,\\s*$/, '') + '...';
  }
  // Shorten std:: prefixes
  name = name.replace(/std::__cxx11::basic_string<char,?[^>]*>?/g, 'std::string');
  name = name.replace(/std::basic_ifstream<char,?[^>]*>?/g, 'std::ifstream');
  name = name.replace(/std::basic_streambuf<char,?[^>]*>?/g, 'std::streambuf');
  name = name.replace(/std::basic_ostream<char,?[^>]*>?/g, 'std::ostream');
  // Strip long param lists — keep just class::method(...)
  var m = name.match(/^([A-Za-z_]\\w*(?:::[A-Za-z_~]\\w*)*)\\((.{40,})\\)/);
  if (m) name = m[1] + '(...)';
  // Strip @GLIBC version tags
  name = name.replace(/@+[A-Z_]+[0-9_.]+/g, '');
  return name;
}

function cleanFoldedStacks(stacks) {
  if (!stacks || stacks.length === 0) return stacks;
  var merged = {};
  stacks.forEach(function(s) {
    var frames = s.stack.split(';');
    var cleaned = [];
    var prevFrame = '';
    for (var i = 0; i < frames.length; i++) {
      var f = frames[i].trim();
      // Skip bare C++ return types (perf demangling artifacts)
      if (CPP_RETURN_TYPES[f]) {
        // Try to merge with next frame: "void" + "PhyloTree::foo" -> "void PhyloTree::foo"
        if (i + 1 < frames.length && !CPP_RETURN_TYPES[frames[i+1].trim()] && frames[i+1].trim() !== '[unknown]') {
          frames[i+1] = f + ' ' + frames[i+1].trim();
        }
        continue;
      }
      // Collapse consecutive [unknown] into one
      if (f === '[unknown]') {
        if (prevFrame === '[unknown]') continue;
        cleaned.push(f);
        prevFrame = f;
        continue;
      }
      // Clean C++ name
      f = cleanFrameName(f);
      cleaned.push(f);
      prevFrame = f;
    }
    // Remove trailing [unknown]
    while (cleaned.length > 1 && cleaned[cleaned.length - 1] === '[unknown]') cleaned.pop();
    // Remove leading [unknown] if followed by a known function
    while (cleaned.length > 1 && cleaned[0] === '[unknown]') cleaned.shift();
    if (cleaned.length === 0) cleaned.push('[unknown]');
    var key = cleaned.join(';');
    merged[key] = (merged[key] || 0) + s.count;
  });
  var result = [];
  for (var k in merged) {
    result.push({stack: k, count: merged[k]});
  }
  return result;
}

function computeUnknownRate(stacks) {
  if (!stacks || stacks.length === 0) return 0;
  var total = 0, unknown = 0;
  stacks.forEach(function(s) {
    total += s.count;
    if (s.stack === '[unknown]') unknown += s.count;
  });
  return total > 0 ? unknown / total : 0;
}

// Build synthetic flamegraph tree from hotspot data (IP-based sampling — no unwinding needed)
// Groups by: module → function name. Much more reliable than folded stacks on stripped binaries.
function buildHotspotTree(hotspots) {
  var totalSamples = hotspots.reduce(function(s, h) { return s + h.samples; }, 0);
  var root = {name: 'all', value: totalSamples, self: 0, childArr: []};

  // Group by module
  var modules = {};
  hotspots.forEach(function(h) {
    var mod = h.module || 'unknown';
    // Normalize module name
    if (mod.indexOf('libgomp') >= 0) mod = 'libgomp (OpenMP)';
    else if (mod.indexOf('libc') >= 0 || mod.indexOf('glibc') >= 0) mod = 'libc';
    else if (mod.indexOf('iqtree') >= 0) mod = 'iqtree3';
    if (!modules[mod]) modules[mod] = {name: mod, value: 0, self: 0, childArr: []};
    modules[mod].value += h.samples;

    // Clean function name
    var fname = h['function'] || '[unknown]';
    // Remove return type prefix (void/double/etc.)
    fname = fname.replace(/^(void|double|int|float|bool)\\s+/, '');
    // Shorten template params
    fname = fname.replace(/PhyloTree::compute(Likelihood|Partial)(\\w+)SIMD<[^>]+>/g, function(m, a, b) {
      return 'PhyloTree::compute' + a + b + 'SIMD<...>';
    });
    // Replace hex addresses (stripped symbols from libgomp etc.)
    if (/^0x[0-9a-f]+$/.test(fname.trim())) {
      fname = mod.replace(' (OpenMP)', '') + ' internal';
    }
    // Truncate long names
    if (fname.length > 70) fname = fname.substring(0, 67) + '...';

    var fnode = {name: fname, value: h.samples, self: h.samples, childArr: [], pct: h.percent};
    modules[mod].childArr.push(fnode);
  });

  // Sort modules by value desc
  var modArr = [];
  for (var m in modules) modArr.push(modules[m]);
  modArr.sort(function(a, b) { return b.value - a.value; });

  // Sort functions within each module by value desc
  modArr.forEach(function(mod) {
    mod.childArr.sort(function(a, b) { return b.value - a.value; });
  });

  root.childArr = modArr;
  return root;
}

// ============ Flamegraph ============
var flameData = null;
var flameHotspotData = null;
var flameZoomStack = [];
var flameSearchTerm = '';
var flameTotalSamples = 0;
var flameMode = 'hotspot'; // 'hotspot' | 'callchain'

function buildFlameTree(stacks) {
  var root = {name: 'all', value: 0, self: 0, children: {}, depth: 0};
  stacks.forEach(function(s) {
    var frames = s.stack.split(';');
    var node = root;
    root.value += s.count;
    frames.forEach(function(frame, idx) {
      if (!node.children[frame]) {
        node.children[frame] = {name: frame, value: 0, self: 0, children: {}, depth: idx + 1};
      }
      node.children[frame].value += s.count;
      if (idx === frames.length - 1) node.children[frame].self += s.count;
      node = node.children[frame];
    });
  });
  function toArray(node) {
    var arr = [];
    for (var k in node.children) {
      var child = node.children[k];
      child.childArr = toArray(child);
      arr.push(child);
    }
    // Sort alphabetically per Brendan Gregg convention (merges towers)
    arr.sort(function(a, b) { return a.name.localeCompare(b.name); });
    return arr;
  }
  root.childArr = toArray(root);
  return root;
}

function flameColor(name) {
  if (name === 'all') return '#334155';
  if (name === '[unknown]') return '#4a5568';
  if (name.indexOf('GOMP') >= 0 || name.indexOf('pthread') >= 0 || name.indexOf('omp') >= 0 || name.indexOf('start_thread') >= 0 || name.indexOf('libgomp') >= 0) return '#3b82f6';
  if (name.indexOf('Likelihood') >= 0 || name.indexOf('LH') >= 0 || name.indexOf('SIMD') >= 0 || name.indexOf('Derv') >= 0 || name.indexOf('Buffer') >= 0) return '#ef4444';
  if (name.indexOf('Phylo') >= 0 || name.indexOf('Tree') >= 0 || name.indexOf('Alignment') >= 0 || name.indexOf('Node') >= 0) return '#f97316';
  if (name.indexOf('Parsimony') >= 0 || name.indexOf('parsimony') >= 0) return '#a855f7';
  if (name.indexOf('alloc') >= 0 || name.indexOf('malloc') >= 0 || name.indexOf('free') >= 0 || name.indexOf('mmap') >= 0) return '#eab308';
  // Warm random color per Brendan Gregg's convention
  var h = 0;
  for (var i = 0; i < name.length; i++) h = ((h << 5) - h + name.charCodeAt(i)) | 0;
  var hue = 20 + (Math.abs(h) %% 35);
  var sat = 60 + (Math.abs(h >> 8) %% 30);
  var lit = 48 + (Math.abs(h >> 16) %% 12);
  return 'hsl(' + hue + ',' + sat + '%%,' + lit + '%%)';
}

function flameCategoryLabel(name) {
  if (name === '[unknown]') return 'unknown';
  if (name.indexOf('GOMP') >= 0 || name.indexOf('pthread') >= 0 || name.indexOf('omp') >= 0 || name.indexOf('libgomp') >= 0) return 'openmp';
  if (name.indexOf('Likelihood') >= 0 || name.indexOf('LH') >= 0 || name.indexOf('SIMD') >= 0 || name.indexOf('Derv') >= 0 || name.indexOf('Buffer') >= 0) return 'likelihood';
  if (name.indexOf('Phylo') >= 0 || name.indexOf('Tree') >= 0) return 'phylo';
  if (name.indexOf('Parsimony') >= 0 || name.indexOf('parsimony') >= 0) return 'parsimony';
  if (name.indexOf('alloc') >= 0 || name.indexOf('malloc') >= 0 || name.indexOf('free') >= 0) return 'memory';
  return '';
}

function renderFlamegraph(stacks, hotspots, unknownRate) {
  var container = document.getElementById('flamegraphContainer');
  var resetBtn = document.getElementById('flamegraphZoomReset');
  var banner = document.getElementById('flameBanner');
  var toggleBtn = document.getElementById('flameToggleMode');

  flameHotspotData = (hotspots && hotspots.length > 0) ? buildHotspotTree(hotspots) : null;
  flameZoomStack = [];
  flameSearchTerm = '';
  document.getElementById('flameSearch').value = '';
  if (resetBtn) resetBtn.style.display = 'none';

  // Show warning banner if unknown rate is high
  if (banner) {
    var unkPct = (unknownRate * 100).toFixed(0);
    if (unknownRate > 0.3) {
      banner.style.display = '';
      banner.innerHTML =
        '<span class="fbanner-icon">&#9888;</span>' +
        '<span><strong>' + unkPct + '%% of call-chain samples are unresolved.</strong> ' +
        'The binary was compiled without <code>-fno-omit-frame-pointer</code> so perf cannot unwind the stack. ' +
        'Showing <strong>IP-sampled hotspot view</strong> (accurate). ' +
        '<a href="#" onclick="switchFlameMode(&#39;callchain&#39;);return false;" style="color:var(--accent);">Switch to raw call chains</a>.' +
        '</span>' +
        '<div class="fbanner-fix"><strong>Fix:</strong> Rebuild IQ-TREE with ' +
        '<code>cmake .. -DCMAKE_CXX_FLAGS="-fno-omit-frame-pointer" -DCMAKE_C_FLAGS="-fno-omit-frame-pointer"</code>' +
        '</div>';
      // Default to hotspot view when unknown rate is high
      flameMode = 'hotspot';
    } else {
      banner.style.display = 'none';
      flameMode = 'callchain';
    }
  }

  if (toggleBtn) {
    toggleBtn.style.display = (flameHotspotData && stacks && stacks.length > 0) ? '' : 'none';
    toggleBtn.textContent = flameMode === 'hotspot' ? 'Switch to Call Chains' : 'Switch to Hotspot View';
  }

  if (!stacks || stacks.length === 0) {
    if (flameHotspotData) {
      drawHotspotFlame(flameHotspotData);
    } else {
      container.innerHTML = '<p class="no-data-msg">No profiling data available.</p>';
    }
    updateBreadcrumb();
    return;
  }

  flameData = buildFlameTree(stacks);
  flameTotalSamples = flameMode === 'hotspot' && flameHotspotData ? flameHotspotData.value : flameData.value;

  if (flameMode === 'hotspot' && flameHotspotData) {
    drawHotspotFlame(flameHotspotData);
  } else {
    drawFlame(flameData);
  }
  updateBreadcrumb();
}

function switchFlameMode(mode) {
  flameMode = mode;
  var toggleBtn = document.getElementById('flameToggleMode');
  if (toggleBtn) toggleBtn.textContent = mode === 'hotspot' ? 'Switch to Call Chains' : 'Switch to Hotspot View';
  flameZoomStack = [];
  if (mode === 'hotspot' && flameHotspotData) {
    flameTotalSamples = flameHotspotData.value;
    drawHotspotFlame(flameHotspotData);
  } else if (flameData) {
    flameTotalSamples = flameData.value;
    drawFlame(flameData);
  }
  document.getElementById('flamegraphZoomReset').style.display = 'none';
  updateBreadcrumb();
}

// Draw hotspot-based icicle: root → module → function
function drawHotspotFlame(root) {
  var container = document.getElementById('flamegraphContainer');
  container.innerHTML = '';
  flameTotalSamples = root.value;

  // Row 0: root
  var rootRow = document.createElement('div');
  rootRow.className = 'flame-row';
  var rootEl = document.createElement('div');
  rootEl.className = 'flame-frame';
  rootEl.style.width = '100%%';
  rootEl.style.background = '#334155';
  rootEl.textContent = 'all  (' + root.value + ' IP samples)';
  rootRow.appendChild(rootEl);
  container.appendChild(rootRow);

  // Row 1: modules
  var modRow = document.createElement('div');
  modRow.className = 'flame-row';
  root.childArr.forEach(function(mod) {
    var pct = mod.value / root.value * 100;
    if (pct < 0.2) return;
    var el = document.createElement('div');
    el.className = 'flame-frame';
    el.style.width = pct + '%%';
    el.style.background = flameColor(mod.name);
    if (pct > 5) el.textContent = mod.name;
    el.addEventListener('mouseenter', function(e) { showFlameTooltip(e, mod); });
    el.addEventListener('mouseleave', hideFlameTooltip);
    modRow.appendChild(el);
  });
  container.appendChild(modRow);

  // Row 2: functions (all laid out inline by module proportion)
  var fnRow = document.createElement('div');
  fnRow.className = 'flame-row';
  root.childArr.forEach(function(mod) {
    mod.childArr.forEach(function(fn) {
      var pct = fn.value / root.value * 100;
      if (pct < 0.2) return;
      var el = document.createElement('div');
      el.className = 'flame-frame';
      el.style.width = pct + '%%';
      el.style.background = flameColor(fn.name);
      // Show label if wide enough
      if (pct > 12) {
        el.textContent = fn.name.length > 55 ? fn.name.substring(0, 52) + '...' : fn.name;
      } else if (pct > 5) {
        var short = fn.name.replace(/.*::/, '').replace(/\\(.*/, '');
        el.textContent = short.substring(0, 22);
      }
      el.addEventListener('mouseenter', function(e) { showFlameTooltip(e, fn); });
      el.addEventListener('mouseleave', hideFlameTooltip);
      fnRow.appendChild(el);
    });
  });
  container.appendChild(fnRow);

  // Row 3: GPU offload opportunity indicator
  var gpuRow = document.createElement('div');
  gpuRow.className = 'flame-row';
  var gpuTotal = 0;
  root.childArr.forEach(function(mod) {
    if (mod.name === 'iqtree3') {
      mod.childArr.forEach(function(fn) {
        if (fn.name.indexOf('LikelihoodDerv') >= 0 || fn.name.indexOf('LikelihoodPartial') >= 0 ||
            fn.name.indexOf('LikelihoodBuffer') >= 0 || fn.name.indexOf('LikelihoodBranch') >= 0 ||
            fn.name.indexOf('LikelihoodFrom') >= 0 || fn.name.indexOf('PartialLikelihood') >= 0) {
          gpuTotal += fn.value;
        }
      });
    }
  });
  if (gpuTotal > 0) {
    var gpuPct = gpuTotal / root.value * 100;
    var gpuEl = document.createElement('div');
    gpuEl.className = 'flame-frame';
    gpuEl.style.width = gpuPct + '%%';
    gpuEl.style.background = 'repeating-linear-gradient(45deg, #1d4ed8, #1d4ed8 4px, #2563eb 4px, #2563eb 8px)';
    gpuEl.style.opacity = '0.85';
    gpuEl.style.fontSize = '0.55rem';
    gpuEl.textContent = gpuPct > 5 ? 'GPU offload target: ' + gpuPct.toFixed(1) + '%%' : '';
    gpuEl.title = 'GPU offload opportunity: ' + gpuPct.toFixed(1) + '%% of CPU time';
    gpuRow.appendChild(gpuEl);
    container.appendChild(gpuRow);
  }
}

function drawFlame(root) {
  var container = document.getElementById('flamegraphContainer');
  container.innerHTML = '';
  var viewTotal = root.value;

  // Icicle layout: root at top, children below (Brendan Gregg style inverted)
  // Build rows top-down
  var rows = [];
  function collectRows(nodes, total, depth) {
    if (nodes.length === 0) return;
    if (!rows[depth]) rows[depth] = [];
    nodes.forEach(function(node) {
      var pct = (node.value / total * 100);
      if (pct < 0.3) return;
      rows[depth].push({node: node, pct: pct, total: total});
    });
    // Recurse children
    nodes.forEach(function(node) {
      if (node.childArr && node.childArr.length > 0) {
        collectRows(node.childArr, total, depth + 1);
      }
    });
  }

  // Root row
  rows[0] = [{node: root, pct: 100, total: viewTotal}];
  collectRows(root.childArr || [], viewTotal, 1);

  rows.forEach(function(rowItems, depth) {
    if (!rowItems || rowItems.length === 0) return;
    var row = document.createElement('div');
    row.className = 'flame-row';
    rowItems.forEach(function(item) {
      var node = item.node;
      var pct = item.pct;
      var el = document.createElement('div');
      el.className = 'flame-frame';
      if (flameSearchTerm && node.name.toLowerCase().indexOf(flameSearchTerm) === -1 && node.name !== 'all') {
        el.classList.add('dim');
      }
      if (flameSearchTerm && node.name.toLowerCase().indexOf(flameSearchTerm) >= 0) {
        el.classList.add('highlight');
      }
      el.style.width = pct + '%%';
      el.style.background = flameColor(node.name);
      // Smarter label: show name if wide enough, abbreviate if medium
      var globalPct = (node.value / flameTotalSamples * 100).toFixed(1);
      if (pct > 8) {
        el.textContent = node.name.length > 60 ? node.name.substring(0, 57) + '...' : node.name;
      } else if (pct > 3) {
        // Show abbreviated
        var shortName = node.name.replace(/.*::/, '').replace(/\\(.*/, '');
        el.textContent = shortName.substring(0, 20);
      } else {
        el.textContent = '';
      }
      el.setAttribute('data-name', node.name);
      el.setAttribute('data-count', node.value);
      el.setAttribute('data-self', node.self || 0);
      el.addEventListener('mouseenter', function(e) { showFlameTooltip(e, node); });
      el.addEventListener('mouseleave', hideFlameTooltip);
      if (node.childArr && node.childArr.length > 0) {
        el.style.cursor = 'pointer';
        el.addEventListener('click', function() { zoomFlame(node); });
      }
      row.appendChild(el);
    });
    container.appendChild(row);
  });
}

function zoomFlame(node) {
  flameZoomStack.push(node.name);
  drawFlame(node);
  document.getElementById('flamegraphZoomReset').style.display = '';
  updateBreadcrumb();
}

function resetFlameZoom() {
  if (flameData) drawFlame(flameData);
  flameZoomStack = [];
  document.getElementById('flamegraphZoomReset').style.display = 'none';
  updateBreadcrumb();
}

function zoomToBreadcrumb(idx) {
  // Navigate to a specific level in the breadcrumb
  if (idx < 0) { resetFlameZoom(); return; }
  var node = flameData;
  for (var i = 0; i <= idx; i++) {
    var targetName = flameZoomStack[i];
    if (node.childArr) {
      for (var j = 0; j < node.childArr.length; j++) {
        if (node.childArr[j].name === targetName) { node = node.childArr[j]; break; }
      }
    }
  }
  flameZoomStack = flameZoomStack.slice(0, idx + 1);
  drawFlame(node);
  updateBreadcrumb();
}

function updateBreadcrumb() {
  var bc = document.getElementById('flameBreadcrumb');
  var parts = ['<span onclick="resetFlameZoom()">all</span>'];
  flameZoomStack.forEach(function(name, i) {
    parts.push('<span style="color:var(--text3);">\\u203a</span>');
    var isLast = i === flameZoomStack.length - 1;
    var shortName = name.length > 30 ? name.substring(0, 27) + '...' : name;
    if (isLast) {
      parts.push('<span class="current">' + escHtml(shortName) + '</span>');
    } else {
      parts.push('<span onclick="zoomToBreadcrumb(' + i + ')">' + escHtml(shortName) + '</span>');
    }
  });
  bc.innerHTML = parts.join('');
}

function filterFlame(term) {
  flameSearchTerm = term.toLowerCase().trim();
  // Count matching samples
  var matchCount = 0;
  if (flameSearchTerm && flameData) {
    function countMatches(node) {
      if (node.name.toLowerCase().indexOf(flameSearchTerm) >= 0) matchCount += node.self || 0;
      if (node.childArr) node.childArr.forEach(countMatches);
    }
    countMatches(flameData);
  }
  // Redraw with current zoom
  var currentRoot = flameData;
  if (flameZoomStack.length > 0) {
    for (var i = 0; i < flameZoomStack.length; i++) {
      var targetName = flameZoomStack[i];
      if (currentRoot.childArr) {
        for (var j = 0; j < currentRoot.childArr.length; j++) {
          if (currentRoot.childArr[j].name === targetName) { currentRoot = currentRoot.childArr[j]; break; }
        }
      }
    }
  }
  drawFlame(currentRoot);
}

var flameTooltipEl = null;
function showFlameTooltip(e, node) {
  if (!flameTooltipEl) {
    flameTooltipEl = document.createElement('div');
    flameTooltipEl.className = 'flame-tooltip';
    document.body.appendChild(flameTooltipEl);
  }
  var pct = (node.value / flameTotalSamples * 100).toFixed(2);
  var selfPct = ((node.self || 0) / flameTotalSamples * 100).toFixed(2);
  var barColor = flameColor(node.name);
  flameTooltipEl.innerHTML =
    '<div class="ft-name">' + escHtml(node.name) + '</div>' +
    '<div class="ft-row"><span>Total</span><span class="ft-val">' + node.value.toLocaleString() + ' samples (' + pct + '%%)</span></div>' +
    '<div class="ft-row"><span>Self</span><span class="ft-val">' + (node.self || 0).toLocaleString() + ' samples (' + selfPct + '%%)</span></div>' +
    '<div class="ft-row"><span>Children</span><span>' + (node.childArr ? node.childArr.length : 0) + '</span></div>' +
    '<div class="ft-bar"><div class="ft-bar-fill" style="width:' + pct + '%%;background:' + barColor + ';"></div></div>';
  flameTooltipEl.style.display = '';
  var x = Math.min(e.clientX + 12, window.innerWidth - 440);
  var y = e.clientY + 20;
  if (y + 120 > window.innerHeight) y = e.clientY - 120;
  flameTooltipEl.style.left = x + 'px';
  flameTooltipEl.style.top = y + 'px';
}
function hideFlameTooltip() {
  if (flameTooltipEl) flameTooltipEl.style.display = 'none';
}

// ============ Call Stack ============
var callStackSortByCount = true;
var callStackShowAll = false;

function renderCallStack(stacks, hotspots, unknownRate) {
  var container = document.getElementById('callStackContainer');
  var summary = document.getElementById('callStackSummary');
  if (!stacks || stacks.length === 0) {
    container.innerHTML = '<p class="no-data-msg">No call stack data available. Run deep profiling with perf record to generate.</p>';
    if (summary) summary.textContent = '';
    return;
  }
  // Filter out pure-unknown stacks — they carry no useful call chain information
  var usefulStacks = stacks.filter(function(s) { return s.stack !== '[unknown]'; });
  var hiddenCount = stacks.reduce(function(acc, s) { return acc + (s.stack === '[unknown]' ? s.count : 0); }, 0);
  container.setAttribute('data-stacks', JSON.stringify(usefulStacks));
  callStackShowAll = false;

  // Add a note about filtered stacks when unknown rate is significant
  var noteEl = document.getElementById('callStackNote');
  if (!noteEl) {
    noteEl = document.createElement('div');
    noteEl.id = 'callStackNote';
    noteEl.style.cssText = 'font-size:0.72rem;color:#64748b;padding:4px 10px 2px;';
    container.parentNode.insertBefore(noteEl, container);
  }
  if (hiddenCount > 0 && unknownRate > 0.1) {
    noteEl.innerHTML =
      '<span style="color:#f59e0b;">&#9888;</span> ' +
      hiddenCount + ' samples (' + (unknownRate * 100).toFixed(0) + '%%) hidden — unresolvable without <code>-fno-omit-frame-pointer</code>. ' +
      'Showing ' + usefulStacks.length + ' call chains with known functions.';
  } else {
    noteEl.textContent = '';
  }
  drawCallStack(usefulStacks, true);
}

function csFrameClass(name) {
  if (name === '[unknown]') return 'cs-unk';
  if (name.indexOf('GOMP') >= 0 || name.indexOf('pthread') >= 0 || name.indexOf('omp') >= 0 || name.indexOf('libgomp') >= 0) return 'cs-omp';
  if (name.indexOf('Likelihood') >= 0 || name.indexOf('LH') >= 0 || name.indexOf('SIMD') >= 0 || name.indexOf('Derv') >= 0 || name.indexOf('Buffer') >= 0) return 'cs-lh';
  if (name.indexOf('Phylo') >= 0 || name.indexOf('Tree') >= 0) return 'cs-phylo';
  return '';
}

function drawCallStack(stacks, byCount) {
  var container = document.getElementById('callStackContainer');
  var sorted = stacks.slice().sort(function(a, b) {
    return byCount ? b.count - a.count : a.stack.localeCompare(b.stack);
  });
  var totalSamples = sorted.reduce(function(s, x) { return s + x.count; }, 0);
  var maxCount = sorted.length > 0 ? sorted[0].count : 1;
  var showCount = callStackShowAll ? sorted.length : Math.min(25, sorted.length);
  var visible = sorted.slice(0, showCount);

  // Summary
  var summaryEl = document.getElementById('callStackSummary');
  if (summaryEl) {
    summaryEl.textContent = sorted.length + ' unique paths | ' + totalSamples.toLocaleString() + ' total samples | showing ' + showCount;
  }

  var html = visible.map(function(s) {
    var frames = s.stack.split(';');
    var pct = (s.count / totalSamples * 100);
    var barPct = (s.count / maxCount * 100);
    var lastFrame = frames[frames.length - 1];
    var barColor = flameColor(lastFrame);

    var formattedFrames = frames.map(function(f, i) {
      var cls = (i === frames.length - 1) ? 'cs-leaf' : csFrameClass(f);
      return '<span class="' + cls + '">' + escHtml(f) + '</span>';
    }).join('<span class="cs-sep">\\u203a</span>');

    var catLabel = flameCategoryLabel(lastFrame);
    var catHtml = catLabel ? '<span class="callstack-category" style="background:' + flameColor(lastFrame) + '22;color:' + flameColor(lastFrame) + ';">' + catLabel + '</span>' : '';

    return '<div class="callstack-row">' +
      '<div class="callstack-count">' + s.count.toLocaleString() + '</div>' +
      '<div class="callstack-bar-wrap"><div class="callstack-bar-fill" style="width:' + barPct + '%%;background:' + barColor + ';"></div><span class="callstack-pct-label">' + pct.toFixed(1) + '%%</span></div>' +
      '<div class="callstack-frames">' + formattedFrames + catHtml + '</div>' +
    '</div>';
  }).join('');

  if (sorted.length > showCount) {
    html += '<div class="callstack-show-more"><button onclick="showAllCallStacks()">Show all ' + sorted.length + ' paths</button></div>';
  }

  container.innerHTML = html || '<p class="no-data-msg">No call stacks to display.</p>';
}

function showAllCallStacks() {
  callStackShowAll = true;
  var container = document.getElementById('callStackContainer');
  var raw = container.getAttribute('data-stacks');
  if (raw) drawCallStack(JSON.parse(raw), callStackSortByCount);
}

function toggleCallStackSort() {
  callStackSortByCount = !callStackSortByCount;
  var container = document.getElementById('callStackContainer');
  var raw = container.getAttribute('data-stacks');
  if (raw) drawCallStack(JSON.parse(raw), callStackSortByCount);
}

function renderBasicProfile(run) {
  var m = (run.profile && run.profile.metrics) || {};
  document.getElementById('profilingCmdsCard').style.display = 'none';

  var grid1 = document.getElementById('profileGrid');
  grid1.innerHTML = [
    renderProfileMetric(m.IPC || 'N/A', 'IPC', '#3b82f6'),
    renderProfileMetric(m['cache-miss-rate'] ? m['cache-miss-rate'] + '%%' : 'N/A', 'Cache Miss Rate', '#eab308'),
    renderProfileMetric(m['L1-dcache-miss-rate'] ? m['L1-dcache-miss-rate'] + '%%' : 'N/A', 'L1 D-Cache Miss', '#f97316'),
    renderProfileMetric(m['branch-miss-rate'] ? m['branch-miss-rate'] + '%%' : 'N/A', 'Branch Mispredict', '#22c55e'),
    renderProfileMetric(m.instructions ? (m.instructions / 1e9).toFixed(1) + 'B' : 'N/A', 'Instructions', '#8b5cf6'),
  ].join('');

  document.getElementById('profileGrid2').innerHTML = '<div class="no-data-msg" style="grid-column:1/-1;">Run <code>./start.sh deepprofile</code> for extended CPU/GPU analysis.</div>';
  document.getElementById('hotspotContainer').innerHTML = '<div class="no-data-msg">No hotspot data. Run deep profiling for function-level CPU analysis.</div>';
  document.getElementById('profileGpuGrid').innerHTML = '';
  document.getElementById('gpuTraceContainer').innerHTML = '';

  if (!m.cycles) return;

  if (charts.counter) charts.counter.destroy();
  var counterCtx = document.getElementById('counterChart').getContext('2d');
  var counterKeys = ['cycles','instructions','cache-references','cache-misses','branch-instructions','branch-misses','L1-dcache-loads','L1-dcache-load-misses'];
  charts.counter = new Chart(counterCtx, {
    type: 'bar',
    data: {
      labels: counterKeys.map(function(k) { return k.replace('L1-dcache-','L1-').replace('-instructions','-instr').replace('-references','-refs'); }),
      datasets: [{ data: counterKeys.map(function(k) { return m[k] || 0; }), backgroundColor: '#3b82f688', borderColor: '#3b82f6', borderWidth: 1, borderRadius: 4 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { color: '#94a3b8', maxRotation: 45, font: { size: 10 } } },
        y: { grid: { color: '#1e293b' }, ticks: { color: '#64748b' }, type: 'logarithmic' }
      }
    }
  });

  if (charts.ipcTrend) charts.ipcTrend.destroy();
  var ipcCtx = document.getElementById('ipcTrendChart').getContext('2d');
  charts.ipcTrend = new Chart(ipcCtx, {
    type: 'line',
    data: {
      labels: DATA.map(function(r) { return r.run_id.substring(0, 10); }),
      datasets: [{
        label: 'IPC',
        data: DATA.map(function(r) { return (r.profile && r.profile.metrics && r.profile.metrics.IPC) || null; }),
        borderColor: '#8b5cf6', backgroundColor: 'rgba(139,92,246,0.1)',
        fill: true, tension: 0.3, pointRadius: 6, pointBackgroundColor: '#8b5cf6',
        pointBorderColor: '#1a2332', pointBorderWidth: 2, spanGaps: true
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { backgroundColor: '#1a2332', borderColor: '#2a3444', borderWidth: 1, cornerRadius: 8 } },
      scales: {
        x: { grid: { color: '#1e293b' }, ticks: { color: '#64748b' } },
        y: { grid: { color: '#1e293b' }, ticks: { color: '#64748b' }, beginAtZero: true }
      }
    }
  });
}

// ============ GPU ============
function renderGPU(run) {
  var gpu = run.gpu_info || '';
  document.getElementById('gpuRaw').textContent = gpu || 'No GPU data for this run. Select a run executed on a GPU node, or run with --gpu flag.';
  var lines = gpu.split('\\n');
  var temp = 'N/A', power = 'N/A', vram = 'N/A', usage = 'N/A';
  var hasGpuData = false;
  for (var i = 0; i < lines.length; i++) {
    var line = lines[i];
    if (/^0\\s+/.test(line)) {
      var parts = line.trim().split(/\\s+/);
      if (parts.length >= 14) {
        temp = parts[4]; power = parts[5]; vram = parts[13]; usage = parts[14] || '0';
        hasGpuData = true;
      }
    }
  }

  // Also check deep profiles for GPU data
  var profGpu = null;
  PROFILES.forEach(function(p) {
    var hw = (p.gpu || {}).hardware || {};
    if (hw.temperature_c != null) profGpu = hw;
  });

  if (!hasGpuData && profGpu) {
    temp = profGpu.temperature_c + '\\u00b0C';
    power = profGpu.power_w ? profGpu.power_w + 'W' : 'N/A';
    vram = profGpu.vram_used_mb ? (profGpu.vram_used_mb / 1024).toFixed(1) + 'GB' : 'N/A';
    usage = profGpu.utilization_pct != null ? profGpu.utilization_pct + '%%' : 'N/A';
    hasGpuData = true;
  }

  if (hasGpuData) {
    document.getElementById('gpuGrid').innerHTML =
      '<div class="gpu-stat"><div class="val" style="color:var(--green)">' + temp + '</div><div class="lbl">Temperature</div></div>' +
      '<div class="gpu-stat"><div class="val" style="color:var(--accent)">' + power + '</div><div class="lbl">Power (Avg)</div></div>' +
      '<div class="gpu-stat"><div class="val" style="color:var(--text3)">' + usage + '</div><div class="lbl">GPU Utilization</div></div>' +
      '<div class="gpu-stat"><div class="val" style="color:var(--accent2)">' + vram + '</div><div class="lbl">VRAM Used</div></div>';
  } else {
    var runType = run.run_type || '';
    var msg = runType === 'cpu_baseline' ? 'CPU-only baseline run \\u2014 no GPU data collected.' : 'No GPU data available. Run on a GPU node to collect metrics.';
    document.getElementById('gpuGrid').innerHTML =
      '<div style="grid-column:1/-1;text-align:center;color:var(--text3);padding:24px;">' +
        '<div style="font-size:2rem;margin-bottom:8px;">\\ud83d\\udcbb</div>' +
        '<div>' + msg + '</div>' +
        '<div style="margin-top:8px;font-size:0.8rem;color:var(--text-muted);">Profiles with GPU data: ' + PROFILES.filter(function(p) { return (p.gpu || {}).hardware && (p.gpu.hardware).temperature_c != null; }).length + '/' + PROFILES.length + '</div>' +
      '</div>';
  }
}

// ============ Allocation ============
function renderAllocation() {
  var cpuTotal = 30000;
  var gpuTotal = 30000;
  var cpuUsed = 0;
  var gpuUsed = 0;
  var history = [];

  // Estimate SU usage from deep profiles
  // GPU partition: each node-hour = 1 SU (MI250X), task-clock gives CPU-ms
  // Rough: task-clock(ms) * threads / 3600000 for node-hours, then scale
  PROFILES.forEach(function(p) {
    var cpu = (p.cpu || {}).counters || {};
    var taskClock = parseFloat(cpu['task-clock'] || '0');
    var threads = parseInt(p.threads) || 1;
    // task-clock is in ms of CPU time, convert to node-hours
    // Setonix GPU nodes have 64 cores, 1 GPU SU = 1 node-hour
    var wallSec = taskClock / 1000;  // Wall time approx (single-thread equivalent)
    var nodeHours = wallSec / 3600;
    // GPU profiles use gpu-dev partition
    var estSU = Math.max(nodeHours, 0.001);
    gpuUsed += estSU;
    history.push({
      id: p.profile_id, date: p.date || 'N/A', dataset: p.dataset || 'N/A',
      threads: threads, partition: 'gpu-dev', su: estSU
    });
  });

  // Also count pipeline runs (CPU partition, lighter)
  DATA.forEach(function(r) {
    var wallSec = r.summary.total_time || 0;
    var nodeHours = wallSec / 3600;
    cpuUsed += Math.max(nodeHours, 0.001);
  });

  var gpuUsedKSU = gpuUsed;
  var cpuUsedKSU = cpuUsed;
  var gpuPct = (gpuUsedKSU / gpuTotal * 100);
  var cpuPct = (cpuUsedKSU / cpuTotal * 100);

  document.getElementById('allocStats').innerHTML =
    '<div class="stat-card"><div class="label">CPU Runs</div><div class="value">' + DATA.length + '</div><div class="change">Pipeline executions</div></div>' +
    '<div class="stat-card"><div class="label">GPU Profiles</div><div class="value">' + PROFILES.length + '</div><div class="change">Deep profiling sessions</div></div>' +
    '<div class="stat-card"><div class="label">CPU Used</div><div class="value" style="color:var(--green)">' + cpuUsedKSU.toFixed(3) + ' kSU</div><div class="change">of ' + cpuTotal.toLocaleString() + ' kSU</div></div>' +
    '<div class="stat-card"><div class="label">GPU Used</div><div class="value" style="color:var(--accent)">' + gpuUsedKSU.toFixed(3) + ' kSU</div><div class="change">of ' + gpuTotal.toLocaleString() + ' kSU</div></div>';

  document.getElementById('allocGrid').innerHTML =
    '<div class="card"><div class="card-header"><h2>CPU Allocation (pawsey1351)</h2></div>' +
    '<div style="display:flex;justify-content:space-between;margin-top:8px;"><span>' + cpuUsedKSU.toFixed(3) + ' kSU used</span><span style="color:var(--text3)">' + cpuTotal.toLocaleString() + ' kSU total</span></div>' +
    '<div class="alloc-bar"><div class="fill" style="width:' + Math.max(cpuPct, 0.05) + '%%;background:var(--green);min-width:2px;"></div></div>' +
    '<div style="color:var(--green);font-size:0.85rem;margin-top:4px;">' + (cpuTotal - cpuUsedKSU).toFixed(1) + ' kSU remaining (' + (100 - cpuPct).toFixed(2) + '%%)</div></div>' +
    '<div class="card"><div class="card-header"><h2>GPU Allocation (pawsey1351-gpu)</h2></div>' +
    '<div style="display:flex;justify-content:space-between;margin-top:8px;"><span>' + gpuUsedKSU.toFixed(3) + ' kSU used</span><span style="color:var(--text3)">' + gpuTotal.toLocaleString() + ' kSU total</span></div>' +
    '<div class="alloc-bar"><div class="fill" style="width:' + Math.max(gpuPct, 0.05) + '%%;background:var(--accent);min-width:2px;"></div></div>' +
    '<div style="color:var(--green);font-size:0.85rem;margin-top:4px;">' + (gpuTotal - gpuUsedKSU).toFixed(1) + ' kSU remaining (' + (100 - gpuPct).toFixed(2) + '%%)</div></div>';

  // Usage history table
  var tbody = document.querySelector('#allocHistoryTable tbody');
  if (history.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" class="no-data-msg">No profiling runs yet — usage will be tracked automatically</td></tr>';
  } else {
    tbody.innerHTML = history.map(function(h) {
      return '<tr><td>' + h.id + '</td><td>' + h.date + '</td><td>' + h.dataset + '</td><td>' + h.threads + '</td><td>' + h.partition + '</td><td>' + h.su.toFixed(4) + '</td></tr>';
    }).join('');
  }
}

// ============ Environment ============
function renderEnvironment(run) {
  var tbody = document.querySelector('#envTable tbody');
  var env = run.env || {};
  var keys = Object.keys(env);
  tbody.innerHTML = keys.map(function(k) {
    return '<tr><td style="font-weight:600;width:200px;">' + k + '</td><td style="font-family:monospace;font-size:0.85rem;">' + env[k] + '</td></tr>';
  }).join('');
}

// ============ Init ============
loadData();''' % (data_json, profiles_json, generated_time)


def main():
    generate()


if __name__ == '__main__':
    main()
