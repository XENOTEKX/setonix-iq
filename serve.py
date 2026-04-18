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
var charts = {};
var expandedRunId = null;

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
      return '<option value="' + i + '"' + (i === currentRunIdx ? ' selected' : '') + '>Run ' + r.run_id + '</option>';
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
  renderRunPills();
  renderOverview(run);
  renderTests(run);
  renderProfiling(run);
  renderGPU(run);
  renderAllocation();
  renderEnvironment(run);
  renderQuickCmds(run);
}

// ============ Overview ============
function renderRunPills() {
  var pills = document.getElementById('runPills');
  if (!pills) return;
  pills.innerHTML = DATA.map(function(r, i) {
    var isActive = i === currentRunIdx;
    var dotColor = r.summary.all_pass ? 'var(--green)' : 'var(--red)';
    var dateShort = (r.env.date || '').substring(0, 10);
    var timeShort = fmtTime(r.summary.total_time);
    return '<div class="run-pill' + (isActive ? ' active' : '') + '" onclick="switchRun(' + i + ')" title="' + r.run_id + '">' +
      '<span class="pill-dot" style="background:' + dotColor + '"></span>' +
      '<span>' + r.run_id.substring(0, 16) + '</span>' +
      '<span class="pill-time">' + timeShort + '</span>' +
    '</div>';
  }).join('');
}

function renderOverview(run) {
  var s = run.summary;
  var profile = (run.profile && run.profile.metrics) || {};
  var latestProf = PROFILES.length > 0 ? PROFILES[PROFILES.length - 1] : null;
  var lpCpu = latestProf ? (latestProf.cpu || {}) : {};
  var lpDerived = lpCpu.derived || {};
  var lpHotspots = lpCpu.hotspots || [];

  document.getElementById('overviewSubtitle').textContent =
    'Run ' + run.run_id + ' | ' + (run.env.date || 'N/A') + ' | ' + (run.env.hostname || 'N/A');

  var badge = document.getElementById('statusBadge');
  badge.textContent = s.all_pass ? 'ALL PASS' : s.fail + ' FAILED';
  badge.className = 'badge ' + (s.all_pass ? 'badge-pass' : 'badge-fail');

  var bestTime = Math.min.apply(null, DATA.map(function(r) { return r.summary.total_time; }));
  var ipcVal = lpDerived.IPC || profile.IPC || 'N/A';
  var feStall = lpDerived['frontend-stall-rate'];
  var cacheMiss = lpDerived['cache-miss-rate'];

  document.getElementById('statsGrid').innerHTML =
    '<div class="stat-card">' +
      '<div class="label">Tests</div>' +
      '<div class="value" style="color:' + (s.all_pass ? 'var(--green)' : 'var(--red)') + '">' + s.pass + '/' + (s.pass + s.fail) + '</div>' +
      '<div class="change">' + (s.all_pass ? 'All passing' : s.fail + ' failed') + '</div>' +
    '</div>' +
    '<div class="stat-card">' +
      '<div class="label">Pipeline Time</div>' +
      '<div class="value">' + fmtTime(s.total_time) + '</div>' +
      '<div class="change">' + run.timing.length + ' commands | Best: ' + fmtTime(bestTime) + '</div>' +
    '</div>' +
    '<div class="stat-card">' +
      '<div class="label">IPC</div>' +
      '<div class="value" style="color:var(--accent2)">' + ipcVal + '</div>' +
      '<div class="change">' + (feStall != null ? 'Frontend stalls: ' + feStall + '%%' : 'Instructions per cycle') + '</div>' +
    '</div>' +
    '<div class="stat-card">' +
      '<div class="label">Cache Miss</div>' +
      '<div class="value" style="color:' + (cacheMiss != null && cacheMiss > 5 ? 'var(--yellow)' : 'var(--green)') + '">' + (cacheMiss != null ? cacheMiss + '%%' : 'N/A') + '</div>' +
      '<div class="change">' + (lpDerived['L1-dcache-miss-rate'] != null ? 'L1-D: ' + lpDerived['L1-dcache-miss-rate'] + '%%' : 'Run deep profile for data') + '</div>' +
    '</div>' +
    '<div class="stat-card">' +
      '<div class="label">Profiles</div>' +
      '<div class="value">' + PROFILES.length + '</div>' +
      '<div class="change">' + DATA.length + ' pipeline run' + (DATA.length > 1 ? 's' : '') + '</div>' +
    '</div>';

  // Latest profile summary card
  var profCard = document.getElementById('latestProfileCard');
  if (latestProf && profCard) {
    profCard.style.display = '';
    var topFunc = lpHotspots.length > 0 ? lpHotspots[0] : null;
    var gpuHw = (latestProf.gpu || {}).hardware || {};
    var aln = latestProf.alignment || {};
    document.getElementById('latestProfileContent').innerHTML =
      '<div class="detail-grid" style="grid-template-columns:repeat(4,1fr);">' +
        '<div class="detail-kv"><div class="dk-label">Dataset</div><div class="dk-value">' + escHtml(latestProf.dataset || 'N/A') + '</div></div>' +
        '<div class="detail-kv"><div class="dk-label">Taxa \\u00d7 Sites</div><div class="dk-value" style="color:var(--accent2);font-weight:600;">' + (aln.taxa || '?') + ' \\u00d7 ' + (aln.sites ? aln.sites.toLocaleString() : '?') + '</div></div>' +
        '<div class="detail-kv"><div class="dk-label">Model</div><div class="dk-value">' + escHtml(aln.substitution_model || latestProf.model || 'N/A') + '</div></div>' +
        '<div class="detail-kv"><div class="dk-label">Threads</div><div class="dk-value">' + (latestProf.threads || 'N/A') + '</div></div>' +
        '<div class="detail-kv"><div class="dk-label">IPC</div><div class="dk-value" style="color:var(--accent2)">' + (lpDerived.IPC || 'N/A') + '</div></div>' +
        '<div class="detail-kv"><div class="dk-label">Frontend Stalls</div><div class="dk-value" style="color:' + (feStall > 10 ? 'var(--red)' : 'var(--green)') + '">' + (feStall != null ? feStall + '%%' : 'N/A') + '</div></div>' +
        '<div class="detail-kv"><div class="dk-label">Log-Likelihood</div><div class="dk-value" style="font-family:monospace;font-size:0.8rem;color:var(--green);">' + (aln.log_likelihood != null ? aln.log_likelihood.toLocaleString() : 'N/A') + '</div></div>' +
        '<div class="detail-kv"><div class="dk-label">Wall Time</div><div class="dk-value">' + (aln.wall_time_sec != null ? aln.wall_time_sec.toFixed(1) + 's' : 'N/A') + '</div></div>' +
        '<div class="detail-kv"><div class="dk-label">Top Hotspot</div><div class="dk-value" style="font-family:monospace;font-size:0.7rem;">' + (topFunc ? topFunc.percent.toFixed(1) + '%%  ' + escHtml(topFunc['function'].substring(0, 35)) : 'N/A') + '</div></div>' +
        '<div class="detail-kv"><div class="dk-label">Site Patterns</div><div class="dk-value">' + (aln.site_patterns ? aln.site_patterns.toLocaleString() : 'N/A') + '</div></div>' +
        '<div class="detail-kv"><div class="dk-label">GPU Temp</div><div class="dk-value">' + (gpuHw.temperature_c != null ? gpuHw.temperature_c + '\\u00b0C' : 'N/A') + '</div></div>' +
        '<div class="detail-kv"><div class="dk-label">Profile ID</div><div class="dk-value" style="font-size:0.75rem;">' + latestProf.profile_id + '</div></div>' +
      '</div>';
  } else if (profCard) {
    profCard.style.display = 'none';
  }

  renderLeaderboard(run);
  renderHotspotChart();
  renderMicroarchChart();
  renderScalingChart();
}

// ============ IQ-TREE Performance Metrics ============

// Known dataset sizes — extended as more benchmarks are added
var DATASET_INFO = {
  'turtle.fa': { taxa: 16, sites: 1998 },
  'medium_dna.fa': { taxa: 50, sites: 5000 },
  'medium_dna.phy': { taxa: 50, sites: 5000 },
  'large_dna.phy': { taxa: 100, sites: 10000 },
  'stress_dna.phy': { taxa: 200, sites: 20000 }
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
  var info = DATASET_INFO[ds] || { taxa: '?', sites: '?' };
  var blockKey = model.replace(/\\+F/g, '').replace(/\\+I/g, '');
  var block = MODEL_BLOCKS[blockKey] || MODEL_BLOCKS['default'];
  return { dataset: ds, taxa: info.taxa, sites: info.sites, threads: threads, model: model, block: block, gpu: gpu };
}

var leaderboardSort = 'time';

function sortLeaderboard(by) {
  leaderboardSort = by;
  document.getElementById('lbSortTime').className = 'btn' + (by === 'time' ? ' active' : '');
  document.getElementById('lbSortSpeedup').className = 'btn' + (by === 'speedup' ? ' active' : '');
  renderLeaderboard(DATA[currentRunIdx]);
}

function renderLeaderboard(run) {
  // Build entries from all commands across all runs + profiles
  var entries = [];

  // From run timing data
  DATA.forEach(function(r) {
    r.timing.forEach(function(t) {
      var p = parseCmd(t.command);
      // Find matching verify entry for log-likelihood
      var loglik = '\\u2014';
      r.verify.forEach(function(v) {
        if (t.command.indexOf(v.file.replace('.iqtree', '')) !== -1) loglik = v.reported;
      });
      // Calc speedup: find 1T baseline for same dataset+model
      entries.push({
        dataset: p.dataset, taxa: p.taxa, sites: p.sites, model: p.model,
        block: p.block, threads: p.threads, gpu: p.gpu ? 'Yes' : '\\u2014',
        time: t.time_s, loglik: loglik, runId: r.run_id, cmd: t.command
      });
    });
  });

  // Calculate speedups — find 1T baseline for each dataset
  var baselines = {};
  entries.forEach(function(e) {
    if (e.threads === 1) {
      var key = e.dataset + '|' + e.model;
      if (!baselines[key] || e.time < baselines[key]) baselines[key] = e.time;
    }
  });
  entries.forEach(function(e) {
    var key = e.dataset + '|' + e.model;
    var base = baselines[key];
    if (base && e.threads > 1) {
      e.speedup = (base / e.time).toFixed(2);
      e.efficiency = ((base / e.time / e.threads) * 100).toFixed(0);
    } else if (e.threads === 1) {
      e.speedup = '1.00';
      e.efficiency = '100';
    } else {
      e.speedup = '\\u2014';
      e.efficiency = '\\u2014';
    }
  });

  // Sort
  if (leaderboardSort === 'speedup') {
    entries.sort(function(a, b) { return parseFloat(b.speedup || 0) - parseFloat(a.speedup || 0); });
  } else {
    entries.sort(function(a, b) { return a.time - b.time; });
  }

  var tbody = document.querySelector('#leaderboardTable tbody');
  tbody.innerHTML = entries.map(function(e, i) {
    var rankClass = i === 0 ? 'gold' : i === 1 ? 'silver' : i === 2 ? 'bronze' : '';
    var spClass = parseFloat(e.speedup) >= 3 ? 'speedup-good' : parseFloat(e.speedup) >= 1.5 ? 'speedup-warn' : 'speedup-bad';
    var effPct = parseInt(e.efficiency) || 0;
    var effColor = effPct >= 80 ? 'var(--green)' : effPct >= 50 ? 'var(--yellow)' : 'var(--red)';
    return '<tr title="' + escAttr(e.cmd) + '">' +
      '<td class="lb-rank ' + rankClass + '">' + (i + 1) + '</td>' +
      '<td class="lb-dataset">' + escHtml(e.dataset) + '</td>' +
      '<td>' + e.taxa + '</td>' +
      '<td>' + e.sites + '</td>' +
      '<td class="lb-model">' + escHtml(e.model) + '</td>' +
      '<td>' + e.block + '</td>' +
      '<td>' + e.threads + '</td>' +
      '<td>' + e.gpu + '</td>' +
      '<td style="font-weight:700;">' + e.time.toFixed(3) + '</td>' +
      '<td><span class="speedup-badge ' + spClass + '">' + e.speedup + '\\u00d7</span></td>' +
      '<td>' + (effPct > 0 ? '<div class="eff-bar"><div class="fill" style="width:' + effPct + '%%;background:' + effColor + '"></div></div> ' + e.efficiency + '%%' : '\\u2014') + '</td>' +
      '<td>' + e.loglik + '</td>' +
    '</tr>';
  }).join('');
}

function renderHotspotChart() {
  if (charts.hotspot) charts.hotspot.destroy();
  var ctx = document.getElementById('hotspotChart');
  if (!ctx) return;

  // Gather hotspot data across profiles
  var labels = [];
  var kernelNames = ['DervSIMD', 'PartialLH', 'BufferSIMD', 'FromBuffer', 'libgomp', 'parsimony', 'Other'];
  var kernelColors = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#6b7280', '#06b6d4', '#3b82f6'];
  var datasets = kernelNames.map(function(name, idx) {
    return { label: name, backgroundColor: kernelColors[idx], data: [] };
  });

  var sources = PROFILES.length > 0 ? PROFILES : [];
  // Also add run profile if available
  DATA.forEach(function(r) {
    if (r.profile && r.profile.metrics && r.profile.metrics.IPC) {
      // No hotspot detail in basic profiles, skip
    }
  });

  if (sources.length === 0) {
    ctx.parentElement.innerHTML = '<div class="no-data-msg">No hotspot data. Run deep profiling on Setonix.</div>';
    return;
  }

  sources.forEach(function(p) {
    var hs = (p.cpu || {}).hotspots || [];
    var dsName = (p.dataset || '?').split('/').pop().replace('.fa', '').replace('.phy', '') + ' T' + (p.threads || 1);
    labels.push(dsName);
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
    // Fill remaining as Other
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
  var colors = ['#3b82f6', '#8b5cf6', '#22c55e', '#f97316', '#ef4444'];

  // Gather from profiles
  var sources = PROFILES.length > 0 ? PROFILES : [];
  if (sources.length === 0 && profile && profile.IPC) {
    // No radar possible from basic metrics only
    ctx.parentElement.innerHTML = '<div class="no-data-msg">Run deep profiling for microarchitecture analysis.</div>';
    return;
  }
  if (sources.length === 0) {
    ctx.parentElement.innerHTML = '<div class="no-data-msg">No profile data. Run deep profiling on Setonix.</div>';
    return;
  }

  sources.forEach(function(p, i) {
    var d = (p.cpu || {}).derived || {};
    var dsName = (p.dataset || '?').split('/').pop().replace('.fa', '').replace('.phy', '') + ' T' + (p.threads || 1);
    // Normalize to 0-100 for radar: IPC scaled to max 4.0 theoretical
    var ipcNorm = Math.min((d.IPC || 0) / 4.0 * 100, 100);
    var cacheHit = 100 - (d['cache-miss-rate'] || 0);
    var branchAcc = 100 - (d['branch-miss-rate'] || 0);
    var feEff = 100 - (d['frontend-stall-rate'] || 0);
    var l1dHit = 100 - (d['L1-dcache-miss-rate'] || 0);
    var dtlbHit = 100 - (d['dTLB-miss-rate'] || 0);
    radarDatasets.push({
      label: dsName,
      data: [ipcNorm, cacheHit, branchAcc, feEff, l1dHit, dtlbHit],
      borderColor: colors[i %% colors.length],
      backgroundColor: colors[i %% colors.length] + '22',
      pointBackgroundColor: colors[i %% colors.length],
      pointRadius: 4, borderWidth: 2
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
  var passing = DATA.filter(function(r) { return r.summary.all_pass; }).length;
  var failing = total - passing;
  var bestTime = Math.min.apply(null, DATA.map(function(r) { return r.summary.total_time; }));
  var avgTime = (DATA.reduce(function(a, r) { return a + r.summary.total_time; }, 0) / total).toFixed(1);
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
      '<div class="label">Passing</div>' +
      '<div class="value" style="color:var(--green)">' + passing + '</div>' +
      '<div class="change">' + (failing > 0 ? failing + ' with failures' : 'All green') + '</div>' +
    '</div>' +
    '<div class="stat-card">' +
      '<div class="label">Best Time</div>' +
      '<div class="value" style="color:var(--accent)">' + fmtTime(bestTime) + '</div>' +
      '<div class="change">Avg: ' + avgTime + 's</div>' +
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
        r.run_id, r.env.hostname || '', r.env.date || '', r.env.cpu || '', r.env.gcc || '', r.env.rocm || ''
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

    return '<div class="run-row ' + (isExpanded ? 'active' : '') + '" id="run-' + r.run_id + '">' +
      '<div class="run-row-summary" onclick="toggleRunDetail(\\'' + r.run_id + '\\', ' + item.idx + ')">' +
        '<div class="rank ' + rankClass + '">#' + (rank + 1) + '</div>' +
        '<div class="run-info">' +
          '<div class="run-id">' + r.run_id + '</div>' +
          '<div class="run-meta">' + (r.env.date || 'N/A') + ' &middot; ' + (r.env.hostname || 'N/A') + ' &middot; ' + (r.env.cores || '?') + ' cores &middot; ' + cpuShort + '</div>' +
        '</div>' +
        '<div class="run-time">' + fmtTime(s.total_time) + '</div>' +
        '<div class="run-tests"><span class="badge ' + (s.all_pass ? 'badge-pass' : 'badge-fail') + '">' + s.pass + '/' + (s.pass + s.fail) + '</span></div>' +
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
  badge.textContent = s.pass + '/' + (s.pass + s.fail) + ' passing';
  badge.className = 'badge ' + (s.all_pass ? 'badge-pass' : 'badge-fail');

  var tbody = document.querySelector('#testsTable tbody');
  tbody.innerHTML = run.verify.map(function(v) {
    return '<tr data-status="' + v.status + '">' +
      '<td class="status-icon">' + (v.status === 'pass' ? '\\u2705' : '\\u274c') + '</td>' +
      '<td style="font-family:monospace;font-size:0.8rem">' + v.file + '</td>' +
      '<td>' + v.expected + '</td><td>' + v.reported + '</td>' +
      '<td style="color:' + (v.diff < 0.01 ? 'var(--green)' : 'var(--yellow)') + '">' + v.diff + '</td></tr>';
  }).join('');

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
  var prof = getActiveProfile();
  var sel = document.getElementById('profRunSelector');

  if (PROFILES.length > 0) {
    sel.innerHTML = PROFILES.map(function(p, i) {
      return '<option value="' + i + '"' + (i === currentProfileIdx ? ' selected' : '') + '>' + p.profile_id + ' (' + p.dataset + ', T' + p.threads + ')</option>';
    }).join('');
    sel.onchange = function() { switchProfile(this.value); };
  } else {
    sel.innerHTML = DATA.map(function(r, i) {
      return '<option value="' + i + '"' + (i === currentRunIdx ? ' selected' : '') + '>Run ' + r.run_id + '</option>';
    }).join('');
    sel.onchange = function() { switchRun(this.value); };
  }

  if (prof) {
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

  // Render flamegraph and call stack from folded stacks
  renderFlamegraph(cpu.folded_stacks || []);
  renderCallStack(cpu.folded_stacks || []);
}

// ============ Flamegraph ============
var flameData = null;
var flameZoomStack = [];

function buildFlameTree(stacks) {
  var root = {name: 'all', value: 0, children: {}};
  stacks.forEach(function(s) {
    var frames = s.stack.split(';');
    var node = root;
    root.value += s.count;
    frames.forEach(function(frame) {
      if (!node.children[frame]) {
        node.children[frame] = {name: frame, value: 0, children: {}};
      }
      node.children[frame].value += s.count;
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
    arr.sort(function(a, b) { return b.value - a.value; });
    return arr;
  }
  root.childArr = toArray(root);
  return root;
}

function flameColor(name) {
  // Color by function type
  if (name === '[unknown]') return '#4a5568';
  if (name.indexOf('GOMP') >= 0 || name.indexOf('pthread') >= 0 || name.indexOf('start_thread') >= 0) return '#3b82f6';
  if (name.indexOf('Phylo') >= 0 || name.indexOf('Tree') >= 0) return '#f97316';
  if (name.indexOf('Likelihood') >= 0 || name.indexOf('LH') >= 0 || name.indexOf('SIMD') >= 0) return '#ef4444';
  if (name.indexOf('Parsimony') >= 0 || name.indexOf('parsimony') >= 0) return '#a855f7';
  if (name.indexOf('alloc') >= 0 || name.indexOf('malloc') >= 0 || name.indexOf('free') >= 0) return '#eab308';
  if (name.indexOf('void') >= 0) return '#6366f1';
  if (name.indexOf('double') >= 0) return '#ec4899';
  // Hash-based color
  var h = 0;
  for (var i = 0; i < name.length; i++) h = ((h << 5) - h + name.charCodeAt(i)) | 0;
  var hue = 20 + (Math.abs(h) %% 40);
  var sat = 50 + (Math.abs(h >> 8) %% 30);
  var lit = 45 + (Math.abs(h >> 16) %% 15);
  return 'hsl(' + hue + ',' + sat + '%%,' + lit + '%%)';
}

function renderFlamegraph(stacks) {
  var container = document.getElementById('flamegraphContainer');
  var resetBtn = document.getElementById('flamegraphZoomReset');
  if (!stacks || stacks.length === 0) {
    container.innerHTML = '<p class="no-data-msg">No call stack data available. Run deep profiling with perf record to generate.</p>';
    if (resetBtn) resetBtn.style.display = 'none';
    return;
  }
  flameData = buildFlameTree(stacks);
  flameZoomStack = [];
  if (resetBtn) resetBtn.style.display = 'none';
  drawFlame(flameData);
}

function drawFlame(root) {
  var container = document.getElementById('flamegraphContainer');
  var totalSamples = root.value;
  container.innerHTML = '';

  function renderLevel(nodes, total) {
    if (nodes.length === 0) return;
    var row = document.createElement('div');
    row.className = 'flame-row';
    var rendered = 0;
    nodes.forEach(function(node) {
      var pct = (node.value / total * 100);
      if (pct < 0.3) return; // Skip tiny frames
      var el = document.createElement('div');
      el.className = 'flame-frame';
      el.style.width = pct + '%%';
      el.style.background = flameColor(node.name);
      el.textContent = pct > 3 ? node.name.substring(0, 50) : '';
      el.title = '';
      el.setAttribute('data-name', node.name);
      el.setAttribute('data-count', node.value);
      el.setAttribute('data-pct', pct.toFixed(2));
      el.addEventListener('mouseenter', function(e) { showFlameTooltip(e, node.name, node.value, totalSamples); });
      el.addEventListener('mouseleave', hideFlameTooltip);
      if (node.childArr && node.childArr.length > 0) {
        el.addEventListener('click', function() { zoomFlame(node); });
      }
      row.appendChild(el);
      rendered++;
    });
    if (rendered > 0) container.insertBefore(row, container.firstChild);
    // Recurse into children of all nodes at this level
    var nextLevel = [];
    nodes.forEach(function(node) {
      if (node.childArr) {
        node.childArr.forEach(function(c) { nextLevel.push({node: c, parentTotal: total}); });
      }
    });
    if (nextLevel.length > 0) {
      var childNodes = nextLevel.map(function(x) { return x.node; });
      renderLevel(childNodes, total);
    }
  }

  renderLevel(root.childArr || [], root.value);

  // Add root bar at bottom
  var rootRow = document.createElement('div');
  rootRow.className = 'flame-row';
  var rootEl = document.createElement('div');
  rootEl.className = 'flame-frame';
  rootEl.style.width = '100%%';
  rootEl.style.background = '#475569';
  rootEl.textContent = root.name + ' (' + totalSamples + ' samples)';
  rootRow.appendChild(rootEl);
  container.appendChild(rootRow);
}

function zoomFlame(node) {
  flameZoomStack.push(flameData);
  drawFlame(node);
  document.getElementById('flamegraphZoomReset').style.display = '';
}

function resetFlameZoom() {
  if (flameData) drawFlame(flameData);
  flameZoomStack = [];
  document.getElementById('flamegraphZoomReset').style.display = 'none';
}

var flameTooltipEl = null;
function showFlameTooltip(e, name, count, total) {
  if (!flameTooltipEl) {
    flameTooltipEl = document.createElement('div');
    flameTooltipEl.className = 'flame-tooltip';
    document.body.appendChild(flameTooltipEl);
  }
  var pct = (count / total * 100).toFixed(2);
  flameTooltipEl.innerHTML = '<div style="font-weight:600;margin-bottom:4px;word-break:break-all;">' + escHtml(name) + '</div>' +
    '<div>' + count + ' samples (' + pct + '%%)</div>';
  flameTooltipEl.style.display = '';
  flameTooltipEl.style.left = Math.min(e.clientX + 10, window.innerWidth - 420) + 'px';
  flameTooltipEl.style.top = (e.clientY - 60) + 'px';
}
function hideFlameTooltip() {
  if (flameTooltipEl) flameTooltipEl.style.display = 'none';
}

// ============ Call Stack ============
var callStackSortByCount = true;

function renderCallStack(stacks) {
  var container = document.getElementById('callStackContainer');
  if (!stacks || stacks.length === 0) {
    container.innerHTML = '<p class="no-data-msg">No call stack data available. Run deep profiling with perf record to generate.</p>';
    return;
  }
  // Store for re-sorting
  container.setAttribute('data-stacks', JSON.stringify(stacks));
  drawCallStack(stacks, true);
}

function drawCallStack(stacks, byCount) {
  var container = document.getElementById('callStackContainer');
  var sorted = stacks.slice().sort(function(a, b) {
    return byCount ? b.count - a.count : a.stack.localeCompare(b.stack);
  });
  var maxCount = sorted.length > 0 ? sorted[0].count : 1;
  var top = sorted.slice(0, 40); // Show top 40

  var html = top.map(function(s) {
    var frames = s.stack.split(';');
    var pct = (s.count / maxCount * 100);
    var lastFrame = frames[frames.length - 1];
    var formattedFrames = frames.map(function(f, i) {
      if (i === frames.length - 1) return '<span style="color:var(--accent2);font-weight:600;">' + escHtml(f) + '</span>';
      return escHtml(f);
    }).join(' <span style="color:var(--text-muted);">\\u2192</span> ');

    return '<div class="callstack-row">' +
      '<div class="callstack-count">' + s.count + '</div>' +
      '<div style="flex:0 0 100px;"><div class="callstack-bar" style="width:' + pct + '%%;background:' + flameColor(lastFrame) + ';"></div></div>' +
      '<div class="callstack-frames">' + formattedFrames + '</div>' +
    '</div>';
  }).join('');

  container.innerHTML = html || '<p class="no-data-msg">No call stacks to display.</p>';
}

function toggleCallStackSort() {
  callStackSortByCount = !callStackSortByCount;
  var container = document.getElementById('callStackContainer');
  var raw = container.getAttribute('data-stacks');
  if (raw) {
    drawCallStack(JSON.parse(raw), callStackSortByCount);
  }
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
  document.getElementById('gpuRaw').textContent = gpu || 'No GPU data available';
  var lines = gpu.split('\\n');
  var temp = 'N/A', power = 'N/A', vram = 'N/A', usage = 'N/A';
  for (var i = 0; i < lines.length; i++) {
    var line = lines[i];
    if (/^0\\s+/.test(line)) {
      var parts = line.trim().split(/\\s+/);
      if (parts.length >= 14) {
        temp = parts[4]; power = parts[5]; vram = parts[13]; usage = parts[14] || '0';
      }
    }
  }
  document.getElementById('gpuGrid').innerHTML =
    '<div class="gpu-stat"><div class="val" style="color:var(--green)">' + temp + '</div><div class="lbl">Temperature</div></div>' +
    '<div class="gpu-stat"><div class="val" style="color:var(--accent)">' + power + '</div><div class="lbl">Power (Avg)</div></div>' +
    '<div class="gpu-stat"><div class="val" style="color:var(--text3)">' + usage + '</div><div class="lbl">GPU Utilization</div></div>' +
    '<div class="gpu-stat"><div class="val" style="color:var(--accent2)">' + vram + '</div><div class="lbl">VRAM Used</div></div>';
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
