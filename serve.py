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

    # Remove flamegraph img that references local file
    output_html = output_html.replace(
        '<img src="assets/flamegraph.svg" alt="CPU Flamegraph" style="width:100%;min-width:800px;" onerror="this.parentElement.innerHTML=\'<p style=color:var(--text3)>No flamegraph available yet. Run profiling first.</p>\'">',
        '<p style="color:var(--text3)">Flamegraph not available in static view. Run profiling on Setonix to generate.</p>'
    )

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
var timingChartType = 'bar';
var expandedRunId = null;

// ============ Navigation ============
document.querySelectorAll('.sidebar nav a').forEach(function(a) {
  a.addEventListener('click', function(e) {
    e.preventDefault();
    var page = a.dataset.page;
    document.querySelectorAll('.sidebar nav a').forEach(function(x) { x.classList.remove('active'); });
    a.classList.add('active');
    document.querySelectorAll('.page').forEach(function(p) { p.classList.remove('active'); });
    document.getElementById('page-' + page).classList.add('active');
  });
});

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
  document.getElementById('serverDot').style.background = 'var(--green)';
  document.getElementById('serverStatus').textContent = 'Static (GitHub Pages)';
  document.getElementById('lastUpdate').textContent = 'Generated: %s';
}

function populateAllSelectors() {
  var selectors = ['runSelector', 'testRunSelector', 'profRunSelector', 'gpuRunSelector', 'envRunSelector'];
  selectors.forEach(function(id) {
    var sel = document.getElementById(id);
    if (!sel) return;
    sel.innerHTML = DATA.map(function(r, i) {
      return '<option value="' + i + '"' + (i === currentRunIdx ? ' selected' : '') + '>Run ' + r.run_id + '</option>';
    }).join('');
  });
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
  renderOverview(run);
  renderTests(run);
  renderProfiling(run);
  renderGPU(run);
  renderAllocation();
  renderEnvironment(run);
  renderQuickCmds(run);
}

// ============ Overview ============
function renderOverview(run) {
  var s = run.summary;
  var profile = (run.profile && run.profile.metrics) || {};

  document.getElementById('overviewSubtitle').textContent =
    'Run ' + run.run_id + ' | ' + (run.env.date || 'N/A') + ' | ' + (run.env.hostname || 'N/A');

  var badge = document.getElementById('statusBadge');
  badge.textContent = s.all_pass ? 'ALL PASS' : s.fail + ' FAILED';
  badge.className = 'badge ' + (s.all_pass ? 'badge-pass' : 'badge-fail');

  var bestTime = Math.min.apply(null, DATA.map(function(r) { return r.summary.total_time; }));
  var avgTime = (DATA.reduce(function(a, r) { return a + r.summary.total_time; }, 0) / DATA.length).toFixed(1);
  var allPass = DATA.every(function(r) { return r.summary.all_pass; });

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
      '<div class="value">' + (profile.IPC || 'N/A') + '</div>' +
      '<div class="change">Instructions per cycle</div>' +
    '</div>' +
    '<div class="stat-card">' +
      '<div class="label">Total Runs</div>' +
      '<div class="value">' + DATA.length + '</div>' +
      '<div class="change">' + (allPass ? 'All green' : 'Some failures') + ' | Avg: ' + avgTime + 's</div>' +
    '</div>';

  renderTimingChart(run);
  renderTrendChart();
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

// ============ Charts ============
function renderTimingChart(run) {
  if (charts.timing) charts.timing.destroy();
  var ctx = document.getElementById('timingChart').getContext('2d');
  var colors = ['#3b82f6','#8b5cf6','#06b6d4','#22c55e','#eab308','#f97316','#ef4444','#ec4899','#6366f1','#14b8a6'];
  charts.timing = new Chart(ctx, {
    type: timingChartType,
    data: {
      labels: run.timing.map(function(t) { return t.command.substring(0, 50); }),
      datasets: [{
        label: 'Time (s)',
        data: run.timing.map(function(t) { return t.time_s; }),
        backgroundColor: run.timing.map(function(_, i) { return colors[i %% colors.length] + '99'; }),
        borderColor: run.timing.map(function(_, i) { return colors[i %% colors.length]; }),
        borderWidth: 1, borderRadius: 4
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      indexAxis: timingChartType === 'bar' ? 'y' : undefined,
      plugins: {
        legend: { display: false },
        tooltip: { backgroundColor: '#1a2332', borderColor: '#2a3444', borderWidth: 1, titleColor: '#e2e8f0', bodyColor: '#94a3b8', cornerRadius: 8, padding: 12 }
      },
      scales: {
        x: { grid: { color: '#1e293b' }, ticks: { color: '#64748b' } },
        y: { grid: { display: false }, ticks: { color: '#94a3b8', font: { size: 10 } } }
      }
    }
  });
}

function renderTrendChart() {
  if (charts.trend) charts.trend.destroy();
  var ctx = document.getElementById('trendChart').getContext('2d');
  charts.trend = new Chart(ctx, {
    type: 'line',
    data: {
      labels: DATA.map(function(r) { return r.run_id.substring(0, 10); }),
      datasets: [{
        label: 'Total Time (s)',
        data: DATA.map(function(r) { return r.summary.total_time; }),
        borderColor: '#22c55e', backgroundColor: 'rgba(34,197,94,0.1)',
        fill: true, tension: 0.3, pointRadius: 6, pointBackgroundColor: '#22c55e',
        pointBorderColor: '#1a2332', pointBorderWidth: 2
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

function toggleChartType() {
  timingChartType = timingChartType === 'bar' ? 'doughnut' : 'bar';
  renderTimingChart(DATA[currentRunIdx]);
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
    renderDeepProfile(prof);
  } else {
    renderBasicProfile(run);
  }
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
  document.getElementById('allocGrid').innerHTML =
    '<div class="card"><div class="card-header"><h2>CPU Allocation (pawsey1351)</h2></div>' +
    '<div style="display:flex;justify-content:space-between;margin-top:8px;"><span>0 kSU used</span><span style="color:var(--text3)">30,000 kSU total</span></div>' +
    '<div class="alloc-bar"><div class="fill" style="width:0%%;background:var(--green)"></div></div>' +
    '<div style="color:var(--green);font-size:0.85rem;margin-top:4px;">30,000 kSU remaining (100%%)</div></div>' +
    '<div class="card"><div class="card-header"><h2>GPU Allocation (pawsey1351-gpu)</h2></div>' +
    '<div style="display:flex;justify-content:space-between;margin-top:8px;"><span>21 kSU used</span><span style="color:var(--text3)">30,000 kSU total</span></div>' +
    '<div class="alloc-bar"><div class="fill" style="width:0.07%%;background:var(--accent);min-width:2px;"></div></div>' +
    '<div style="color:var(--green);font-size:0.85rem;margin-top:4px;">29,979 kSU remaining (99.9%%)</div></div>';
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
