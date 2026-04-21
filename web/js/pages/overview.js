// web/js/pages/overview.js

import { store } from '../state.js';
import { loadRun } from '../data.js';
import * as runSelector from '../components/run-selector.js';
import { bindCopyButtons } from '../components/copy-button.js';
import * as hotspotChart from '../charts/hotspot.js';
import * as microarch from '../charts/microarch.js';
import * as scaling from '../charts/scaling.js';
import * as callstack from '../charts/callstack.js';
import { escHtml, fmtTime, fmtNum } from '../utils.js';

const TMPL = `
  <div class="page-header">
    <div>
      <h1>Pipeline Overview</h1>
      <div class="subtitle" id="ovSubtitle">Loading…</div>
    </div>
    <span class="badge badge-info" id="ovBadge">—</span>
  </div>
  <div class="run-selector-bar" id="ovRunSelector"></div>

  <div class="stats-grid" id="ovStats"></div>

  <div class="card">
    <div class="card-header">
      <h2>IQ-TREE Configuration</h2>
      <div class="actions">
        <button class="copy-btn" data-copy="#ovConfigText">Copy config</button>
      </div>
    </div>
    <div class="card-body" id="ovConfig"></div>
    <pre id="ovConfigText" class="sr-only"></pre>
  </div>

  <div class="charts-row">
    <div class="card">
      <div class="card-header"><h2>Hotspot Breakdown</h2></div>
      <div id="ovHotspots"></div>
    </div>
    <div class="card">
      <div class="card-header">
        <h2>Microarchitecture Profile (all runs)</h2>
        <div class="actions"><span class="btn" id="ovMicroarchCount"></span></div>
      </div>
      <div class="chart-wrapper"><canvas id="ovMicroarchCanvas"></canvas></div>
    </div>
  </div>

  <div class="card">
    <div class="card-header"><h2>Thread Scaling (all datasets)</h2></div>
    <div class="chart-wrapper"><canvas id="ovScalingCanvas"></canvas></div>
  </div>

  <div class="card">
    <div class="card-header">
      <h2>Call Stack (top paths)</h2>
      <div class="actions"><span class="btn-sm" id="ovStacksMeta">—</span></div>
    </div>
    <div id="ovCallstack"></div>
  </div>

  <div class="card">
    <div class="card-header">
      <h2>Commands</h2>
      <div class="actions">
        <button class="copy-btn" data-copy="#ovCmdsText">Copy all</button>
      </div>
    </div>
    <div class="card-body" id="ovCmds"></div>
    <pre id="ovCmdsText" class="sr-only"></pre>
  </div>
`;

export async function mount(root) {
  root.innerHTML = TMPL;

  // Paint cross-run charts first (don't depend on a single run)
  renderStats();
  scaling.render(document.getElementById('ovScalingCanvas'), store.get('runsIndex'));

  // All-runs microarch
  const runsIndex = store.get('runsIndex');
  const withMetrics = runsIndex.filter((r) => r.IPC != null);
  const microarchEl = document.getElementById('ovMicroarchCount');
  if (microarchEl) microarchEl.textContent = `${withMetrics.length} runs`;

  // Load all runs that have metrics so we can plot microarch for all of them
  const runsFull = await Promise.all(
    withMetrics.slice(0, 20).map((r) => loadRun(r.run_id))
  );
  microarch.render(document.getElementById('ovMicroarchCanvas'), runsFull);

  // Bind selector → per-run sections
  runSelector.render(document.getElementById('ovRunSelector'), (run) => {
    updateRunSections(run);
  });

  bindCopyButtons(root);
}

function renderStats() {
  const idx = store.get('runsIndex');
  const total = idx.length;
  const allPass = idx.every((r) => r.all_pass);
  const fastest = idx.reduce((a, b) => (b.wall_s < a.wall_s ? b : a), idx[0] || { wall_s: 0 });
  const bestIPC = idx.reduce((a, b) => (Number(b.IPC || 0) > Number(a.IPC || 0) ? b : a), idx[0] || {});

  // Speedup vs 1T of same dataset
  let bestSpeedup = 0;
  const byDS = new Map();
  for (const r of idx) {
    if (!r.dataset) continue;
    if (!byDS.has(r.dataset)) byDS.set(r.dataset, {});
    byDS.get(r.dataset)[String(r.threads)] = r.wall_s;
  }
  for (const m of byDS.values()) {
    if (!m['1']) continue;
    for (const k in m) {
      if (k === '1') continue;
      const s = m['1'] / m[k];
      if (s > bestSpeedup) bestSpeedup = s;
    }
  }

  document.getElementById('ovStats').innerHTML = `
    <div class="stat-card">
      <div class="label">Total runs</div>
      <div class="value">${total}</div>
      <div class="change">${allPass ? 'All tests passing' : 'Some failures'}</div>
    </div>
    <div class="stat-card">
      <div class="label">Fastest run</div>
      <div class="value">${fmtTime(fastest.wall_s)}</div>
      <div class="change">${escHtml(fastest.label || '—')}</div>
    </div>
    <div class="stat-card">
      <div class="label">Best speedup</div>
      <div class="value">${bestSpeedup ? bestSpeedup.toFixed(2) + '×' : '—'}</div>
      <div class="change">vs 1-thread baseline</div>
    </div>
    <div class="stat-card">
      <div class="label">Peak IPC</div>
      <div class="value">${bestIPC.IPC ? bestIPC.IPC.toFixed(2) : '—'}</div>
      <div class="change">${escHtml(bestIPC.label || '—')}</div>
    </div>
  `;
}

function updateRunSections(run) {
  if (!run) return;

  document.getElementById('ovSubtitle').textContent =
    `Run ${run.run_id} · ${run.env?.date || 'n/a'} · ${run.env?.hostname || 'n/a'}`;

  const badge = document.getElementById('ovBadge');
  const s = run.summary || {};
  if (s.fail === 0) {
    badge.className = 'badge badge-pass';
    badge.textContent = s.pass ? `ALL PASS · ${s.pass}` : 'OK';
  } else {
    badge.className = 'badge badge-fail';
    badge.textContent = `${s.fail} FAILED`;
  }

  renderConfig(run);
  hotspotChart.render(document.getElementById('ovHotspots'), run.profile?.hotspots);
  callstack.render(document.getElementById('ovCallstack'), run.profile?.folded_stacks);

  const stacksMeta = document.getElementById('ovStacksMeta');
  if (stacksMeta) {
    const n = run.profile?.folded_stacks?.length || 0;
    stacksMeta.textContent = `${n.toLocaleString()} unique stacks`;
  }

  renderCmds(run);
}

function renderConfig(run) {
  const p = run.profile || {};
  const m = p.metrics || {};
  const env = run.env || {};
  const h = run.hints || {};

  const items = (pairs) => pairs
    .filter(([, v]) => v != null && v !== '')
    .map(([k, v]) => `
      <div class="config-item"><span class="ci-label">${escHtml(k)}</span><span class="ci-value">${escHtml(String(v))}</span></div>
    `).join('');

  const alignment = items([
    ['Dataset', p.dataset || h.dataset],
    ['Threads', p.threads ?? h.threads],
    ['Run type', run.run_type],
  ]);

  const model = items([
    ['Model', h.model],
    ['Wall', fmtTime(run.summary?.total_time)],
    ['Commands', (run.timing || []).length],
    ['Verify pass', run.summary?.pass],
    ['Verify fail', run.summary?.fail],
  ]);

  const sys = items([
    ['Hostname', env.hostname],
    ['CPU', env.cpu],
    ['Cores', env.cores],
    ['GCC', env.gcc],
    ['ROCm', env.rocm],
    ['IPC', m.IPC],
    ['FE-stall %', m['frontend-stall-rate']],
  ]);

  document.getElementById('ovConfig').innerHTML = `
    <div class="config-grid">
      <div class="config-section"><h3>Alignment</h3><div class="config-items">${alignment}</div></div>
      <div class="config-section"><h3>Model &amp; Results</h3><div class="config-items">${model}</div></div>
      <div class="config-section"><h3>System</h3><div class="config-items">${sys}</div></div>
    </div>
    ${run.description ? `<div class="config-cmd" style="margin-top:12px;">${escHtml(run.description)}</div>` : ''}
  `;

  // Plain-text version for copy button
  const toText = (label, pairs) =>
    `# ${label}\n` + pairs.filter(([, v]) => v != null && v !== '').map(([k, v]) => `${k}: ${v}`).join('\n');
  document.getElementById('ovConfigText').textContent = [
    `# Run ${run.run_id}`,
    toText('Alignment', [['Dataset', p.dataset || h.dataset], ['Threads', p.threads ?? h.threads], ['Run type', run.run_type]]),
    toText('Model & Results', [['Model', h.model], ['Wall', fmtTime(run.summary?.total_time)], ['Commands', (run.timing || []).length], ['Pass', run.summary?.pass], ['Fail', run.summary?.fail]]),
    toText('System', [['Host', env.hostname], ['CPU', env.cpu], ['Cores', env.cores], ['GCC', env.gcc], ['ROCm', env.rocm], ['IPC', m.IPC]]),
  ].join('\n\n');
}

function renderCmds(run) {
  const timing = run.timing || [];
  document.getElementById('ovCmds').innerHTML = timing.length
    ? timing.map((t, i) => `
      <div class="cmd-block">
        <span class="cmd-num">${i + 1}</span>
        <span class="cmd-text">${escHtml(t.command)}</span>
        <span class="cmd-time">${fmtTime(t.time_s)}</span>
      </div>
    `).join('')
    : '<div class="empty">No commands recorded.</div>';
  document.getElementById('ovCmdsText').textContent =
    '#!/usr/bin/env bash\nset -euo pipefail\n\n' + timing.map((t) => t.command).join('\n');
}
