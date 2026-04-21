// web/js/pages/profiling.js — deep dive for selected run: hotspots + callstack + flamegraph

import { store } from '../state.js';
import { loadRun } from '../data.js';
import { mountRunPicker } from '../components/run-picker.js';
import * as hotspotChart from '../charts/hotspot.js';
import * as callstack from '../charts/callstack.js';
import * as flamegraph from '../charts/flamegraph.js';
import { escHtml, fmtNum, fmtPercent } from '../utils.js';

const TMPL = `
  <div class="page-header"><div><h1>Profiling</h1>
    <div class="subtitle">Deep CPU counters, hotspots, call stacks, and flamegraph</div></div></div>

  <div class="run-picker-wrap" style="margin-bottom:22px;">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
      <span style="font-size:0.72rem; color:var(--text3); text-transform:uppercase; letter-spacing:0.1em; font-weight:700;">Selected run</span>
      <span style="font-size:0.68rem; color:var(--text-muted);">Only runs with profile data shown first</span>
    </div>
    <div id="profRunPicker"></div>
  </div>

  <div class="stats-grid" id="profMetrics"></div>
  <div class="charts-row">
    <div class="card">
      <div class="card-header"><h2>Top hotspots</h2></div>
      <div id="profHotspots"></div>
    </div>
    <div class="card">
      <div class="card-header"><h2>Top call stacks</h2></div>
      <div id="profCallstack"></div>
    </div>
  </div>
  <div class="card">
    <div class="card-header"><h2>Flamegraph</h2></div>
    <div id="profFlame"></div>
  </div>
`;

export async function mount(root) {
  root.innerHTML = TMPL;
  const idx = store.get('runsIndex') || [];
  if (!idx.length) {
    root.querySelector('#profRunPicker').innerHTML = '<div class="empty">No runs indexed.</div>';
    return;
  }
  // Runs with hotspots first
  const sorted = [...idx].sort((a, b) => (Number(!!b.has_hotspots) - Number(!!a.has_hotspots)));
  const first = sorted[0];

  mountRunPicker(document.getElementById('profRunPicker'), sorted, {
    selectedId: first?.run_id,
    onChange: async (r) => {
      const full = await loadRun(r.run_id);
      updateForRun(full);
    },
  });

  const full = await loadRun(first.run_id);
  updateForRun(full);
}

function updateForRun(run) {
  if (!run) return;
  const m = run.profile?.metrics || {};
  const metrics = [
    ['IPC', fmtNum(m.IPC, 2)],
    ['FE-stall', fmtPercent(m['frontend-stall-rate'], 2)],
    ['Cache miss', fmtPercent(m['cache-miss-rate'], 2)],
    ['Branch miss', fmtPercent(m['branch-miss-rate'], 3)],
    ['L1-D miss', fmtPercent(m['L1-dcache-miss-rate'], 2)],
  ];
  document.getElementById('profMetrics').innerHTML = metrics
    .map(([k, v]) => `<div class="stat-card"><div class="label">${escHtml(k)}</div><div class="value">${escHtml(v)}</div></div>`)
    .join('');

  const hotspots = run.profile?.hotspots;
  const stacks = run.profile?.folded_stacks;

  const hs = document.getElementById('profHotspots');
  if (hotspots?.length) hotspotChart.render(hs, hotspots, { limit: 15 });
  else hs.innerHTML = '<div class="empty" style="padding:32px 20px; text-align:center;">No perf hotspot data for this run.<br><span style="color:var(--text-muted);font-size:0.76rem;">Pick a run with profile data from the dropdown above.</span></div>';

  const cs = document.getElementById('profCallstack');
  if (stacks?.length) callstack.render(cs, stacks, { limit: 15 });
  else cs.innerHTML = '<div class="empty" style="padding:32px 20px; text-align:center;">No call stacks recorded.</div>';

  const fl = document.getElementById('profFlame');
  if (stacks?.length) flamegraph.render(fl, stacks);
  else fl.innerHTML = '<div class="empty" style="padding:32px 20px; text-align:center;">Flamegraph unavailable — folded stacks missing.</div>';
}
