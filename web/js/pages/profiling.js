// web/js/pages/profiling.js — deep dive for selected run: hotspots + callstack + flamegraph

import * as runSelector from '../components/run-selector.js';
import * as hotspotChart from '../charts/hotspot.js';
import * as callstack from '../charts/callstack.js';
import * as flamegraph from '../charts/flamegraph.js';
import { escHtml, fmtNum, fmtPercent } from '../utils.js';

const TMPL = `
  <div class="page-header"><div><h1>Profiling</h1>
    <div class="subtitle">Deep CPU counters, hotspots, call stacks, and flamegraph</div></div></div>
  <div class="run-selector-bar" id="profRunSelector"></div>
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

export function mount(root) {
  root.innerHTML = TMPL;
  runSelector.render(document.getElementById('profRunSelector'), (run) => {
    updateForRun(run);
  });
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

  hotspotChart.render(document.getElementById('profHotspots'), run.profile?.hotspots, { limit: 15 });
  callstack.render(document.getElementById('profCallstack'), run.profile?.folded_stacks, { limit: 15 });
  flamegraph.render(document.getElementById('profFlame'), run.profile?.folded_stacks);
}
