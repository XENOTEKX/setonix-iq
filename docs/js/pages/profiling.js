// web/js/pages/profiling.js — deep dive for selected run: hotspots + callstack + flamegraph

import { store } from '../state.js?v=20260430153743';
import { loadRun } from '../data.js?v=20260430153743';
import { mountRunPicker } from '../components/run-picker.js?v=20260430153743';
import * as hotspotChart from '../charts/hotspot.js?v=20260430153743';
import * as callstack from '../charts/callstack.js?v=20260430153743';
import * as flamegraph from '../charts/flamegraph.js?v=20260430153743';
import { escHtml, fmtNum, fmtPercent } from '../utils.js?v=20260430153743';

const TMPL = `
  <div class="page-header"><div><h1>Profiling</h1>
    <div class="subtitle">Deep CPU counters, hotspots, call stacks, and flamegraph</div></div></div>

  <div class="run-picker-wrap" style="margin-bottom:22px;">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
      <span style="font-size:0.72rem; color:var(--text3); text-transform:uppercase; letter-spacing:0.1em; font-weight:700;">Selected run</span>
      <span style="font-size:0.68rem; color:var(--text-muted);">Only runs with profile data listed</span>
    </div>
    <div id="profRunPicker"></div>
  </div>

  <div class="stats-grid" id="profMetrics"></div>

  <div class="card" id="profMemCard" hidden>
    <div class="card-header"><h2>Memory &amp; context switches</h2><div class="actions"><span class="btn-sm">10s sampler · /proc/&lt;pid&gt;</span></div></div>
    <div id="profMem"></div>
  </div>
  <div class="card" id="profNumaCard" hidden>
    <div class="card-header"><h2>NUMA residency</h2><div class="actions"><span class="btn-sm">per-node MB · numastat</span></div></div>
    <div id="profNuma"></div>
  </div>
  <div class="card" id="profIoCard" hidden>
    <div class="card-header"><h2>I/O totals</h2><div class="actions"><span class="btn-sm">/proc/&lt;pid&gt;/io</span></div></div>
    <div id="profIo"></div>
  </div>
  <div class="card" id="profThreadCard" hidden>
    <div class="card-header"><h2>Per-thread CPU time</h2><div class="actions"><span class="btn-sm">utime + stime · clock ticks</span></div></div>
    <div id="profThreads"></div>
  </div>

  <div class="card">
    <div class="card-header"><h2>Top hotspots</h2><div class="actions"><span class="btn-sm">perf sampling · self time</span></div></div>
    <div id="profHotspots"></div>
  </div>
  <div class="card">
    <div class="card-header"><h2>Top call stacks</h2><div class="actions"><span class="btn-sm">folded · grouped by path</span></div></div>
    <div id="profCallstack"></div>
  </div>
  <div class="card">
    <div class="card-header"><h2>Flamegraph</h2><div class="actions"><span class="btn-sm">X = samples · Y = depth</span></div></div>
    <div id="profFlame"></div>
  </div>
`;

export async function mount(root) {
  root.innerHTML = TMPL;
  const idx = store.get('runsIndex') || [];
  const profileRuns = idx.filter((r) => r.has_hotspots || r.has_stacks);

  if (!profileRuns.length) {
    root.querySelector('#profRunPicker').innerHTML =
      '<div class="empty" style="padding:22px;">No runs with profile data indexed yet. Run <code>perf record</code> and include <code>hotspots</code> / <code>folded_stacks</code> in the log JSON.</div>';
    document.getElementById('profMetrics').innerHTML = '';
    document.getElementById('profHotspots').innerHTML = '';
    document.getElementById('profCallstack').innerHTML = '';
    document.getElementById('profFlame').innerHTML = '';
    return;
  }

  const sorted = [...profileRuns].sort((a, b) => {
    // hotspots+stacks first, then stacks only, then hotspots only
    const rank = (r) => (r.has_hotspots && r.has_stacks ? 2 : (r.has_stacks ? 1 : 0));
    return rank(b) - rank(a);
  });
  const first = sorted[0];

  mountRunPicker(document.getElementById('profRunPicker'), sorted, {
    selectedId: first?.run_id,
    onChange: async (r) => updateForRun(await loadRun(r.run_id)),
  });

  updateForRun(await loadRun(first.run_id));
}

function updateForRun(run) {
  if (!run) return;
  const m = run.profile?.metrics || {};
  const metrics = [
    ['IPC', fmtNum(m.IPC, 2)],
    ['FE-stall', fmtPercent(m['frontend-stall-rate'], 2)],
    ['BE-stall', fmtPercent(m['backend-stall-rate'], 2)],
    ['Cache miss', fmtPercent(m['cache-miss-rate'], 2)],
    ['Branch miss', fmtPercent(m['branch-miss-rate'], 3)],
    ['L1-D miss', fmtPercent(m['L1-dcache-miss-rate'], 2)],
    ['dTLB miss', fmtPercent(m['dTLB-miss-rate'], 2)],
    ['iTLB miss', fmtPercent(m['iTLB-miss-rate'], 2)],
  ].filter(([, v]) => v && v !== '—');
  document.getElementById('profMetrics').innerHTML = metrics
    .map(([k, v]) => `<div class="stat-card"><div class="label">${escHtml(k)}</div><div class="value">${escHtml(v)}</div></div>`)
    .join('');

  renderMemoryTimeseries(run);
  renderNuma(run);
  renderIo(run);
  renderPerThread(run);

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

function renderMemoryTimeseries(run) {
  const card = document.getElementById('profMemCard');
  const host = document.getElementById('profMem');
  const ts = run.profile?.memory_timeseries;
  const peak = run.profile?.peak_rss_kb;
  if (!ts?.length && !peak) { card.hidden = true; return; }
  card.hidden = false;
  let html = '';
  if (peak) html += `<div style="margin-bottom:10px;font-size:0.82rem;">Peak RSS: <b>${(peak/1024/1024).toFixed(2)} GB</b></div>`;
  if (ts?.length) {
    const maxRss = Math.max(...ts.map((p) => p.rss_kb || 0)) || 1;
    const w = 720, h = 140, pad = 22;
    const pts = ts.map((p, i) => {
      const x = pad + (i / (ts.length - 1 || 1)) * (w - 2 * pad);
      const y = h - pad - ((p.rss_kb || 0) / maxRss) * (h - 2 * pad);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');
    const tMax = ts[ts.length - 1]?.t_s || 0;
    html += `
      <svg viewBox="0 0 ${w} ${h}" style="width:100%;height:${h}px;" preserveAspectRatio="none">
        <polyline fill="rgba(106,170,255,0.15)" stroke="none"
          points="${pad},${h-pad} ${pts} ${w-pad},${h-pad}"/>
        <polyline fill="none" stroke="#6aaaff" stroke-width="1.6" points="${pts}"/>
        <text x="${pad}" y="${h-4}" font-size="10" fill="var(--text-muted)">0s</text>
        <text x="${w-pad}" y="${h-4}" font-size="10" fill="var(--text-muted)" text-anchor="end">${tMax.toFixed(0)}s</text>
        <text x="2" y="14" font-size="10" fill="var(--text-muted)">${(maxRss/1024/1024).toFixed(1)} GB</text>
      </svg>
      <div style="font-size:0.72rem;color:var(--text-muted);margin-top:6px;">
        ${ts.length} samples · final threads ${fmtNum(ts[ts.length-1]?.threads_now, 0)}
      </div>`;
  }
  host.innerHTML = html;
}

function renderNuma(run) {
  const card = document.getElementById('profNumaCard');
  const host = document.getElementById('profNuma');
  const n = run.profile?.numa;
  if (!n?.per_node_mb) { card.hidden = true; return; }
  card.hidden = false;
  const nodes = Object.entries(n.per_node_mb);
  const total = n.total_mb || nodes.reduce((a, [, v]) => a + (+v || 0), 0);
  const max = Math.max(...nodes.map(([, v]) => +v || 0)) || 1;
  const rows = nodes.map(([node, mb]) => {
    const pct = ((+mb || 0) / max) * 100;
    const shareOfTotal = total ? ((+mb || 0) / total) * 100 : 0;
    return `
      <div style="display:grid;grid-template-columns:70px 1fr 90px 60px;gap:8px;align-items:center;margin-bottom:4px;font-size:0.78rem;">
        <span>node ${escHtml(node)}</span>
        <div style="background:var(--surface-2);height:14px;border-radius:3px;overflow:hidden;">
          <div style="width:${pct.toFixed(1)}%;height:100%;background:#5fb864;"></div>
        </div>
        <span style="text-align:right;font-variant-numeric:tabular-nums;">${fmtNum(+mb, 1)} MB</span>
        <span style="text-align:right;color:var(--text-muted);">${shareOfTotal.toFixed(1)}%</span>
      </div>`;
  }).join('');
  host.innerHTML = rows + `<div style="margin-top:8px;font-size:0.76rem;color:var(--text-muted);">Total: <b>${fmtNum(total, 1)} MB</b> across ${nodes.length} nodes</div>`;
}

function renderIo(run) {
  const card = document.getElementById('profIoCard');
  const host = document.getElementById('profIo');
  const io = run.profile?.io;
  if (!io) { card.hidden = true; return; }
  card.hidden = false;
  const fmtBytes = (b) => {
    if (!b) return '0';
    const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
    let i = 0, x = b;
    while (x >= 1024 && i < units.length - 1) { x /= 1024; i++; }
    return `${x.toFixed(2)} ${units[i]}`;
  };
  host.innerHTML = `
    <div class="stats-grid">
      <div class="stat-card"><div class="label">Disk read</div><div class="value">${fmtBytes(io.read_bytes)}</div></div>
      <div class="stat-card"><div class="label">Disk write</div><div class="value">${fmtBytes(io.write_bytes)}</div></div>
      <div class="stat-card"><div class="label">rchar</div><div class="value">${fmtBytes(io.rchar)}</div></div>
      <div class="stat-card"><div class="label">wchar</div><div class="value">${fmtBytes(io.wchar)}</div></div>
      <div class="stat-card"><div class="label">read syscalls</div><div class="value">${fmtNum(io.syscr, 0)}</div></div>
      <div class="stat-card"><div class="label">write syscalls</div><div class="value">${fmtNum(io.syscw, 0)}</div></div>
    </div>`;
}

function renderPerThread(run) {
  const card = document.getElementById('profThreadCard');
  const host = document.getElementById('profThreads');
  const pt = run.profile?.per_thread;
  if (!pt?.length) { card.hidden = true; return; }
  card.hidden = false;
  const HZ = 100; // CLK_TCK
  const rows = [...pt].sort((a, b) => (b.utime + b.stime) - (a.utime + a.stime)).slice(0, 32);
  const max = Math.max(...rows.map((t) => (t.utime || 0) + (t.stime || 0))) || 1;
  const html = rows.map((t) => {
    const total = (t.utime || 0) + (t.stime || 0);
    const pct = (total / max) * 100;
    const uPct = total ? ((t.utime || 0) / total) * 100 : 0;
    return `
      <div style="display:grid;grid-template-columns:70px 1fr 90px;gap:8px;align-items:center;margin-bottom:3px;font-size:0.76rem;">
        <span>tid ${t.tid}</span>
        <div style="background:var(--surface-2);height:12px;border-radius:3px;overflow:hidden;display:flex;">
          <div style="width:${(pct*uPct/100).toFixed(1)}%;background:#6aaaff;"></div>
          <div style="width:${(pct*(1-uPct/100)).toFixed(1)}%;background:#f59e0b;"></div>
        </div>
        <span style="text-align:right;font-variant-numeric:tabular-nums;">${(total/HZ).toFixed(1)}s</span>
      </div>`;
  }).join('');
  host.innerHTML = `<div style="margin-bottom:8px;font-size:0.72rem;color:var(--text-muted);">showing top ${rows.length} of ${pt.length} threads · blue=user, orange=sys</div>${html}`;
}
