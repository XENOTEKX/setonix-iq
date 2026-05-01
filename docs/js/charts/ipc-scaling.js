// web/js/charts/ipc-scaling.js — IPC vs threads, one line per (dataset, platform).
// NOTE: Gadi runs currently do not populate profile.metrics (perf_stat.txt
// harvesting pending), so they will be absent from this view — that is expected
// and an empty-state hint is rendered when no IPC points exist.

import { platformColour, dimLegendHidden } from '../utils.js?v=20260501115130';

function platformOf(r) {
  return r.platform || (r.pbs_id ? 'gadi' : (r.slurm_id ? 'setonix' : 'unknown'));
}
function platformLabel(p) {
  return p === 'gadi' ? 'Gadi' : p === 'setonix' ? 'Setonix' : p;
}
function isPilot(name) {
  return typeof name === 'string' && /(_gadi_pilot|_setonix_pilot)\.fa$/.test(name);
}

export function render(canvas, runsIndex) {
  const existing = window.Chart?.getChart?.(canvas);
  if (existing) existing.destroy();
  const byKey = new Map();
  const byKeyNC = new Map();
  for (const r of runsIndex) {
    if (!r.dataset_short || r.threads == null || r.IPC == null) continue;
    if (isPilot(r.dataset_short)) continue;
    if (r.archived) continue;
    const plat = platformOf(r);
    // L1d miss rate is the canonical cross-platform memory metric (same
    // perf events on AMD Zen3 and Intel SPR — see CHANGELOG follow-up #18).
    // Stored as percent (0-100). MPKI is the secondary derived metric.
    const point = {
      x: Number(r.threads),
      y: r.IPC,
      l1MissPct: (typeof r.l1_dcache_miss_rate === 'number') ? r.l1_dcache_miss_rate : null,
      l1Mpki: (typeof r.l1d_mpki === 'number') ? r.l1d_mpki : null,
    };
    if (r.non_canonical) {
      const refLabel = r.non_canonical_label || 'ref';
      const key = `${platformLabel(plat)} · ${r.dataset_short} · ${refLabel}`;
      if (!byKeyNC.has(key)) byKeyNC.set(key, { plat, ds: r.dataset_short, points: [] });
      byKeyNC.get(key).points.push(point);
    } else {
      const key = `${platformLabel(plat)} · ${r.dataset_short}`;
      if (!byKey.has(key)) byKey.set(key, { plat, ds: r.dataset_short, points: [] });
      byKey.get(key).points.push(point);
    }
  }
  const datasets = [];
  const ordered = [...byKey.entries()].sort(([, a], [, b]) =>
    a.plat.localeCompare(b.plat) || a.ds.localeCompare(b.ds));
  for (const [label, { plat, ds, points }] of ordered) {
    points.sort((a, b) => a.x - b.x);
    datasets.push({
      label,
      data: points,
      borderColor: platformColour(plat, ds, 0.95),
      backgroundColor: platformColour(plat, ds, 0.25),
      borderDash: plat === 'gadi' ? [6, 4] : [],
      pointStyle: plat === 'gadi' ? 'triangle' : 'circle',
      tension: 0.25,
      pointRadius: 4,
      borderWidth: 2,
    });
  }
  const orderedNC = [...byKeyNC.entries()].sort(([, a], [, b]) =>
    a.plat.localeCompare(b.plat) || a.ds.localeCompare(b.ds));
  for (const [label, { plat, ds, points }] of orderedNC) {
    points.sort((a, b) => a.x - b.x);
    datasets.push({
      label,
      data: points,
      borderColor: platformColour(plat, ds, 0.45),
      backgroundColor: platformColour(plat, ds, 0.1),
      borderDash: [3, 5],
      pointStyle: 'crossRot',
      tension: 0.25,
      pointRadius: 3,
      borderWidth: 1.5,
      hidden: true,
    });
  }

  if (!datasets.length) {
    // Leave canvas blank but annotate so Gadi-only views don't look broken.
    const ctx = canvas.getContext?.('2d');
    if (ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#8b97ad';
      ctx.font = '12px -apple-system, system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('CPU counters not yet harvested for any run.',
        canvas.width / 2, canvas.height / 2);
    }
    return;
  }

  new Chart(canvas, {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      parsing: false,
      plugins: {
        legend: { position: 'bottom', labels: { color: '#a0a7bd', font: { size: 10 }, generateLabels: dimLegendHidden } },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const p = ctx.raw || {};
              const lines = [`${ctx.dataset.label}: IPC ${ctx.parsed.y.toFixed(3)} @ T=${ctx.parsed.x}`];
              if (p.l1MissPct != null) {
                // L1d-cache miss rate — canonical cross-platform metric
                // (identical perf events on AMD Zen3 and Intel SPR PMUs).
                lines.push(`  L1d miss: ${p.l1MissPct.toFixed(2)}%`);
              }
              if (p.l1Mpki != null) {
                lines.push(`  L1d-MPKI: ${p.l1Mpki.toFixed(1)}`);
              }
              return lines;
            },
          },
        },
      },
      scales: {
        x: {
          type: 'logarithmic',
          title: { display: true, text: 'Threads (log)', color: '#a0a7bd' },
          ticks: { color: '#a0a7bd' },
          grid: { color: 'rgba(160,167,189,0.08)' },
        },
        y: {
          title: { display: true, text: 'Instructions per cycle (IPC)', color: '#a0a7bd' },
          ticks: { color: '#a0a7bd' },
          grid: { color: 'rgba(160,167,189,0.08)' },
        },
      },
    },
  });
}
