// web/js/charts/scaling.js — wall-time vs threads,
// one series per (dataset, platform) combo — Setonix and Gadi regenerate
// the same named datasets with different dimensions, so they must not share
// a line.

import { platformColour } from '../utils.js?v=20260430114511';

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
  for (const r of runsIndex) {
    if (!r.dataset || r.threads == null || r.wall_s == null || !r.all_pass || r.wall_s <= 0) continue;
    if (isPilot(r.dataset_short) || isPilot(r.dataset)) continue;
    if (r.archived) continue;
    const plat = platformOf(r);
    const ds = r.dataset_short || r.dataset;
    // Platform goes first in the label so legend entries group by platform.
    const key = `${platformLabel(plat)} · ${ds}`;
    if (!byKey.has(key)) byKey.set(key, { plat, ds, points: [] });
    byKey.get(key).points.push({ x: Number(r.threads), y: r.wall_s });
  }

  const datasets = [];
  // Sort so Setonix series come first, then Gadi — matches colour bands.
  const ordered = [...byKey.entries()].sort(([, a], [, b]) => {
    if (a.plat !== b.plat) return a.plat.localeCompare(b.plat);
    return a.ds.localeCompare(b.ds);
  });
  for (const [label, { plat, ds, points }] of ordered) {
    points.sort((a, b) => a.x - b.x);
    datasets.push({
      label,
      data: points,
      borderColor: platformColour(plat, ds, 0.95),
      backgroundColor: platformColour(plat, ds, 0.25),
      borderDash: plat === 'gadi' ? [6, 4] : [],
      tension: 0.2,
      pointRadius: 4,
      pointStyle: plat === 'gadi' ? 'triangle' : 'circle',
    });
  }

  new Chart(canvas, {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      parsing: false,
      plugins: {
        legend: { position: 'bottom', labels: { color: '#8b97ad', font: { size: 10 } } },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)}s @ T=${ctx.parsed.x}`,
          },
        },
      },
      scales: {
        x: {
          type: 'logarithmic',
          title: { display: true, text: 'Threads (log)', color: '#8b97ad' },
          ticks: { color: '#8b97ad' },
          grid: { color: 'rgba(139,151,173,0.1)' },
        },
        y: {
          type: 'logarithmic',
          title: { display: true, text: 'Wall time (s, log)', color: '#8b97ad' },
          ticks: { color: '#8b97ad' },
          grid: { color: 'rgba(139,151,173,0.1)' },
        },
      },
    },
  });
}
