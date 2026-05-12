// web/js/charts/efficiency.js — parallel efficiency (speedup/threads) vs threads,
// grouped per (dataset, platform) so the Gadi and Setonix curves stay distinct.
// Non-canonical / patch-variant series get their own named series using buildFamily colours.

import { platformColour, buildFamily, dimLegendHidden } from '../utils.js';

// Colour overrides for known MF2/patch families.
const FAMILY_COLOURS = {
  'MF2 Full':        { h: 162, s: 75, l: 52 },
  'MF2 MF-only':     { h: 145, s: 55, l: 45 },
  'MF2 Dispatch':    { h: 180, s: 70, l: 40 },
  'R2 · NUMA patch': { h: 270, s: 65, l: 58 },
  'R2 · MPI':        { h: 290, s: 60, l: 55 },
  'AVX-512 + R2':    { h: 195, s: 80, l: 50 },
};
function familyColour(family, alpha = 1) {
  const c = FAMILY_COLOURS[family];
  if (c) return `hsla(${c.h}, ${c.s}%, ${c.l}%, ${alpha})`;
  return null;
}

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
    if (!r.dataset_short || r.threads == null || r.efficiency == null) continue;
    if (isPilot(r.dataset_short)) continue;
    if (r.archived) continue;
    const plat = platformOf(r);
    if (r.non_canonical) {
      const family = buildFamily(r);
      const refLabel = r.non_canonical_label || family || 'ref';
      const key = `${platformLabel(plat)} · ${r.dataset_short} · ${refLabel}`;
      if (!byKeyNC.has(key)) byKeyNC.set(key, { plat, ds: r.dataset_short, family, points: [] });
      byKeyNC.get(key).points.push({ x: Number(r.threads), y: r.efficiency });
    } else {
      const key = `${platformLabel(plat)} · ${r.dataset_short}`;
      if (!byKey.has(key)) byKey.set(key, { plat, ds: r.dataset_short, points: [] });
      byKey.get(key).points.push({ x: Number(r.threads), y: r.efficiency });
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
  for (const [label, { plat, ds, family, points }] of orderedNC) {
    points.sort((a, b) => a.x - b.x);
    const fColour = familyColour(family);
    datasets.push({
      label,
      data: points,
      borderColor: fColour ? fColour.replace('1)', '0.85)') : platformColour(plat, ds, 0.45),
      backgroundColor: fColour ? fColour.replace('1)', '0.15)') : platformColour(plat, ds, 0.1),
      borderDash: family === 'MF2 Full' ? [] : [3, 5],
      pointStyle: family === 'MF2 Full' ? 'rectRot' : 'crossRot',
      tension: 0.25,
      pointRadius: family === 'MF2 Full' ? 5 : 3,
      borderWidth: family === 'MF2 Full' ? 2 : 1.5,
      hidden: true,
    });
  }
  // Ideal efficiency = 1.0 reference
  const maxT = Math.max(...runsIndex.map(r => r.threads || 0), 1);
  datasets.push({
    label: 'Ideal',
    data: [{ x: 1, y: 1 }, { x: maxT, y: 1 }],
    borderColor: 'rgba(160,167,189,0.4)',
    borderDash: [4, 4],
    pointRadius: 0,
    borderWidth: 1.5,
  });

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
            label: (ctx) => `${ctx.dataset.label}: ${(ctx.parsed.y * 100).toFixed(1)}% @ T=${ctx.parsed.x}`,
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
          min: 0,
          max: 1.1,
          title: { display: true, text: 'Parallel efficiency (speedup / threads)', color: '#a0a7bd' },
          ticks: { color: '#a0a7bd', callback: (v) => `${(v * 100).toFixed(0)}%` },
          grid: { color: 'rgba(160,167,189,0.08)' },
        },
      },
    },
  });
}
