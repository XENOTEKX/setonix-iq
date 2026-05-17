// web/js/charts/scaling.js — wall-time vs threads,
// one series per (dataset, platform, build_variant) combo.
// Canonical series are solid/dashed by platform; non-canonical patch/variant
// series are shown as lighter named series (initially hidden).

import { platformColour, buildFamily, dimLegendHidden } from '../utils.js?v=06ea76a79376';

// Colour overrides for known MF2/patch families so they stand out.
const FAMILY_COLOURS = {
  'MF2 Full':        { h: 162, s: 75, l: 52 },  // teal-green
  'MF2 MF-only':     { h: 145, s: 55, l: 45 },  // muted green
  'MF2 Dispatch':    { h: 180, s: 70, l: 40 },  // cyan
  'R2 · NUMA patch': { h: 270, s: 65, l: 58 },  // purple
  'R2 · MPI':        { h: 290, s: 60, l: 55 },  // violet
  'AVX-512 + R2':    { h: 195, s: 80, l: 50 },  // azure
};

function familyColour(family, alpha = 1) {
  const c = FAMILY_COLOURS[family];
  if (c) return `hsla(${c.h}, ${c.s}%, ${c.l}%, ${alpha})`;
  return null; // fall back to platformColour
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
  const byKeyNC = new Map(); // non_canonical / patch-variant series
  for (const r of runsIndex) {
    if (!r.dataset || r.threads == null || r.wall_s == null || !r.all_pass || r.wall_s <= 0) continue;
    if (isPilot(r.dataset_short) || isPilot(r.dataset)) continue;
    if (r.archived) continue;
    const plat = platformOf(r);
    const ds = r.dataset_short || r.dataset;
    const family = buildFamily(r);

    if (r.non_canonical) {
      // Non-canonical: group by (platform, dataset, nc_label) — one line per patch variant.
      const refLabel = r.non_canonical_label || family || 'ref';
      const key = `${platformLabel(plat)} · ${ds} · ${refLabel}`;
      if (!byKeyNC.has(key)) byKeyNC.set(key, { plat, ds, family, points: [] });
      byKeyNC.get(key).points.push({ x: Number(r.threads), y: r.wall_s, n: r.n_mpi_ranks || null, t: r.threads_per_node || null });
    } else {
      // Canonical: group by (platform, dataset) — all canonical variants share one line.
      const key = `${platformLabel(plat)} · ${ds}`;
      if (!byKey.has(key)) byKey.set(key, { plat, ds, family, points: [] });
      byKey.get(key).points.push({ x: Number(r.threads), y: r.wall_s, n: r.n_mpi_ranks || null, t: r.threads_per_node || null });
    }
  }

  const datasets = [];
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

  // Non-canonical / patch-variant series — shown as named lighter lines, hidden by default.
  // MF2 / R2 / AVX-512 patch families get their own tinted colour.
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
      tension: 0.2,
      pointRadius: family === 'MF2 Full' ? 5 : 3,
      pointStyle: family === 'MF2 Full' ? 'rectRot' : 'crossRot',
      borderWidth: family === 'MF2 Full' ? 2 : 1.5,
      hidden: true,
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
        legend: { position: 'bottom', labels: { color: '#8b97ad', font: { size: 10 }, generateLabels: dimLegendHidden } },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const r = ctx.raw;
              const tLabel = (r.n && r.t) ? `${r.n} nodes \xd7 ${r.t}T` : `T=${ctx.parsed.x}`;
              return `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)}s @ ${tLabel}`;
            },
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
