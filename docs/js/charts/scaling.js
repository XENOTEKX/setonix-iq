// web/js/charts/scaling.js — wall-time vs threads,
// one series per (dataset, platform, build_variant) combo.
// Canonical series are solid/dashed by platform; non-canonical patch/variant
// series are shown as lighter named series (initially hidden).

import { platformColour, buildFamily } from '../utils.js?v=dd1c431ade92';

// Colour overrides for known MF2/patch families so they stand out.
const FAMILY_COLOURS = {
  'MF2 Full':             { h: 162, s: 75, l: 52 },  // teal-green
  'MF2 MF-only':          { h: 145, s: 55, l: 45 },  // muted green
  'MF2 Dispatch':         { h: 180, s: 70, l: 40 },  // cyan
  'R2 · NUMA patch':      { h: 270, s: 65, l: 58 },  // purple
  'R2 · MPI':             { h: 290, s: 60, l: 55 },  // violet
  'AVX-512 + R2':         { h: 195, s: 80, l: 50 },  // azure
  'FCA mf-iso (full)':    { h:  38, s: 90, l: 55 },  // amber/gold
  'FCA mf-iso (MF-only)': { h:  20, s: 75, l: 62 },  // coral-orange
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

// Format raw seconds as a human-readable duration string.
// < 60s   → "Xs"  (redundant to show, used as pass-through)
// < 3600s → "Xm Ys"
// < 86400s→ "Xh Ym"
// ≥ 86400s→ "Xd Yh Zm"
function fmtDuration(s) {
  if (s < 60) return `${s.toFixed(1)}s`;
  const d = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.round(s % 60);
  if (d > 0) return `${d}d ${h}h ${m}m`;
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m ${sec}s`;
}

// Per-family line/point style for non-canonical series.
const NC_STYLES = {
  'MF2 Full':             { borderDash: [],     pointRadius: 5, pointStyle: 'rectRot',  borderWidth: 2   },
  'FCA mf-iso (full)':    { borderDash: [],     pointRadius: 5, pointStyle: 'star',     borderWidth: 2   },
  'FCA mf-iso (MF-only)': { borderDash: [5, 3], pointRadius: 4, pointStyle: 'triangle', borderWidth: 1.5 },
};
function ncStyle(family) {
  return NC_STYLES[family] || { borderDash: [3, 5], pointRadius: 3, pointStyle: 'crossRot', borderWidth: 1.5 };
}

function renderExternalLegend(canvas, chart) {
  const wrapper = canvas.parentElement;
  if (!wrapper) return;
  wrapper.querySelector('.scaling-legend')?.remove();
  wrapper.style.display = 'flex';
  wrapper.style.flexDirection = 'column';
  canvas.style.flex = '1';
  canvas.style.minHeight = '0';
  const legendDiv = document.createElement('div');
  legendDiv.className = 'scaling-legend';
  legendDiv.style.cssText = 'max-height:110px;overflow-y:auto;display:flex;flex-wrap:wrap;gap:2px 10px;padding:6px 4px 2px;border-top:1px solid rgba(139,151,173,0.12);flex-shrink:0;';
  chart.data.datasets.forEach((ds, i) => {
    const isHidden = !chart.isDatasetVisible(i);
    const item = document.createElement('span');
    item.style.cssText = `display:inline-flex;align-items:center;gap:5px;cursor:pointer;font-size:10px;color:#8b97ad;padding:1px 3px;border-radius:3px;flex-shrink:0;opacity:${isHidden ? '0.35' : '1'};`;
    const swatch = document.createElement('span');
    const bc = typeof ds.borderColor === 'string' ? ds.borderColor : '#8b97ad';
    swatch.style.cssText = `display:inline-block;width:22px;height:2px;background:${bc};border-radius:1px;flex-shrink:0;`;
    const label = document.createElement('span');
    label.textContent = ds.label;
    label.style.cssText = 'max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;';
    if (isHidden) label.style.textDecoration = 'line-through';
    item.appendChild(swatch);
    item.appendChild(label);
    item.addEventListener('click', () => {
      const nowVisible = !chart.isDatasetVisible(i);
      chart.setDatasetVisibility(i, nowVisible);
      item.style.opacity = nowVisible ? '1' : '0.35';
      label.style.textDecoration = nowVisible ? '' : 'line-through';
      chart.update();
    });
    legendDiv.appendChild(item);
  });
  wrapper.appendChild(legendDiv);
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
      // Non-canonical: group by (platform, dataset, family, nc_label) — one line per patch
      // variant, but with family included so e.g. mf-iso (full) vs (MF-only) remain separate
      // even when they share the same nc_label.
      const refLabel = r.non_canonical_label || family || 'ref';
      const key = `${platformLabel(plat)} · ${ds} · ${family} · ${refLabel}`;
      if (!byKeyNC.has(key)) byKeyNC.set(key, { plat, ds, family, points: [] });
      byKeyNC.get(key).points.push({ x: Number(r.threads), y: r.wall_s });
    } else {
      // Canonical: group by (platform, dataset) — all canonical variants share one line.
      const key = `${platformLabel(plat)} · ${ds}`;
      if (!byKey.has(key)) byKey.set(key, { plat, ds, family, points: [] });
      byKey.get(key).points.push({ x: Number(r.threads), y: r.wall_s });
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
      ...ncStyle(family),
      tension: 0.2,
      hidden: !(family === 'FCA mf-iso (full)' || family === 'FCA mf-iso (MF-only)'),
    });
  }

  const instance = new Chart(canvas, {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      parsing: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const s = ctx.parsed.y;
              const human = fmtDuration(s);
              const timeStr = s < 60 ? `${s.toFixed(1)}s` : `${s.toFixed(1)}s (${human})`;
              return `${ctx.dataset.label}: ${timeStr} @ T=${ctx.parsed.x}`;
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
  renderExternalLegend(canvas, instance);
}
