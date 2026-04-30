// web/js/charts/performance-matrix.js — bubble plot:
//   x = threads (log), y = wall time (log), bubble size = sites,
//   colour = (dataset, platform). Gadi is drawn with triangle markers so the
//   two platforms can be distinguished at a glance.

import { platformColour } from '../utils.js?v=20260430153021';

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

  // Compute min/max sites for bubble-size scaling
  const sites = runsIndex.map(r => r.sites || 0).filter(Boolean);
  const minS = Math.min(...sites, 1);
  const maxS = Math.max(...sites, 1);
  const sizeFor = (s) => {
    if (!s) return 6;
    const t = (s - minS) / Math.max(1, maxS - minS);
    return 6 + t * 18; // 6..24
  };

  const byKey = new Map();
  const byKeyNC = new Map();
  for (const r of runsIndex) {
    if (!r.dataset_short || r.threads == null || r.wall_s == null || !r.all_pass || r.wall_s <= 0) continue;
    if (isPilot(r.dataset_short)) continue;
    if (r.archived) continue;
    const plat = platformOf(r);
    if (r.non_canonical) {
      const refLabel = r.non_canonical_label || 'ref';
      const key = `${platformLabel(plat)} · ${r.dataset_short} · ${refLabel}`;
      if (!byKeyNC.has(key)) byKeyNC.set(key, { plat, ds: r.dataset_short, points: [] });
      byKeyNC.get(key).points.push({
        x: Number(r.threads),
        y: r.wall_s,
        r: sizeFor(r.sites),
        run_id: r.run_id,
        taxa: r.taxa,
        sites: r.sites,
        size_mb: r.size_mb,
        platform: plat,
      });
    } else {
      const key = `${platformLabel(plat)} · ${r.dataset_short}`;
      if (!byKey.has(key)) byKey.set(key, { plat, ds: r.dataset_short, points: [] });
      byKey.get(key).points.push({
        x: Number(r.threads),
        y: r.wall_s,
        r: sizeFor(r.sites),
        run_id: r.run_id,
        taxa: r.taxa,
        sites: r.sites,
        size_mb: r.size_mb,
        platform: plat,
      });
    }
  }

  const datasets = [];
  const ordered = [...byKey.entries()].sort(([, a], [, b]) =>
    a.plat.localeCompare(b.plat) || a.ds.localeCompare(b.ds));
  for (const [label, { plat, ds, points }] of ordered) {
    datasets.push({
      label,
      data: points,
      backgroundColor: platformColour(plat, ds, 0.55),
      borderColor: platformColour(plat, ds, 0.95),
      borderWidth: 1.5,
      pointStyle: plat === 'gadi' ? 'triangle' : 'circle',
    });
  }
  const orderedNC = [...byKeyNC.entries()].sort(([, a], [, b]) =>
    a.plat.localeCompare(b.plat) || a.ds.localeCompare(b.ds));
  for (const [label, { plat, ds, points }] of orderedNC) {
    datasets.push({
      label,
      data: points,
      backgroundColor: platformColour(plat, ds, 0.2),
      borderColor: platformColour(plat, ds, 0.45),
      borderWidth: 1,
      pointStyle: 'crossRot',
    });
  }

  new Chart(canvas, {
    type: 'bubble',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'bottom', labels: { color: '#a0a7bd', font: { size: 10 } } },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const d = ctx.raw;
              return [
                `${ctx.dataset.label}`,
                `  run: ${d.run_id}`,
                `  threads: ${d.x}`,
                `  wall: ${d.y.toFixed(1)}s`,
                `  sites: ${d.sites || '—'} · taxa: ${d.taxa || '—'}`,
                `  size: ${d.size_mb ? d.size_mb.toFixed(2) + ' MB' : '—'}`,
              ];
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
          type: 'logarithmic',
          title: { display: true, text: 'Wall time (s, log) — bubble size = sites', color: '#a0a7bd' },
          ticks: { color: '#a0a7bd' },
          grid: { color: 'rgba(160,167,189,0.08)' },
        },
      },
    },
  });
}
