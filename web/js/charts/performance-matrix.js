// web/js/charts/performance-matrix.js — bubble plot:
//   x = threads (log), y = wall time (log), bubble size = sites, colour = dataset

import { hashColour } from '../utils.js';

let chart;

export function render(canvas, runsIndex) {
  if (chart) chart.destroy();

  // Compute min/max sites for bubble-size scaling
  const sites = runsIndex.map(r => r.sites || 0).filter(Boolean);
  const minS = Math.min(...sites, 1);
  const maxS = Math.max(...sites, 1);
  const sizeFor = (s) => {
    if (!s) return 6;
    const t = (s - minS) / Math.max(1, maxS - minS);
    return 6 + t * 18; // 6..24
  };

  const byDataset = new Map();
  for (const r of runsIndex) {
    if (!r.dataset_short || r.threads == null || r.wall_s == null) continue;
    if (!byDataset.has(r.dataset_short)) byDataset.set(r.dataset_short, []);
    byDataset.get(r.dataset_short).push({
      x: Number(r.threads),
      y: r.wall_s,
      r: sizeFor(r.sites),
      run_id: r.run_id,
      taxa: r.taxa,
      sites: r.sites,
      size_mb: r.size_mb,
    });
  }

  const datasets = [];
  for (const [ds, pts] of byDataset) {
    datasets.push({
      label: ds,
      data: pts,
      backgroundColor: hashColour(ds, 0.55),
      borderColor: hashColour(ds, 0.95),
      borderWidth: 1.5,
    });
  }

  chart = new Chart(canvas, {
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
