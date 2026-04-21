// web/js/charts/efficiency.js — parallel efficiency (speedup/threads) vs threads

import { hashColour } from '../utils.js';

export function render(canvas, runsIndex) {
  const existing = window.Chart?.getChart?.(canvas);
  if (existing) existing.destroy();
  const byDataset = new Map();
  for (const r of runsIndex) {
    if (!r.dataset_short || r.threads == null || r.efficiency == null) continue;
    if (!byDataset.has(r.dataset_short)) byDataset.set(r.dataset_short, []);
    byDataset.get(r.dataset_short).push({ x: Number(r.threads), y: r.efficiency });
  }
  const datasets = [];
  for (const [ds, pts] of byDataset) {
    pts.sort((a, b) => a.x - b.x);
    datasets.push({
      label: ds,
      data: pts,
      borderColor: hashColour(ds, 0.95),
      backgroundColor: hashColour(ds, 0.2),
      tension: 0.25,
      pointRadius: 4,
      borderWidth: 2,
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
        legend: { position: 'bottom', labels: { color: '#a0a7bd', font: { size: 10 } } },
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
