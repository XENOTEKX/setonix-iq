// web/js/charts/scaling.js — wall-time vs threads, one series per dataset

import { hashColour } from '../utils.js';

export function render(canvas, runsIndex) {
  const existing = window.Chart?.getChart?.(canvas);
  if (existing) existing.destroy();

  const byDataset = new Map();
  for (const r of runsIndex) {
    if (!r.dataset || r.threads == null || r.wall_s == null) continue;
    if (!byDataset.has(r.dataset)) byDataset.set(r.dataset, []);
    byDataset.get(r.dataset).push({ x: Number(r.threads), y: r.wall_s });
  }

  const datasets = [];
  for (const [ds, pts] of byDataset) {
    pts.sort((a, b) => a.x - b.x);
    datasets.push({
      label: ds,
      data: pts,
      borderColor: hashColour(ds, 0.9),
      backgroundColor: hashColour(ds, 0.2),
      tension: 0.2,
      pointRadius: 4,
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
