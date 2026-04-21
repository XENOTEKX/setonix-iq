// web/js/charts/ipc-scaling.js — IPC vs threads, one line per dataset

import { hashColour } from '../utils.js';

let chart;

export function render(canvas, runsIndex) {
  if (chart) chart.destroy();
  const byDataset = new Map();
  for (const r of runsIndex) {
    if (!r.dataset_short || r.threads == null || r.IPC == null) continue;
    if (!byDataset.has(r.dataset_short)) byDataset.set(r.dataset_short, []);
    byDataset.get(r.dataset_short).push({ x: Number(r.threads), y: r.IPC });
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

  chart = new Chart(canvas, {
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
            label: (ctx) => `${ctx.dataset.label}: IPC ${ctx.parsed.y.toFixed(3)} @ T=${ctx.parsed.x}`,
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
