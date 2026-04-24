// web/js/charts/ipc-scaling.js — IPC vs threads, one line per (dataset, platform).
// NOTE: Gadi runs currently do not populate profile.metrics (perf_stat.txt
// harvesting pending), so they will be absent from this view — that is expected
// and an empty-state hint is rendered when no IPC points exist.

import { hashColour } from '../utils.js';

function platformOf(r) {
  return r.platform || (r.pbs_id ? 'gadi' : (r.slurm_id ? 'setonix' : 'unknown'));
}

export function render(canvas, runsIndex) {
  const existing = window.Chart?.getChart?.(canvas);
  if (existing) existing.destroy();
  const byKey = new Map();
  for (const r of runsIndex) {
    if (!r.dataset_short || r.threads == null || r.IPC == null) continue;
    const plat = platformOf(r);
    const key = `${r.dataset_short} · ${plat}`;
    if (!byKey.has(key)) byKey.set(key, { plat, points: [] });
    byKey.get(key).points.push({ x: Number(r.threads), y: r.IPC });
  }
  const datasets = [];
  for (const [label, { plat, points }] of byKey) {
    points.sort((a, b) => a.x - b.x);
    datasets.push({
      label,
      data: points,
      borderColor: hashColour(label, 0.95),
      backgroundColor: hashColour(label, 0.2),
      borderDash: plat === 'gadi' ? [6, 4] : [],
      pointStyle: plat === 'gadi' ? 'triangle' : 'circle',
      tension: 0.25,
      pointRadius: 4,
      borderWidth: 2,
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
