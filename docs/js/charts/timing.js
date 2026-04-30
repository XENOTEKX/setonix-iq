// web/js/charts/timing.js — per-command timing bars

import { escHtml, fmtTime } from '../utils.js?v=20260430160708';

let chart;

export function render(canvas, timing) {
  if (chart) chart.destroy();
  if (!timing || !timing.length) {
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
    return;
  }

  chart = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: timing.map((_, i) => `#${i + 1}`),
      datasets: [{
        label: 'Wall time (s)',
        data: timing.map((t) => t.time_s),
        backgroundColor: 'rgba(79,143,255,0.6)',
        borderColor: 'rgba(79,143,255,1)',
        borderWidth: 1,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            afterLabel: (ctx) => escHtml(timing[ctx.dataIndex].command || ''),
            label: (ctx) => `${fmtTime(ctx.parsed.y)} wall`,
          },
        },
      },
      scales: {
        x: { ticks: { color: '#8b97ad' }, grid: { color: 'rgba(139,151,173,0.08)' } },
        y: {
          type: 'logarithmic',
          title: { display: true, text: 'seconds (log)', color: '#8b97ad' },
          ticks: { color: '#8b97ad' },
          grid: { color: 'rgba(139,151,173,0.1)' },
        },
      },
    },
  });
}
