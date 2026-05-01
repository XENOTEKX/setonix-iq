// web/js/charts/microarch.js — radar-style comparison of CPU microarch metrics

import { hashColour } from '../utils.js?v=20260501004534';

let chart;

const AXES = [
  { key: 'IPC', label: 'IPC', normalize: (v) => Math.min(100, (v / 3.5) * 100) },
  // Replaced the kernel `cache-miss-rate` axis (L2 on AMD vs L3 on Intel —
  // not comparable across platforms; see CHANGELOG follow-up #15).
  // L1d-MPKI uses identical events on both PMUs and is the recommended
  // primary cross-platform memory-pressure metric. Lower MPKI = healthier
  // memory behaviour, so we invert + clamp to a 0-50 MPKI window where
  // 0 → 100 score and ≥50 → 0 score (50 MPKI is "very memory-bound" in
  // most HPC workloads).
  { key: 'L1d-mpki', label: 'Mem light', normalize: (v) => Math.max(0, 100 - Math.min(100, (v / 50) * 100)) },
  { key: 'branch-miss-rate', label: 'Branch hit', normalize: (v) => 100 - Math.min(100, v) },
  { key: 'L1-dcache-miss-rate', label: 'L1 hit', normalize: (v) => 100 - Math.min(100, v) },
  { key: 'frontend-stall-rate', label: 'FE alive', normalize: (v) => 100 - Math.min(100, v) },
];

export function render(canvas, runs) {
  if (chart) chart.destroy();

  const datasets = runs
    .filter((r) => r && r.profile && r.profile.metrics)
    .map((r) => {
      const m = r.profile.metrics;
      const c = hashColour(r.run_id, 0.8);
      const cFill = hashColour(r.run_id, 0.15);
      return {
        label: r.label || r.run_id,
        data: AXES.map((a) => (m[a.key] != null ? a.normalize(m[a.key]) : 0)),
        borderColor: c,
        backgroundColor: cFill,
        borderWidth: 2,
        pointRadius: 3,
      };
    });

  chart = new Chart(canvas, {
    type: 'radar',
    data: { labels: AXES.map((a) => a.label), datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'bottom', labels: { color: '#8b97ad', font: { size: 10 } } },
      },
      scales: {
        r: {
          min: 0,
          max: 100,
          angleLines: { color: 'rgba(139,151,173,0.15)' },
          grid: { color: 'rgba(139,151,173,0.1)' },
          pointLabels: { color: '#8b97ad', font: { size: 11 } },
          ticks: { display: false },
        },
      },
    },
  });
}
