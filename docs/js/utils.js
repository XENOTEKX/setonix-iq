// web/js/utils.js — small pure helpers

export function escHtml(s) {
  if (s == null) return '';
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

export function escAttr(s) {
  return escHtml(s);
}

export function fmtTime(seconds) {
  if (seconds == null || isNaN(seconds)) return 'N/A';
  const s = Number(seconds);
  if (s < 1) return `${(s * 1000).toFixed(0)}ms`;
  if (s < 60) return `${s.toFixed(2)}s`;
  if (s < 3600) {
    const m = Math.floor(s / 60);
    const r = Math.round(s - m * 60);
    return `${m}m${r}s`;
  }
  const h = Math.floor(s / 3600);
  const m = Math.floor((s - h * 3600) / 60);
  return `${h}h${m.toString().padStart(2, '0')}m`;
}

/**
 * Chart.js generateLabels override — replaces strikethrough on hidden datasets
 * with dimmed text so legend entries remain readable after toggle.
 * Usage: plugins.legend.labels.generateLabels = dimLegendHidden
 */
export function dimLegendHidden(chart) {
  const base = window.Chart?.defaults?.plugins?.legend?.labels?.generateLabels;
  if (!base) return [];
  return base(chart).map(item => {
    if (item.hidden) {
      item.lineThrough = false;
      item.fontColor = 'rgba(139,151,173,0.3)';
    }
    return item;
  });
}

export function fmtNum(n, digits = 2) {
  if (n == null || isNaN(n)) return 'N/A';
  return Number(n).toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: 0,
  });
}

export function fmtPercent(n, digits = 1) {
  if (n == null || isNaN(n)) return 'N/A';
  return `${Number(n).toFixed(digits)}%`;
}

export function shortFn(fn) {
  if (!fn) return '';
  // Strip template args noise, keep first 80 chars
  return fn.length > 80 ? fn.slice(0, 77) + '…' : fn;
}

export async function copyToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // Fallback for older browsers / non-secure context
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.left = '-9999px';
    document.body.appendChild(ta);
    ta.select();
    try {
      document.execCommand('copy');
      return true;
    } catch {
      return false;
    } finally {
      ta.remove();
    }
  }
}

export function debounce(fn, ms = 150) {
  let t;
  return function (...args) {
    clearTimeout(t);
    t = setTimeout(() => fn.apply(this, args), ms);
  };
}

// Simple hash-based colour for deterministic palette assignment
export function hashColour(s, alpha = 1) {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
  const hue = Math.abs(h) % 360;
  return `hsla(${hue}, 65%, 60%, ${alpha})`;
}

// Platform-tinted colour: Setonix runs live in the cool half of the wheel
// (blue/purple/teal), Gadi runs in the warm half (orange/red/amber). The
// dataset name chooses the exact hue inside each band, so each
// (platform, dataset) pair has a stable but platform-recognisable colour.
export function platformColour(platform, datasetKey, alpha = 1) {
  let h = 0;
  for (let i = 0; i < datasetKey.length; i++) h = (h * 31 + datasetKey.charCodeAt(i)) | 0;
  const band = Math.abs(h) % 80; // 0..79
  let hue;
  if (platform === 'gadi') {
    hue = 15 + band * 0.6;   // ~15..63  (orange → amber)
  } else if (platform === 'setonix') {
    hue = 200 + band * 0.9;  // ~200..272 (blue → violet)
  } else {
    hue = 140 + band * 0.5;  // teal fallback
  }
  const sat = platform === 'gadi' ? 78 : 68;
  const lum = platform === 'gadi' ? 58 : 62;
  return `hsla(${hue}, ${sat}%, ${lum}%, ${alpha})`;
}

/**
 * buildFamily — maps a run's build_tag (and fallback non_canonical_label)
 * to a human-readable patch-family name used for filter chips.
 *
 * Returns one of:
 *   "GCC canonical"      sr_gcc_pin
 *   "ICX baseline (ref)" sr_icx
 *   "R2 · NUMA patch"    icx_omp_pin_numa_ft_r2 / *_v312
 *   "R2 · MPI"           icx_mpi* without avx512
 *   "AVX-512 + R2"       *avx512*
 *   "MF2 Full"           mf2_full*
 *   "MF2 MF-only"        mf2_mfonly*
 *   "MF2 Dispatch"       mf2_dispatch*  /  nc_label containing "MF-only MF2 audit"
 *   "AOCC / Setonix"     clang_*, smtoff_pin, baseline_smton
 *   "FCA mf-iso (MF-only)" mf_iso_mfonly / mf_iso_baseline_repro (TESTONLY, no SPR)
 *   "FCA mf-iso (full)"    mf_iso full run (MF+SPR, -m TEST)
 *   "Other"              everything else
 */
export function buildFamily(r) {
  const tag = r?.build_tag || '';
  const nc  = r?.non_canonical_label || '';
  if (!tag && nc.includes('MF-only MF2')) return 'MF2 MF-only';
  if (!tag && nc.includes('AVX-512+R2 anchor')) return 'AVX-512 + R2';
  if (tag === 'sr_gcc_pin') return 'GCC canonical';
  if (tag === 'sr_icx') return 'ICX baseline (ref)';
  if (tag.startsWith('mf2_full')) return 'MF2 Full';
  if (tag.startsWith('mf2_mfonly')) return 'MF2 MF-only';
  if (tag.startsWith('mf2_dispatch')) return 'MF2 Dispatch';
  if (tag.startsWith('mf_iso') || r?.run_type === 'mf_iso_baseline_repro') {
    // model_finder_only is the authoritative field; run_type 'mf_iso_mfonly' and
    // 'mf_iso_baseline_repro' are the legacy/repro discriminators.
    const isMFOnly = r?.model_finder_only === true
                  || r?.run_type === 'mf_iso_mfonly'
                  || r?.run_type === 'mf_iso_baseline_repro';
    return isMFOnly ? 'FCA mf-iso (MF-only)' : 'FCA mf-iso (full)';
  }
  if (tag.includes('avx512') || tag.includes('avx_512') || tag.includes('r2_anchor')) return 'AVX-512 + R2';
  if (tag.startsWith('icx_omp_pin_numa_ft_r2')) return 'R2 · NUMA patch';
  if (tag.startsWith('icx_mpi')) return 'R2 · MPI';
  if (tag.startsWith('clang_') || tag === 'smtoff_pin' || tag === 'baseline_smton') return 'AOCC / Setonix';
  return 'Other';
}

// Ordered list of all recognised families for stable chip ordering.
export const BUILD_FAMILIES = [
  'GCC canonical',
  'ICX baseline (ref)',
  'R2 · NUMA patch',
  'R2 · MPI',
  'AVX-512 + R2',
  'MF2 Full',
  'MF2 MF-only',
  'MF2 Dispatch',
  'FCA mf-iso (MF-only)',
  'FCA mf-iso (full)',
  'AOCC / Setonix',
  'Other',
];

export function createEl(tag, attrs = {}, children = []) {
  const el = document.createElement(tag);
  for (const k in attrs) {
    if (k === 'class') el.className = attrs[k];
    else if (k === 'html') el.innerHTML = attrs[k];
    else if (k.startsWith('on')) el.addEventListener(k.slice(2).toLowerCase(), attrs[k]);
    else el.setAttribute(k, attrs[k]);
  }
  for (const c of [].concat(children)) {
    if (c == null) continue;
    el.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
  }
  return el;
}
