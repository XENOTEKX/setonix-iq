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
