// web/js/components/toast.js — minimal toast

let current;

export function showToast(msg, ms = 1800) {
  if (current) current.remove();
  const el = document.createElement('div');
  el.className = 'toast';
  el.setAttribute('role', 'status');
  el.textContent = msg;
  document.body.appendChild(el);
  current = el;
  setTimeout(() => {
    if (el === current) {
      el.remove();
      current = null;
    }
  }, ms);
}
