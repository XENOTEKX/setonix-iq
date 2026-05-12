// web/js/components/copy-button.js

import { copyToClipboard } from '../utils.js?v=b53a2b04f6be';
import { showToast } from './toast.js?v=b53a2b04f6be';

export function bindCopyButtons(root = document) {
  root.querySelectorAll('[data-copy]').forEach((btn) => {
    if (btn.dataset.bound) return;
    btn.dataset.bound = '1';
    btn.addEventListener('click', async () => {
      const src = btn.dataset.copy;
      const text = src.startsWith('#')
        ? (document.querySelector(src)?.textContent || '')
        : src;
      const ok = await copyToClipboard(text);
      if (ok) {
        btn.classList.add('copied');
        const original = btn.textContent;
        btn.textContent = '✓ Copied';
        setTimeout(() => {
          btn.classList.remove('copied');
          btn.textContent = original;
        }, 1200);
      } else {
        showToast('Copy failed — clipboard unavailable');
      }
    });
  });
}
