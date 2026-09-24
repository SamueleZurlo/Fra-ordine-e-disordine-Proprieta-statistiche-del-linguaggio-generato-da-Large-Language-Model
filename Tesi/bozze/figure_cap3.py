"""Figura del Capitolo 3: deriva degli stimatori con la lunghezza di analisi.

Produce  figure/cap3_dimensione_finita.{png,pdf}

Pannello A: esponenti misurati sul livello dei token, in funzione del numero di
            token analizzati (baseline umana, continuazione di Guerra e pace).
Pannello B: coefficiente di variazione degli intervalli fra occorrenze della
            parola 'prince', in funzione del numero di caratteri analizzati.

USO
---
    py figure_cap3.py              # in bozze/anteprima
    py figure_cap3.py --applica    # in Tesi/figure
"""
import collections, re, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tiktoken

BASE = str(Path(__file__).resolve().parents[2])
DEST = BASE + ('/Tesi/figure' if '--applica' in sys.argv else '/Tesi/bozze/anteprima')
os.makedirs(DEST, exist_ok=True)


# ------------------------------------------- stimatori, dal notebook dello sweep
# Stesse definizioni delle celle 4.1 e 4.2 di
# 'Analisi Sweep di temperature/analisi_sweep_temperature.ipynb'.
def fit_powerlaw(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = (x > 0) & (y > 0)
    if m.sum() < 3:
        return np.nan, np.nan, np.nan
    lx, ly = np.log10(x[m]), np.log10(y[m])
    slope, inter = np.polyfit(lx, ly, 1)
    yhat = 10.0 ** (inter + slope * lx)
    return -slope, inter, float(np.mean(np.abs((y[m] - yhat) / y[m])))


def zipf_cumulative_fit(tokens):
    c = collections.Counter(tokens)
    conteggi = np.array(sorted(c.values()), dtype=float)
    x = np.unique(conteggi)
    surv = np.array([(conteggi > xi).sum() for xi in x], dtype=float) / len(conteggi)
    x, y = x[surv > 0], surv[surv > 0]
    if len(x) < 5:
        return np.nan, np.nan
    a, _, mape = fit_powerlaw(x, y)
    return a, mape


def heaps_beta(tokens, tmin=1000):
    seen, n, k = set(), np.empty(len(tokens), dtype=np.int64), 0
    for i, w in enumerate(tokens):
        if w not in seen:
            seen.add(w); k += 1
        n[i] = k
    if len(n) < tmin + 100:
        tmin = max(10, len(n) // 10)
    idx = np.unique(np.logspace(np.log10(tmin), np.log10(len(n)), 60).astype(int))
    idx = idx[(idx >= tmin) & (idx <= len(n))]
    if len(idx) < 5:
        return np.nan
    a, _, _ = fit_powerlaw(idx, n[idx - 1])
    return -a


def descriptor_R(tokens):
    if len(tokens) < 20:
        return np.nan
    h = len(tokens) // 2
    V1, V2 = set(tokens[:h]), set(tokens[h:])
    return len(V2 - V1) / len(V1) if V1 else np.nan

plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.25,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False,
})

C_ALPHA, C_HEAPS, C_R, C_MAPE, C_CV = '#2F4E7E', '#A6432F', '#4E7E2F', '#8A6210', '#2F4E7E'

# ------------------------------------------------------------------ corpus
raw = open(BASE + '/corpora_cache/f6c229e9b581.txt', 'rb').read().decode('utf-8', 'replace')
i = raw.find('Well, Prince, so Genoa and Lucca')
libro = raw[i:]
j = libro.find('*** END OF')
if j != -1:
    libro = libro[:j]

# la baseline umana e' la continuazione del prompt (8262 caratteri)
continuazione = libro[8262:]

enc = tiktoken.get_encoding('cl100k_base')
ids = enc.encode(continuazione[:3_000_000], disallowed_special=())
print(f'continuazione: {len(continuazione):,} caratteri, {len(ids):,} token')

# --------------------------------------------- A: esponenti sui token
N_TOK = [5000, 10000, 20000, 35000, 50000, 100000, 200000, 300000, 500000]
N_TOK = [n for n in N_TOK if n <= len(ids)]
alpha, mape, heaps, Rd = [], [], [], []
for n in N_TOK:
    tk = [str(t) for t in ids[:n]]
    a, m = zipf_cumulative_fit(tk)
    alpha.append(a); mape.append(m)
    heaps.append(heaps_beta(tk))
    Rd.append(descriptor_R(tk))
    print(f'  N={n:>7,}  alpha={a:.3f}  MAPE={m:.3f}  gamma_H={heaps[-1]:.3f}  R={Rd[-1]:.3f}')

# --------------------------------------------- B: cv di 'prince'
pos = np.array([m.start() for m in re.finditer(r'\bprince\b', libro, re.I)])
N_CHR = np.unique(np.logspace(np.log10(30000), np.log10(len(libro)), 26).astype(int))
cv = []
for n in N_CHR:
    p = pos[pos < n]
    if len(p) < 5:
        cv.append(np.nan); continue
    tau = np.diff(p)
    cv.append(tau.std() / tau.mean())
cv = np.array(cv)
print(f'  cv a 50k = {cv[np.argmin(np.abs(N_CHR-50000))]:.3f}   '
      f'cv sul libro intero = {cv[-1]:.3f}')

# ------------------------------------------------------------------ figura
fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.4))

ax = axes[0]
ax.plot(N_TOK, alpha, 'o-', color=C_ALPHA, ms=5, lw=1.7, label=r'$\alpha$  (Zipf)')
ax.plot(N_TOK, heaps, 's-', color=C_HEAPS, ms=5, lw=1.7, label=r'$\gamma_H$  (Heaps)')
ax.plot(N_TOK, Rd,    '^-', color=C_R,     ms=5, lw=1.7, label=r'$R$')
ax.plot(N_TOK, mape,  'D-', color=C_MAPE,  ms=5, lw=1.7, label='MAPE')
ax.axvline(20000, color='crimson', ls=':', lw=1.4)
ax.annotate('lunghezza di lavoro', xy=(20000, 0.03), xytext=(-4, 0),
            textcoords='offset points', fontsize=8, color='crimson',
            rotation=90, va='bottom', ha='right')
ax.set_xscale('log')
ax.set_xlabel('token analizzati')
ax.set_ylabel('valore dello stimatore')
ax.set_ylim(0, 1.22)
ax.set_title('A. Esponenti misurati sul livello dei token', fontsize=10.5, loc='left')
ax.legend(fontsize=8, ncol=2)

ax = axes[1]
ax.plot(N_CHR, cv, 'o-', color=C_CV, ms=4.5, lw=1.7)
ax.axhline(1.0, color='grey', ls=':', lw=1.2)
ax.annotate('processo di Poisson', xy=(N_CHR[0]*1.1, 1.03), fontsize=8, color='grey')
for x, testo, dx, dy, ha in ((50000, '50 mila caratteri', 8, 12, 'left'),
                             (len(libro), 'libro intero', -10, 14, 'right')):
    k = int(np.argmin(np.abs(N_CHR - x)))
    ax.plot([N_CHR[k]], [cv[k]], 'o', ms=10, mfc='none', mec='crimson', mew=1.8)
    ax.annotate(f'{testo}\ncv = {cv[k]:.2f}', xy=(N_CHR[k], cv[k]),
                xytext=(dx, dy), textcoords='offset points',
                fontsize=8, color='crimson', ha=ha)
ax.set_xscale('log')
ax.set_xlim(N_CHR[0] * 0.8, len(libro) * 1.6)
ax.set_ylim(0.9, 4.35)
ax.set_xlabel('caratteri analizzati')
ax.set_ylabel(r'$\sigma_\tau/\langle\tau\rangle$')
ax.set_title("B. Burstiness della parola \u2018prince\u2019", fontsize=10.5, loc='left')

plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(f'{DEST}/cap3_dimensione_finita.{ext}')
print('\nscritta', DEST + '/cap3_dimensione_finita.png')
