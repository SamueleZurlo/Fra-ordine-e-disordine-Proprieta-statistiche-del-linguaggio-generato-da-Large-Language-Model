"""
Figura della sottosezione sull'effetto della lista di parole funzione.

A sinistra il grafico che ha reso visibile il problema: quali parole popolavano
il quadrante "regolare e scorrelata" con la lista originaria. A destra il
risultato che ne dipendeva di piu', il test interno della Figura 5.3A, con le
due classificazioni a confronto.

USO
---
    py figure_cap5_stoplist.py              # in bozze/anteprima
    py figure_cap5_stoplist.py --applica    # in Tesi/figure
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[2]
VECCHIO = BASE / 'Wiki' / 'risultati_tre_corpora' / 'dati'
NUOVO = BASE / 'Wiki' / 'risultati_tre_corpora_v2' / 'dati'
OUT = (BASE / 'Tesi' / 'figure') if '--applica' in sys.argv \
    else (BASE / 'Tesi' / 'bozze' / 'anteprima')
OUT.mkdir(parents=True, exist_ok=True)

ORDINE = ['letterario', 'wikipedia', 'grok_v01', 'grok_oggi']
ETICHETTE = {'letterario': 'Guerra e pace', 'wikipedia': 'Wikipedia',
             'grok_v01': 'Grokipedia v0.1', 'grok_oggi': 'Grokipedia oggi'}
COLORI = {'letterario': '#333333', 'wikipedia': '#0072B2',
          'grok_v01': '#E69F00', 'grok_oggi': '#D55E00'}

plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
                     'font.size': 10, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.axisbelow': True, 'legend.frameon': False,
                     'axes.spines.top': False, 'axes.spines.right': False})

fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.2))

# --- A: che cosa popolava il quadrante con la lista originaria -------------
A = pd.read_csv(VECCHIO / 'sequenze.csv')
K = A[(A.tipo == 'keyword') & np.isfinite(A.cv_tau) & np.isfinite(A.gamma)]
bs = K[(K.cv_tau < 1) & (K.gamma < 1)]
conta = {c: Counter(bs[bs.corpus == c].sequenza) for c in ('wikipedia', 'grok_oggi')}
parole = [w for w, _ in conta['grok_oggi'].most_common(6)]
parole += [w for w, _ in conta['wikipedia'].most_common(3) if w not in parole]

ax = axes[0]
y = np.arange(len(parole))
h = .38
ax.barh(y - h / 2, [conta['wikipedia'][w] for w in parole], height=h,
        color=COLORI['wikipedia'], label='Wikipedia')
ax.barh(y + h / 2, [conta['grok_oggi'][w] for w in parole], height=h,
        color=COLORI['grok_oggi'], label='Grokipedia oggi')
ax.set_yticks(y)
ax.set_yticklabels([f'"{w}"' for w in parole], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('voci in cui la parola cade fra le keyword\n'
              'regolari e scorrelate', fontsize=9)
ax.set_title('A. Con la lista originaria di parole funzione\n'
             'i connettivi entravano fra le parole chiave di Grokipedia',
             fontsize=10)
ax.legend(fontsize=8, loc='lower right')
ax.grid(axis='y', visible=False)

# --- B: il risultato che ne dipendeva di piu' ------------------------------
ax = axes[1]
Tv = pd.read_csv(VECCHIO / 'test_interno.csv')
Tn = pd.read_csv(NUOVO / 'test_interno.csv')
y = np.arange(len(ORDINE))
for eti, T, dy, mk, alpha in (('lista originaria', Tv, -.16, 's', .45),
                              ('lista estesa', Tn, +.16, 'o', 1.0)):
    s = T[T.metrica == 'cv_tau'].set_index('corpus').reindex(ORDINE)
    for i, c in enumerate(ORDINE):
        err = np.array([[max(s.loc[c, 'frazione'] - s.loc[c, 'ic_lo'], 0)],
                        [max(s.loc[c, 'ic_hi'] - s.loc[c, 'frazione'], 0)]])
        ax.errorbar([s.loc[c, 'frazione']], [i + dy], xerr=err, fmt=mk, ms=8,
                    capsize=4, color=COLORI[c], lw=1.8, alpha=alpha,
                    label=eti if i == 0 else None)
ax.set_yticks(y)
ax.set_yticklabels([ETICHETTE[c] for c in ORDINE], fontsize=9)
ax.set_ylim(-.6, len(ORDINE) - .4)
ax.set_xlim(0.45, 0.95)
ax.axvline(.5, color='grey', ls=':', lw=1.4)
ax.set_xlabel('frazione di coppie con keyword più bursty')
ax.set_title('B. Il test interno della Fig. 5.3A\n'
             'quadrati: lista originaria; cerchi: lista estesa', fontsize=10)
ax.grid(axis='y', visible=False)

for a in axes:
    a.set_box_aspect(1)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'cap5_effetto_stoplist.{ext}')
plt.close(fig)
print(f'-> {OUT / "cap5_effetto_stoplist.png"}')
