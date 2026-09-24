"""
Rifa' la figura sulla robustezza rispetto alla lunghezza di analisi (§5.10).

La cella del notebook che la disegna legge confronto_N_eff/confronto.json, che
non viene rigenerato dal notebook stesso. Questo programma la ridisegna dal JSON
prodotto da confronto_neff.py con la lista di parole funzione estesa.

Rispetto alla versione precedente cambia il titolo del pannello B: con la
classificazione corretta il divario appaiato NON e' piu' invariante rispetto
alla lunghezza al livello delle parole chiave, e la figura deve dirlo.

USO
---
    py figure_cap5_robustezza.py              # in bozze/anteprima
    py figure_cap5_robustezza.py --applica    # in Tesi/figure
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[2]
SORGENTE = BASE / 'Wiki' / 'confronto_N_eff_v2' / 'confronto.json'
OUT = (BASE / 'Tesi' / 'figure') if '--applica' in sys.argv \
    else (BASE / 'Tesi' / 'bozze' / 'anteprima')
OUT.mkdir(parents=True, exist_ok=True)

ORDINE = ['letterario', 'wikipedia', 'grok_v01', 'grok_oggi']
ETICHETTE = {'letterario': 'Guerra e pace', 'wikipedia': 'Wikipedia',
             'grok_v01': 'Grokipedia v0.1', 'grok_oggi': 'Grokipedia oggi'}
COLORI = {'letterario': '#333333', 'wikipedia': '#0072B2',
          'grok_v01': '#E69F00', 'grok_oggi': '#D55E00'}
LIV = [('lettera', 'lettere'), ('funzione', 'parole funzione'),
       ('appaiata', 'controlli appaiati'), ('keyword', 'keyword')]

plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
                     'font.size': 10, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.axisbelow': True, 'legend.frameon': True})

R = json.loads(SORGENTE.read_text(encoding='utf-8'))
# il confronto fra lunghezze deve isolare la lunghezza: a 60k si usa il
# blocco ristretto alle stesse 35 terne che sopravvivono a 80k
a, b = R['60000_35'], R['80000']

fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
xs = np.arange(len(LIV))

ax = axes[0]
for c in ORDINE:
    ax.plot(xs, [a[f'cv_tau_{c}_{t}'] for t, _ in LIV], '-o',
            color=COLORI[c], ms=6, lw=1.7, label=ETICHETTE[c])
    ax.plot(xs, [b[f'cv_tau_{c}_{t}'] for t, _ in LIV], '--s',
            color=COLORI[c], ms=5, lw=1.2, alpha=.6)
ax.set_xticks(xs)
ax.set_xticklabels([n for _, n in LIV], rotation=12, fontsize=8)
ax.set_ylabel(r'$\sigma_\tau/\langle\tau\rangle$')
ax.set_title('A) burstiness: 60k (pieno) e 80k (tratteggio),\n'
             'stesse 35 terne a entrambe le lunghezze', fontsize=10)
ax.legend(fontsize=7)

ax = axes[1]
ax.plot(xs, [a[f'pair_{t}'][1] for t, _ in LIV], '-o', color='#0072B2',
        ms=7, lw=1.8, label='60k')
ax.plot(xs, [b[f'pair_{t}'][1] for t, _ in LIV], '--s', color='#D55E00',
        ms=6, lw=1.6, label='80k')
ax.axhline(0, color='grey', ls=':', lw=1.2)
ax.set_xticks(xs)
ax.set_xticklabels([n for _, n in LIV], rotation=12, fontsize=8)
ax.set_ylabel('Grokipedia − Wikipedia')
ax.set_title('B) divario appaiato: robusto fino ai controlli,\n'
             'non sulle parole chiave', fontsize=10)
ax.legend(fontsize=8)

fig.suptitle('Robustezza rispetto alla lunghezza di analisi (35 terne appaiate)',
             fontweight='bold')
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'cap5_robustezza_neff.{ext}')
plt.close(fig)
print(f'-> {OUT / "cap5_robustezza_neff.png"}')
