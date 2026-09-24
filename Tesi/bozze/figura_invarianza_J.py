# -*- coding: utf-8 -*-
"""Figura del §2.5: fino a che punto J e' invariante per dilatazione.

Gli intervalli sono interi. La legge continua e' invariante per dilatazione, ma
l'arrotondamento agli interi non lo e': a intervalli medi piccoli la
quantizzazione domina e J dipende dalla scala. La figura misura dove la
dipendenza sparisce.

Il pannello B riporta il residuo di bias in funzione del numero di intervalli,
su leggi di forma nota.

USO
---
    py figura_invarianza_J.py              # in bozze/anteprima
    py figura_invarianza_J.py --applica    # in Tesi/figure
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rigenera_cap5_stoplist import celle_codice   # noqa: E402
from autoanalisi import WIKI                      # noqa: E402

BASE = Path(__file__).resolve().parents[2]
OUT = (BASE / 'Tesi' / 'figure') if '--applica' in sys.argv \
    else (BASE / 'Tesi' / 'bozze' / 'anteprima')
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260901
N_REP = 60
MU = (5, 10, 30, 100, 300, 900)
CS = np.array([1, 2, 5, 10, 50])
NS = (30, 60, 150, 400, 1000)

plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 150,
                     'savefig.bbox': 'tight', 'font.size': 10,
                     'axes.grid': True, 'grid.alpha': .25,
                     'axes.axisbelow': True, 'legend.frameon': True})


def funzioni_J():
    ce = celle_codice(WIKI / 'entropia_intervalli.ipynb')
    nj = {'__name__': '__main__', 'display': lambda *a, **k: None}
    for i in (0, 1, 2):
        exec(compile(ce[i], f'<ent#{i}>', 'exec'), nj)
    return nj


def main():
    cwd = Path.cwd()
    os.chdir(WIKI)
    try:
        nj = funzioni_J()
    finally:
        os.chdir(cwd)
    J = nj['J_intervalli']
    r = np.random.default_rng(SEED)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5))

    # ---------------------------------------------------- A) dilatazione
    ax = axes[0]
    colori = plt.cm.viridis(np.linspace(0, .88, len(MU)))
    for mu, col in zip(MU, colori):
        v = []
        for c in CS:
            vals = []
            for _ in range(N_REP):
                base = np.maximum(1, np.round(
                    r.lognormal(np.log(mu), 1.0, 200))).astype(np.int64)
                x = J(np.maximum(1, np.round(base * c)).astype(np.int64), r)
                if np.isfinite(x):
                    vals.append(x)
            v.append(np.mean(vals))
        ax.plot(CS, v, '-o', color=col, ms=5, lw=1.6,
                label=r'$\langle\tau\rangle = %d$' % mu)
    ax.set_xscale('log')
    ax.set_xlabel('fattore di dilatazione $c$')
    ax.set_ylabel('$J$ [bit]')
    ax.set_title('A) stessa distribuzione, intervalli moltiplicati per $c$\n'
                 r'$J$ è costante solo per $\langle\tau\rangle \gtrsim 30$',
                 fontsize=10)
    ax.legend(fontsize=7.5, ncol=2)

    # ------------------------------------------------------- B) bias in n
    ax = axes[1]
    forme = [
        ('esponenziale (Poisson)', lambda n, m: r.exponential(m, n)),
        (r'log-normale $\sigma=1$',
         lambda n, m: r.lognormal(np.log(m) - .5, 1.0, n)),
        (r'log-normale $\sigma=1.5$',
         lambda n, m: r.lognormal(np.log(m) - 1.1, 1.5, n)),
        (r'Pareto $\mu=2.5$',
         lambda n, m: m * 0.6 * (1 - r.random(n)) ** (-1 / 1.5)),
        ('quasi deterministica',
         lambda n, m: m * (1 + 0.15 * r.standard_normal(n))),
    ]
    for nome, gen in forme:
        v = []
        for n in NS:
            vals = []
            for _ in range(N_REP):
                x = J(np.maximum(1, np.round(gen(n, 200))).astype(np.int64), r)
                if np.isfinite(x):
                    vals.append(x)
            v.append(np.mean(vals))
        ax.plot(NS, v, '-o', ms=4.5, lw=1.5, label=nome)
    ax.axhline(0, color='grey', ls='--', lw=1.2)
    ax.set_xscale('log')
    ax.set_xlabel('numero di intervalli $n$')
    ax.set_ylabel('$J$ [bit]')
    ax.set_title('B) leggi di forma nota, $n$ crescente\n'
                 'le curve piatte indicano che il bias si cancella',
                 fontsize=10)
    ax.legend(fontsize=7.5)

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'cap2_invarianza_J.{ext}')
    plt.close(fig)
    print(f'-> {OUT / "cap2_invarianza_J.png"}')


if __name__ == '__main__':
    main()
