# -*- coding: utf-8 -*-
"""Rigenera la Figura 5.7 (cap5_esponente_coda).

Due modifiche rispetto alla versione precedente.

  PANNELLO A. Le linee di collegamento non attraversano piu' le celle in cui il
  fit degenera in esponenziale (tau_c < 3), dove mu non e' interpretabile. Prima
  la linea le univa agli altri punti e suggeriva un andamento fra valori di cui
  tre quarti non significano nulla: era la ragione per cui il romanzo sembrava
  avere un comportamento anomalo alle parole funzione, quando in realta' e'
  l'unico corpus il cui fit NON degenera a quel livello.

  PANNELLO B. Il test gamma_A2 <= 4 - mu viene sostituito da tau_c ai quattro
  livelli. Il vecchio pannello era quasi vuoto: gamma <= 2 per costruzione
  (X(t) <= t implica sigma^2_X <= t^2/4), quindi ogni volta che mu < 2 il tetto
  4 - mu supera 2 e la disuguaglianza e' vera per aritmetica, qualunque cosa
  dicano i dati. Accadeva in 9 celle su 16, e in 7 mu non era nemmeno
  interpretabile. Il cut-off, invece, e' l'altra meta' di cio' che Altmann
  assume senza misurare, ed e' la grandezza che spiega il pannello A.

Gli intervalli di confidenza su tau_c sono di profilo, come quelli su mu, e
vengono calcolati qui per la prima volta.

Uso:
    py rigenera_fig_coda.py             # figura in bozze/anteprima
    py rigenera_fig_coda.py --applica   # figura in Tesi/figure
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate, optimize

RADICE = Path(__file__).resolve().parents[2]
WIKI = RADICE / 'Wiki'
APPLICA = '--applica' in sys.argv
OUT = (RADICE / 'Tesi' / 'figure') if APPLICA \
    else (RADICE / 'Tesi' / 'bozze' / 'anteprima')
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path(__file__).resolve().parent / 'tau_c_ic.csv'

CELLE = [2, 3, 5, 7, 9, 10]
ORD = ['letterario', 'wikipedia', 'grok_v01', 'grok_oggi']
NOMI = {'letterario': 'Guerra e pace', 'wikipedia': 'Wikipedia',
        'grok_v01': 'Grokipedia v0.1', 'grok_oggi': 'Grokipedia oggi'}
COL = {'letterario': '#333333', 'wikipedia': '#0072B2',
       'grok_v01': '#E69F00', 'grok_oggi': '#D55E00'}
LIV = [('lettera', 'lettere'), ('funzione', 'parole funzione'),
       ('appaiata', 'controlli appaiati'), ('keyword', 'keyword')]
MARK = ['o', 's', '^', 'D']
QUANT_XMIN = 90
TAUC_MIN = 3.0


# ------------------------------------------------------------- pool
def carica_pool():
    prima = os.getcwd()
    os.chdir(WIKI)
    try:
        nb = json.load(open('esponente_coda.ipynb', encoding='utf-8'))
        ns = {'__name__': '__main__'}
        for i in CELLE:
            src = ''.join(nb['cells'][i]['source'])
            righe = [l for l in src.splitlines()
                     if not l.lstrip().startswith(('%', '!'))]
            exec('\n'.join(righe), ns)
        return ns['POOL']
    finally:
        os.chdir(prima)


# ------------------------------------------- modello a potenza troncata
def _Z(mu, xc, xmin):
    v, _ = integrate.quad(lambda t: t ** (-mu) * math.exp(-t / xc), xmin, np.inf,
                          limit=200)
    return v


def nll(par, xmin, S_log, S_tau, n):
    mu, lxc = par
    xc = math.exp(lxc)
    if mu <= -1 or mu > 8 or xc <= 1e-3 or xc > 1e6:
        return 1e12
    Z = _Z(mu, xc, xmin)
    if not np.isfinite(Z) or Z <= 0:
        return 1e12
    return -(-mu * S_log - S_tau / xc - n * math.log(Z))


def fit(x, xmin):
    S_log, S_tau, n = float(np.log(x).sum()), float(x.sum()), len(x)
    s = [2.5, math.log(max(float(x.mean()), xmin * 2))]
    r = optimize.minimize(nll, s, args=(xmin, S_log, S_tau, n),
                          method='Nelder-Mead',
                          options=dict(maxiter=1200, xatol=1e-5, fatol=1e-5))
    return float(r.x[0]), float(math.exp(r.x[1])), float(r.fun)


def profilo_lxc(lxc, x, xmin, S, mu0):
    """tau_c fissato, mu riottimizzato. Warm start dal mu precedente."""
    f = lambda m: nll([m[0], lxc], xmin, *S)
    r = optimize.minimize(f, [mu0], method='Nelder-Mead',
                          options=dict(maxiter=300, xatol=1e-4, fatol=1e-4))
    return float(r.fun), float(r.x[0])


def ic_profilo_tauc(x, xmin, xc_h, f_h, soglia=3.84):
    """IC 95% di profilo su tau_c: l'insieme dei tau_c per cui la
    verosimiglianza, riottimizzata in mu, resta entro chi2(1, 0.95)/2."""
    S = (float(np.log(x).sum()), float(x.sum()), len(x))
    l_h = math.log(xc_h)
    fuori = []
    for direzione in (-1, +1):
        dentro, l, mu0 = l_h, l_h, 2.0
        for _ in range(30):
            l = l + direzione * 0.15
            if not (math.log(0.02) < l < math.log(5e3)):
                break
            f_, mu0 = profilo_lxc(l, x, xmin, S, mu0)
            if 2 * (f_ - f_h) > soglia:
                break
            dentro = l
        a, b, mu0 = dentro, l, 2.0
        for _ in range(18):
            m = (a + b) / 2
            f_, mu0 = profilo_lxc(m, x, xmin, S, mu0)
            if 2 * (f_ - f_h) > soglia:
                b = m
            else:
                a = m
        fuori.append(math.exp(a))
    return min(fuori), max(fuori)


# ------------------------------------------------------------- stima
def stima():
    if CACHE.exists():
        print(f'riuso {CACHE.name}')
        return pd.read_csv(CACHE)
    print('Ricostruzione del pool...')
    POOL = carica_pool()
    righe = []
    for c in ORD:
        for tipo, _ in LIV:
            if (c, tipo) not in POOL:
                continue
            x0 = np.concatenate(POOL[(c, tipo)])
            xmin = float(np.percentile(x0, QUANT_XMIN))
            x = x0[x0 >= xmin]
            if len(x) < 150:
                continue
            mu_h, xc_h, f_h = fit(x, xmin)
            lo, hi = ic_profilo_tauc(x, xmin, xc_h, f_h)
            righe.append(dict(corpus=c, tipo=tipo, n_coda=len(x), xmin=xmin,
                              mu=mu_h, tau_c=xc_h, tau_c_lo=lo, tau_c_hi=hi))
            print(f'  {NOMI[c]:17s} {tipo:10s} mu={mu_h:5.2f}  '
                  f'tau_c={xc_h:6.2f} [{lo:5.2f}, {hi:6.2f}]  n={len(x)}')
    df = pd.DataFrame(righe)
    df.to_csv(CACHE, index=False)
    return df


# ------------------------------------------------------------- figura
def figura(TC):
    T = pd.read_csv(WIKI / 'risultati_coda_v2' / 'dati' / 'test_eq8.csv')
    TC = TC.set_index(['corpus', 'tipo'])
    T = T.set_index(['corpus', 'tipo'])

    plt.rcParams.update({
        'figure.dpi': 110, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.25,
        'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.7))
    xs = np.arange(len(LIV))

    # ---------------- A: mu, con le linee spezzate sulle celle degenerate
    ax = axes[0]
    ax.axhspan(2, 3, color='green', alpha=.08)
    ax.axhline(2.4, color='crimson', ls='--', lw=1.4)
    ax.annotate('$\\mu$ = 2.4 (Altmann)', (len(LIV) - .55, 2.44), fontsize=7.5,
                color='crimson', va='bottom', ha='right')
    for i, c in enumerate(ORD):
        v, lo, hi, deg = [], [], [], []
        for tipo, _ in LIV:
            r = T.loc[(c, tipo)]
            v.append(r['mu']); lo.append(r['mu_lo']); hi.append(r['mu_hi'])
            deg.append(bool(r['tau_c'] < TAUC_MIN))
        v, lo, hi, deg = map(np.array, (v, lo, hi, np.array(deg)))
        x = xs + .06 * (i - 1.5)
        err = np.vstack([np.clip(v - lo, 0, None), np.clip(hi - v, 0, None)])
        # la linea salta le celle non interpretabili
        vv = np.where(deg, np.nan, v)
        ax.plot(x, vv, '-', color=COL[c], lw=1.6, alpha=.9, label=NOMI[c])
        ax.errorbar(x[~deg], v[~deg], yerr=err[:, ~deg], fmt='none',
                    ecolor=COL[c], capsize=3, lw=1.4)
        ax.errorbar(x[deg], v[deg], yerr=err[:, deg], fmt='none',
                    ecolor='#bbbbbb', capsize=2, lw=1.0, zorder=1)
        ax.scatter(x[~deg], v[~deg], s=75, marker=MARK[i], color=COL[c],
                   edgecolors='k', linewidths=.6, zorder=3)
        ax.scatter(x[deg], v[deg], s=55, marker=MARK[i], facecolors='white',
                   edgecolors='#bbbbbb', linewidths=1.2, zorder=2)
    ax.scatter([], [], s=55, marker='o', facecolors='white',
               edgecolors='#bbbbbb', linewidths=1.2,
               label='fit degenerato in esponenziale\n($\\tau_c < 3$): $\\mu$ non interpretabile')
    ax.set_xticks(xs); ax.set_xticklabels([n for _, n in LIV], rotation=12, fontsize=8)
    ax.set_ylabel('$\\mu$  (modello troncato)')
    # La banda NON e' una regione a varianza divergente per queste stime: nel
    # modello troncato il taglio rende i momenti finiti per qualunque mu. E' la
    # regione entro cui l'Eq. 8 e' derivata per la legge di potenza pura.
    ax.set_title('A) esponente di coda\n'
                 'banda verde: la regione $2<\\mu<3$ della legge di potenza pura',
                 fontsize=9.5)
    ax.legend(fontsize=6.5, loc='lower right')

    # ---------------- B: tau_c, il cut-off, con IC di profilo
    ax = axes[1]
    ax.axhspan(0.5, TAUC_MIN, color='#bbbbbb', alpha=.22, lw=0)
    ax.axhline(TAUC_MIN, color='#888888', ls='--', lw=1.2)
    ax.text(0.98, 0.05,
            'sotto $\\tau_c = 3$ la coda e' + chr(39) + ' di fatto esponenziale:'
            + chr(10) + 'non resta tratto a legge di potenza da misurare',
            transform=ax.transAxes, fontsize=7.5, color='#666666',
            ha='right', va='bottom')
    for i, c in enumerate(ORD):
        v = np.array([TC.loc[(c, t)]['tau_c'] for t, _ in LIV])
        lo = np.array([TC.loc[(c, t)]['tau_c_lo'] for t, _ in LIV])
        hi = np.array([TC.loc[(c, t)]['tau_c_hi'] for t, _ in LIV])
        x = xs + .06 * (i - 1.5)
        ax.errorbar(x, v, yerr=np.vstack([np.clip(v - lo, 0, None),
                                          np.clip(hi - v, 0, None)]),
                    color=COL[c], lw=1.6, marker=MARK[i], ms=7, capsize=3,
                    mec='k', mew=.6, label=NOMI[c])
    ax.set_yscale('log')
    ax.set_xticks(xs); ax.set_xticklabels([n for _, n in LIV], rotation=12, fontsize=8)
    ax.set_ylabel('$\\tau_c$  (in unità di $\\langle\\tau\\rangle$)')
    ax.set_title('B) cut-off della distribuzione degli intervalli\n'
                 'Altmann lo introduce senza stimarlo', fontsize=9.5)
    ax.legend(fontsize=7, loc='upper left')

    # ---------------- C: gamma e gamma_A2  (invariato)
    ax = axes[2]
    for i, c in enumerate(ORD):
        g = [T.loc[(c, t)]['gamma_mean'] for t, _ in LIV]
        g2 = [T.loc[(c, t)]['gamma_A2_mean'] for t, _ in LIV]
        ax.plot(xs, g, '-', marker=MARK[i], color=COL[c], ms=6, lw=1.6,
                mec='k', mew=.6, label=NOMI[c])
        ax.plot(xs, g2, '--', marker=MARK[i], color=COL[c], ms=4, lw=1.2, alpha=.55)
    ax.axhline(1, color='grey', ls=':', lw=1.2)
    ax.set_xticks(xs); ax.set_xticklabels([n for _, n in LIV], rotation=12, fontsize=8)
    ax.set_ylabel('$\\hat\\gamma$')
    ax.set_title('C) $\\hat\\gamma$ (pieno) e $\\hat\\gamma_{A2}$ (tratteggio)\n'
                 'il paper prevede $\\hat\\gamma \\geq \\hat\\gamma_{A2}$', fontsize=9.5)
    ax.legend(fontsize=7, loc='upper left')

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'cap5_esponente_coda.{ext}')
    plt.close(fig)
    print(f"\n-> {OUT / 'cap5_esponente_coda.png'}")


if __name__ == '__main__':
    TC = stima()
    print()
    print('tau_c ai quattro livelli, con IC di profilo al 95%')
    print(f"{'corpus':18s}" + ''.join(f'{n:>22s}' for _, n in LIV))
    tc = TC.set_index(['corpus', 'tipo'])
    for c in ORD:
        riga = f'{NOMI[c]:18s}'
        for t, _ in LIV:
            r = tc.loc[(c, t)]
            riga += f"{r['tau_c']:8.1f} [{r['tau_c_lo']:.1f},{r['tau_c_hi']:6.1f}]".rjust(22)
        print(riga)
    figura(TC)
