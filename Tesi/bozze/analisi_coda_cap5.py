"""
Due analisi supplementari sulla coda della distribuzione degli intervalli.

  (4) SENSIBILITA' A x_min. La stima riportata in tesi fissa x_min al 90esimo
      percentile degli intervalli. Clauset et al. prescrivono invece di
      sceglierlo minimizzando la distanza di Kolmogorov-Smirnov. Qui si
      confrontano le due strade e alcuni quantili intermedi.

  (2) SUPERFICIE DI VEROSIMIGLIANZA in (mu, tau_c). I due parametri sono
      correlati: la stima non e' un punto ma una cresta. La figura la mostra
      invece di limitarsi a dichiararla.

Il pool degli intervalli viene ricostruito eseguendo le celle del notebook
Wiki/esponente_coda.ipynb, cosi' che i dati siano esattamente gli stessi.

Uso:
    py analisi_coda_cap5.py             # figura in bozze/anteprima
    py analisi_coda_cap5.py --applica   # figura in Tesi/figure
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize, stats as sps

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from rigenera_cap5_stoplist import STOP_EXTRA, inietta   # noqa: E402

RADICE = Path(__file__).resolve().parents[2]
WIKI = RADICE / 'Wiki'
OUT = (RADICE / 'Tesi' / 'figure') if '--applica' in sys.argv \
    else (RADICE / 'Tesi' / 'bozze' / 'anteprima')
OUT.mkdir(parents=True, exist_ok=True)

CELLE = [2, 3, 5, 7, 9, 10]          # config, import, hill, validazione, corpora, pool
ORD = ['letterario', 'wikipedia', 'grok_v01', 'grok_oggi']
NOMI = {'letterario': 'Guerra e pace', 'wikipedia': 'Wikipedia',
        'grok_v01': 'Grokipedia v0.1', 'grok_oggi': 'Grokipedia oggi'}
COL = {'letterario': '#333333', 'wikipedia': '#0072B2',
       'grok_v01': '#E69F00', 'grok_oggi': '#D55E00'}


def carica_pool():
    """Esegue le celle del notebook necessarie a costruire POOL."""
    import os
    prima = os.getcwd()
    os.chdir(WIKI)
    try:
        nb = json.load(open('esponente_coda.ipynb', encoding='utf-8'))
        ns = {'__name__': '__main__', 'STOP_EXTRA_INIETTATA': STOP_EXTRA}
        for i in CELLE:
            src = ''.join(nb['cells'][i]['source'])
            righe = [l for l in src.splitlines()
                     if not l.lstrip().startswith(('%', '!'))]
            src = '\n'.join(righe)
            # la lista di parole funzione va estesa PRIMA che la cella selezioni
            # i bersagli: vedi rigenera_cap5_stoplist.py
            src, _ = inietta(src)
            exec(src, ns)
        return ns['POOL']
    finally:
        os.chdir(prima)


# ------------------------------------------------ modello a potenza troncata
def _Z(mu, xc, xmin):
    v, _ = integrate.quad(lambda t: t ** (-mu) * math.exp(-t / xc), xmin, np.inf,
                          limit=200)
    return v


def nll(par, x, xmin):
    mu, lxc = par
    xc = math.exp(lxc)
    if mu <= -1 or mu > 8 or xc <= 1e-3 or xc > 1e6:
        return 1e12
    Z = _Z(mu, xc, xmin)
    if not np.isfinite(Z) or Z <= 0:
        return 1e12
    return -(-mu * np.sum(np.log(x)) - np.sum(x) / xc - len(x) * math.log(Z))


def fit(x, xmin):
    s = [2.5, math.log(max(float(x.mean()), xmin * 2))]
    r = optimize.minimize(nll, s, args=(x, xmin), method='Nelder-Mead',
                          options=dict(maxiter=1200, xatol=1e-5, fatol=1e-5))
    return float(r.x[0]), float(math.exp(r.x[1])), float(r.fun)


def ks_xmin(x0, quantili=np.arange(50, 96, 2.5)):
    """x_min alla Clauset: si sceglie il valore che minimizza la distanza KS
    fra la coda osservata e la potenza troncata stimata su quella coda."""
    migliore = None
    for q in quantili:
        xmin = float(np.percentile(x0, q))
        x = x0[x0 >= xmin]
        if len(x) < 150:
            continue
        mu, xc, _ = fit(x, xmin)
        Z = _Z(mu, xc, xmin)
        xs = np.sort(x)
        cdf_t = np.array([1 - _Z(mu, xc, v) / Z for v in xs])
        cdf_e = np.arange(1, len(xs) + 1) / len(xs)
        D = float(np.max(np.abs(cdf_t - cdf_e)))
        if migliore is None or D < migliore[0]:
            migliore = (D, q, xmin, mu, xc, len(x))
    return migliore


def main():
    print('Ricostruzione del pool degli intervalli...\n')
    POOL = carica_pool()

    dati = {c: np.concatenate(POOL[(c, 'keyword')]) for c in ORD}

    # =================================================================
    print('\n' + '=' * 74)
    print('(4)  SENSIBILITA DELLA STIMA A x_min  —  livello keyword')
    print('=' * 74)
    print(f"{'corpus':13s} {'criterio':16s} {'q':>5s} {'x_min':>7s} "
          f"{'n':>6s} {'mu':>7s} {'tau_c':>8s}")
    riep = {}
    for c in ORD:
        x0 = dati[c]
        for q in (80, 85, 90, 95):
            xmin = float(np.percentile(x0, q))
            x = x0[x0 >= xmin]
            if len(x) < 150:
                continue
            mu, xc, _ = fit(x, xmin)
            marca = '  <- usato in tesi' if q == 90 else ''
            print(f"{NOMI[c]:13s} {'quantile fisso':16s} {q:5d} {xmin:7.3f} "
                  f"{len(x):6d} {mu:7.3f} {xc:8.3f}{marca}")
            if q == 90:
                riep[c] = dict(mu90=mu, xc90=xc, xmin90=xmin, n90=len(x))
        b = ks_xmin(x0)
        if b:
            D, q, xmin, mu, xc, n = b
            print(f"{NOMI[c]:13s} {'KS (Clauset)':16s} {q:5.1f} {xmin:7.3f} "
                  f"{n:6d} {mu:7.3f} {xc:8.3f}   D = {D:.4f}")
            riep[c].update(muks=mu, xcks=xc, qks=q)
        print()

    print('Scarto fra la stima al quantile 90 e quella con x_min alla Clauset:')
    for c in ORD:
        r = riep[c]
        print(f"  {NOMI[c]:16s} mu {r['mu90']:.3f} -> {r['muks']:.3f} "
              f"({r['muks']-r['mu90']:+.3f})   "
              f"tau_c {r['xc90']:.2f} -> {r['xcks']:.2f}")

    # =================================================================
    print('\n' + '=' * 74)
    print('(2)  SUPERFICIE DI VEROSIMIGLIANZA IN (mu, tau_c)')
    print('=' * 74)

    plt.rcParams.update({
        'figure.dpi': 110, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.25,
        'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False,
    })
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.2, 5.6))

    MU = np.linspace(0.9, 3.4, 52)
    LXC = np.log(np.geomspace(3.0, 45.0, 52))

    ax.axvspan(2, 3, color='#2ca02c', alpha=.08, lw=0)
    for c in ORD:
        x0 = dati[c]
        xmin = float(np.percentile(x0, 90))
        x = x0[x0 >= xmin]
        mu_h, xc_h, f_h = fit(x, xmin)
        S = np.empty((len(LXC), len(MU)))
        for i, l in enumerate(LXC):
            for j, m in enumerate(MU):
                S[i, j] = nll([m, l], x, xmin)
        # regione di confidenza al 95% per due parametri: 2*Delta(-logL) <= 5.99
        ax.contour(MU, np.exp(LXC), 2 * (S - f_h), levels=[5.99],
                   colors=[COL[c]], linewidths=1.9)
        ax.plot([mu_h], [xc_h], marker='o', ms=7, color=COL[c],
                label=f"{NOMI[c]}  ({mu_h:.2f}, {xc_h:.1f})")
        print(f"  {NOMI[c]:16s} massimo in mu = {mu_h:.3f}, tau_c = {xc_h:.2f}")

    ax.axvline(2.4, color='#666666', ls='--', lw=1.3)
    ax.annotate(r'$\mu = 2.4$, assunto da Altmann', xy=(2.4, 41),
                xytext=(2.44, 41), fontsize=8.5, color='#555555', va='top')
    ax.set_yscale('log')
    ax.set_ylim(3.0, 45.0)
    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'$\tau_c$  (in unità di $\langle\tau\rangle$)')
    ax.set_title(r'A: regioni di confidenza al $95\%$, livello keyword'
                 '\n(la banda verde è $2 < \\mu < 3$)', fontsize=10)
    ax.legend(fontsize=8, loc='lower right')

    # --- pannello B: quanto la stima dipende da x_min ------------------
    QQ = [75, 80, 85, 90, 95]
    for c in ORD:
        x0 = dati[c]
        xs, ys = [], []
        for q in QQ:
            xmin = float(np.percentile(x0, q))
            x = x0[x0 >= xmin]
            if len(x) < 150:
                continue
            xs.append(q)
            ys.append(fit(x, xmin)[0])
        bx.plot(xs, ys, marker='o', ms=5, lw=1.7, color=COL[c], label=NOMI[c])
        print(f"  {NOMI[c]:16s} mu al variare di x_min: "
              + ', '.join(f'{v:.2f}' for v in ys))
    bx.axvline(90, color='#666666', ls=':', lw=1.2)
    bx.text(90.2, bx.get_ylim()[0], ' quantile adottato', fontsize=8,
            color='#555555', rotation=90, va='bottom')
    bx.axhspan(2, 3, color='#2ca02c', alpha=.08, lw=0)
    bx.set_xlabel(r'quantile che definisce $x_{\min}$')
    bx.set_ylabel(r'$\mu$')
    bx.set_title(r'B: la stima di $\mu$ dipende da dove comincia la coda',
                 fontsize=10)
    bx.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'cap5_verosimiglianza_coda.{ext}')
    plt.close(fig)
    print(f"\n-> {OUT / 'cap5_verosimiglianza_coda.png'}")


if __name__ == '__main__':
    main()
