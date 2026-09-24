# -*- coding: utf-8 -*-
"""L'uso di J e' legittimo? Le affermazioni sull'invarianza reggono?

Il §2.5 afferma tre cose su J = h(ln tau) - h_Poisson(n, <tau>):

  (I)   e' invariante per dilatazione dell'asse dei tempi;
  (II)  vale zero su un processo di Poisson;
  (III) il bias dello stimatore si cancella nella differenza anziche' sommarsi.

E il §5.7 afferma che J e' molto piu' stabile del cv rispetto alla lunghezza di
analisi, confrontando un fattore 3.08 sul cv con uno spostamento di 0.21 bit
su J.

Qui le quattro affermazioni vengono messe alla prova numericamente, usando lo
stimatore ESATTO del notebook, non una riscrittura.

Nessun file della tesi viene modificato.
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
from autoanalisi import WIKI, OUT                 # noqa: E402

SEED = 20260901


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
    MIN_TAU = nj['MIN_TAU']
    print(f'stimatore del notebook | MIN_TAU = {MIN_TAU}, '
          f'N_NULL = {nj["N_NULL"]}\n')

    r = np.random.default_rng(SEED)

    # =================================================== (II) zero su Poisson
    print('=' * 76)
    print('(II)  J DEVE VALERE ZERO SU UN PROCESSO DI POISSON')
    print('=' * 76)
    print(f"{'<tau>':>8}{'n':>7}{'J medio':>12}{'sd':>10}"
          f"{'|J| > 2 sd?':>14}")
    for mu in (10, 30, 100, 200, 900):
        for n in (30, 60, 150, 400):
            vals = []
            for _ in range(60):
                tau = np.maximum(1, np.round(r.exponential(mu, n))
                                 ).astype(np.int64)
                v = J(tau, r)
                if np.isfinite(v):
                    vals.append(v)
            v = np.array(vals)
            sdm = v.std(ddof=1) / np.sqrt(len(v))
            flag = 'SI, distorto' if abs(v.mean()) > 2 * sdm else 'no'
            print(f'{mu:>8}{n:>7}{v.mean():>12.4f}{v.std(ddof=1):>10.4f}'
                  f'{flag:>14}')
    print()

    # ============================================ (I) invarianza per dilatazione
    print('=' * 76)
    print('(I)   INVARIANZA PER DILATAZIONE: tau -> c*tau')
    print('=' * 76)
    print('la legge continua e\' invariante; qui gli intervalli sono INTERI e')
    print('vengono arrotondati, e l\'arrotondamento NON e\' invariante di scala.\n')
    print(f"{'<tau> base':>11}" + ''.join(f'{f"c={c}":>10}'
                                          for c in (1, 2, 5, 10, 50)))
    for mu in (5, 10, 30, 100, 300, 900):
        riga = []
        for c in (1, 2, 5, 10, 50):
            vals = []
            for _ in range(40):
                # stessa forma di partenza, log-normale larga (non Poisson)
                base = np.maximum(1, np.round(
                    r.lognormal(np.log(mu), 1.0, 200))).astype(np.int64)
                tau = np.maximum(1, np.round(base * c)).astype(np.int64)
                v = J(tau, r)
                if np.isfinite(v):
                    vals.append(v)
            riga.append(np.mean(vals))
        print(f'{mu:>11}' + ''.join(f'{x:>10.3f}' for x in riga))
    print('\nse la riga e\' piatta l\'invarianza tiene a quella scala.\n')

    # ================================= (III) il bias si cancella davvero?
    print('=' * 76)
    print('(III) IL BIAS SI CANCELLA? J su leggi di forma NOTA, a n crescente')
    print('=' * 76)
    print('per ciascuna forma J dovrebbe essere costante in n: se deriva,')
    print('la sottrazione del null non ha cancellato il bias.\n')
    forme = {
        'esponenziale (Poisson)': lambda n, m: r.exponential(m, n),
        'log-normale sigma=1': lambda n, m: r.lognormal(np.log(m) - .5, 1.0, n),
        'log-normale sigma=1.5': lambda n, m: r.lognormal(np.log(m) - 1.1, 1.5, n),
        'Pareto mu=2.5': lambda n, m: m * 0.6 * (1 - r.random(n)) ** (-1 / 1.5),
        'quasi deterministica': lambda n, m: m * (1 + 0.15 * r.standard_normal(n)),
    }
    NS = (30, 60, 150, 400, 1000)
    print(f"{'forma':26}" + ''.join(f'{f"n={n}":>10}' for n in NS) + f'{"deriva":>10}')
    for nome, gen in forme.items():
        riga = []
        for n in NS:
            vals = []
            for _ in range(50):
                tau = np.maximum(1, np.round(gen(n, 200))).astype(np.int64)
                v = J(tau, r)
                if np.isfinite(v):
                    vals.append(v)
            riga.append(np.mean(vals))
        print(f'{nome:26}' + ''.join(f'{x:>10.3f}' for x in riga)
              + f'{riga[-1]-riga[0]:>+10.3f}')
    print()

    # ========================== (IV) J vs cv: il confronto del §5.7 e' equo?
    print('=' * 76)
    print('(IV)  IL CONFRONTO "cv cambia di 3.08x, J di 0.21 bit" E\' EQUO?')
    print('=' * 76)
    print('cv e\' un rapporto, J e\' un logaritmo: 0.21 bit corrispondono a un')
    print('fattore 2^0.21 sulla dispersione efficace, non a 0.21.\n')
    for nome, dcv, dJ in [('prince', 3.08, 0.21), ('pierre', 3.49, 0.08)]:
        print(f'  {nome:8} cv x{dcv:.2f}   J {dJ:+.2f} bit '
              f'= dispersione efficace x{2**dJ:.2f}')
    print('\n  il vantaggio resta reale ma vale un fattore '
          f'{3.08/2**0.21:.1f} e {3.49/2**0.08:.1f}, non 15 e 44.\n')

    # ---------------------------------------------------------------- figura
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    ax = axes[0]
    for mu, col in zip((5, 10, 30, 100, 300, 900),
                       plt.cm.viridis(np.linspace(0, .9, 6))):
        cs = np.array([1, 2, 5, 10, 50])
        vals = []
        for c in cs:
            v = []
            for _ in range(40):
                base = np.maximum(1, np.round(
                    r.lognormal(np.log(mu), 1.0, 200))).astype(np.int64)
                tau = np.maximum(1, np.round(base * c)).astype(np.int64)
                x = J(tau, r)
                if np.isfinite(x):
                    v.append(x)
            vals.append(np.mean(v))
        ax.plot(cs, vals, '-o', color=col, label=f'$\\langle\\tau\\rangle$={mu}')
    ax.set_xscale('log')
    ax.set_xlabel('fattore di dilatazione $c$')
    ax.set_ylabel('$J$ [bit]')
    ax.set_title("A) invarianza per dilatazione\n"
                 "tiene solo dove $\\langle\\tau\\rangle$ e' grande", fontsize=10)
    ax.legend(fontsize=7)

    ax = axes[1]
    for nome, gen in forme.items():
        vals = []
        for n in NS:
            v = []
            for _ in range(50):
                tau = np.maximum(1, np.round(gen(n, 200))).astype(np.int64)
                x = J(tau, r)
                if np.isfinite(x):
                    v.append(x)
            vals.append(np.mean(v))
        ax.plot(NS, vals, '-o', ms=4, label=nome)
    ax.set_xscale('log')
    ax.axhline(0, color='grey', ls='--', lw=1.2)
    ax.set_xlabel('numero di intervalli $n$')
    ax.set_ylabel('$J$ [bit]')
    ax.set_title('B) residuo di bias in funzione di $n$', fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / 'verifica_J.png', dpi=150)
    plt.close(fig)
    print(f'-> {OUT / "verifica_J.png"}')


if __name__ == '__main__':
    main()
