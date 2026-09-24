"""
Rifà la figura di Heaps e del descrittore R del Capitolo 4.

Unica differenza rispetto alla versione prodotta dal notebook: l'esponente di
Heaps viene indicato con $\\gamma_H$, come nel §2.1 e nella tabella di
notazione, invece che con $\\beta$. Il notebook usa $\\beta$ perché è il simbolo
di [Z]; nella tesi $\\beta$ non è disponibile, e la tabella delle collisioni
registra la corrispondenza.

I dati vengono letti dai CSV già prodotti dal notebook: nulla viene ricalcolato.

Uso:
    py figure_cap4_heaps.py             # in bozze/anteprima
    py figure_cap4_heaps.py --applica   # in Tesi/figure
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RADICE = Path(__file__).resolve().parents[2]
RIS = RADICE / 'Analisi Sweep di temperature' / 'Risultati Sweep'
OUT = (RADICE / 'Tesi' / 'figure') if '--applica' in sys.argv \
    else (RADICE / 'Tesi' / 'bozze' / 'anteprima')
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.25,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False,
})

MODELLI = ['Qwen2.5-1.5B', 'Qwen2.5-3B', 'Qwen2.5-14B',
           'Qwen3.5-2B-Base', 'Qwen3.5-4B-Base']
COLORI_MODELLO = {'Qwen2.5-1.5B': '#1b9e77', 'Qwen2.5-3B': '#d95f02',
                  'Qwen2.5-14B': '#7570b3', 'Qwen3.5-2B-Base': '#e7298a',
                  'Qwen3.5-4B-Base': '#66a61e'}
MARKER_MODELLO = {'Qwen2.5-1.5B': 'o', 'Qwen2.5-3B': 's', 'Qwen2.5-14B': '^',
                  'Qwen3.5-2B-Base': 'D', 'Qwen3.5-4B-Base': 'v'}
COLORE_UMANO = '#000000'
N_TOK = 20000

# Valori pubblicati in [Z] su opere intere.
RIF_Z = dict(R=(0.17, 0.05), beta=(0.801, 0.02))


def plot_vs_T(df, col, ylabel, titolo, ax, umano=None, chiave_rif=None,
              estremo=None):
    """Copiata dal notebook (cella 23)."""
    if chiave_rif in RIF_Z:
        mu, sd = RIF_Z[chiave_rif]
        ax.axhspan(mu - sd, mu + sd, color='#999999', alpha=.16, zorder=0)
        ax.axhline(mu, color='#555555', ls='-.', lw=1.1, zorder=1,
                   label='[Z], opere intere')
    for m in MODELLI:
        d = df[df.modello == m]
        if col not in d.columns or d[col].notna().sum() == 0:
            continue
        g = d.groupby('temperatura')[col]
        mu_, sd_, n = g.mean(), g.std(), g.count()
        sd_ = sd_.where(n > 1, 0.0).fillna(0.0)
        st = dict(color=COLORI_MODELLO[m], marker=MARKER_MODELLO[m])
        ax.errorbar(mu_.index, mu_.values, yerr=sd_.values, capsize=3, ms=5,
                    lw=1.6, label=m, **st)
        if estremo == 'max' and mu_.notna().any():
            ax.axvline(mu_.idxmax(), ls=':', lw=1, alpha=.45, color=st['color'])
    if umano is not None and np.isfinite(umano):
        ax.axhline(umano, color=COLORE_UMANO, lw=1.9, ls='--',
                   label=f'umano, {N_TOK // 1000}k token')
    ax.set_xlabel('temperatura $T$')
    ax.set_ylabel(ylabel)
    ax.set_title(titolo, fontsize=10)
    ax.legend(fontsize=6.5)


def main():
    DF = pd.read_csv(RIS / 'metriche_per_documento.csv')
    UM = pd.to_numeric(pd.read_csv(RIS / 'baseline_umana.csv', index_col=0).iloc[:, 0],
                       errors='coerce')

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    plot_vs_T(DF, 'R', 'descrittore $R$',
              '$R$ = tipi nuovi nella 2ª metà / nella 1ª\n[Z] §4.2',
              axes[0], UM['R'], chiave_rif='R', estremo='max')
    plot_vs_T(DF, 'beta', r'esponente di Heaps $\gamma_H$',
              'Crescita del vocabolario', axes[1], UM['beta'], chiave_rif='beta')
    plot_vs_T(DF, 'TTR', 'rapporto fra tipi e occorrenze',
              'Diversità lessicale (token)', axes[2], UM['TTR'])

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'cap4_heaps_R_vs_T.{ext}')
    plt.close(fig)
    print(f"-> {OUT / 'cap4_heaps_R_vs_T.png'}")


if __name__ == '__main__':
    main()
