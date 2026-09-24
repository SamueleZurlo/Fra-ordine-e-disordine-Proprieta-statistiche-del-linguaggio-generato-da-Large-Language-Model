"""
Rifà la figura sugli artefatti di giunzione del Capitolo 4.

La versione prodotta dal notebook riportava la correlazione GREZZA fra numero
di riprese e metriche, che vale $-0.66$ per gli 8-grammi ripetuti e che il
titolo del pannello etichettava «da approfondire». Quella correlazione è però
confusa dalla temperatura: alle basse temperature le riprese sono zero e la
ripetizione è massima, alle temperature intermedie le riprese sono molte e la
ripetizione è nulla, quindi la correlazione marginale misura la temperatura e
non la giunzione.

Qui il confronto è stratificato: a temperatura fissata, i documenti ricuciti da
più segmenti si comportano come quelli generati in un colpo solo?

I dati vengono letti dal CSV già prodotto dal notebook.

Uso:
    py figure_cap4_giunzioni.py             # in bozze/anteprima
    py figure_cap4_giunzioni.py --applica   # in Tesi/figure
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

RADICE = Path(__file__).resolve().parents[2]
RIS = RADICE / 'Analisi Sweep di temperature' / 'Risultati Sweep'
OUT = (RADICE / 'Tesi' / 'figure') if '--applica' in sys.argv \
    else (RADICE / 'Tesi' / 'bozze' / 'anteprima')
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.size': 9.5, 'axes.grid': True, 'grid.alpha': 0.25,
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

C_SENZA, C_CON = '#4363d8', '#e6194B'
METRICHE = [('alpha_DFA', r'$\alpha_{\mathrm{DFA}}$'),
            ('rip_8gram', '8-grammi ripetuti')]


def pannello_riprese(ax, D):
    for m in MODELLI:
        g = D[D.modello == m].groupby('temperatura')['n_continuations'].median()
        ax.plot(g.index, g.values, lw=1.5, ms=5, label=m,
                color=COLORI_MODELLO[m], marker=MARKER_MODELLO[m])
    ax.set_xlabel('temperatura $T$')
    ax.set_ylabel('riprese (mediana)')
    ax.set_title('Dove avvengono le giunzioni', fontsize=10)
    ax.legend(fontsize=6.5)


def pannello_rho(ax, D):
    """Correlazione fra riprese e metrica, dentro ciascuna temperatura,
    contro la correlazione marginale calcolata su tutto l'insieme."""
    Ts = sorted(D.temperatura.unique())
    for (col, etichetta), colore, off in zip(METRICHE, [C_SENZA, C_CON], [-.012, .012]):
        d_tot = D[[col, 'n_continuations']].dropna()
        rho_marg = stats.spearmanr(d_tot.n_continuations, d_tot[col])[0]
        ax.axhline(rho_marg, color=colore, ls='--', lw=1.2, alpha=.8)
        # scosta l'etichetta quando la riga marginale sfiora lo zero
        dy = 0.11 if abs(rho_marg) < 0.10 else 0.0
        ax.text(1.50, rho_marg + dy, f'  marginale\n  {rho_marg:+.2f}',
                color=colore, fontsize=7.5, va='center')
        xs, ys = [], []
        for T in Ts:
            d = D[D.temperatura == T][[col, 'n_continuations']].dropna()
            if len(d) < 5 or d.n_continuations.nunique() < 2 or d[col].nunique() < 2:
                continue
            xs.append(T + off)
            ys.append(stats.spearmanr(d.n_continuations, d[col])[0])
        ax.plot(xs, ys, ls='', marker='o', ms=6, color=colore,
                label=f'{etichetta}, per temperatura')
    ax.axhline(0, color='k', lw=.8)
    ax.set_xlim(0.35, 1.58)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel('temperatura $T$')
    ax.set_ylabel(r'$\rho$ di Spearman con le riprese')
    ax.set_title('Correlazione con le riprese, dentro ciascuna temperatura',
                 fontsize=10)
    ax.legend(fontsize=7, loc='lower left', ncol=2)


def pannello_confronto(ax, D, col, etichetta):
    """Documenti senza riprese contro documenti con almeno una ripresa,
    alla stessa temperatura."""
    for maschera, colore, nome, off in [
            (D.n_continuations == 0, C_SENZA, 'nessuna ripresa', -.012),
            (D.n_continuations > 0, C_CON, 'almeno una ripresa', .012)]:
        d = D[maschera]
        g = d.groupby('temperatura')[col]
        mu, sd, n = g.mean(), g.std().fillna(0.0), g.count()
        tieni = n >= 2
        ax.errorbar(mu.index[tieni] + off, mu.values[tieni],
                    yerr=sd.values[tieni], capsize=3, ms=5, lw=1.5,
                    marker='o', color=colore, label=nome)
        solo = ~tieni & (n > 0)
        ax.plot(mu.index[solo] + off, mu.values[solo], ls='', marker='x',
                ms=6, color=colore, alpha=.8)
    ax.set_xlabel('temperatura $T$')
    ax.set_ylabel(etichetta)
    ax.set_title(f'{etichetta} con e senza giunzioni', fontsize=10)
    ax.legend(fontsize=7.5)


def main():
    D = pd.read_csv(RIS / 'metriche_per_documento.csv')
    D = D[D.n_continuations.notna()]

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0))
    pannello_riprese(axes[0, 0], D)
    pannello_rho(axes[0, 1], D)
    pannello_confronto(axes[1, 0], D, 'alpha_DFA', r'$\alpha_{\mathrm{DFA}}$')
    pannello_confronto(axes[1, 1], D, 'rip_8gram', '8-grammi ripetuti')

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'cap4_artefatti_continuazione.{ext}')
    plt.close(fig)
    print(f"-> {OUT / 'cap4_artefatti_continuazione.png'}")

    # --- numeri citati nel testo -------------------------------------
    print("\nCorrelazione marginale (quella confusa dalla temperatura):")
    for col, _ in METRICHE:
        d = D[[col, 'n_continuations']].dropna()
        r, p = stats.spearmanr(d.n_continuations, d[col])
        print(f"  {col:12s} rho = {r:+.3f}  p = {p:.4f}")

    print("\nA temperatura fissata:")
    for col, _ in METRICHE:
        righe = []
        for T in sorted(D.temperatura.unique()):
            d = D[D.temperatura == T][[col, 'n_continuations']].dropna()
            if len(d) < 5 or d.n_continuations.nunique() < 2 or d[col].nunique() < 2:
                continue
            r, p = stats.spearmanr(d.n_continuations, d[col])
            righe.append((T, r, p))
        R = pd.DataFrame(righe, columns=['T', 'rho', 'p'])
        print(f"  {col:12s} mediana {R.rho.median():+.3f}   "
              f"|rho| max {R.rho.abs().max():.3f}   "
              f"p<0.05 in {(R.p < 0.05).sum()}/{len(R)} temperature")

    n0 = (D.n_continuations == 0).sum()
    print(f"\nDocumenti senza alcuna ripresa: {n0} su {len(D)}")
    d12 = D[D.temperatura >= 1.1]
    print(f"Documenti con T >= 1.1 e 8-grammi ripetuti esattamente nulli: "
          f"{(d12.rip_8gram == 0).sum()} su {len(d12)}")


if __name__ == '__main__':
    main()
