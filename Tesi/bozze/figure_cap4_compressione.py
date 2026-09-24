"""
Rifà la figura delle misure di compressione del Capitolo 4.

La pendenza della curva di prefisso è stata rimossa dalla figura. Quel numero
non è normalizzato sulla lunghezza del documento: è la derivata per frase, e il
numero di frasi crolla di un fattore 170 lungo lo sweep, cosicché il pannello
misurava in sostanza l'inverso del numero di frasi. La pendenza è inoltre decisa
dal primo punto della curva, che su prefissi corti risente dell'overhead di
gzip. Al suo posto compare l'entropia normalizzata dei caratteri.

Rispetto alla versione prodotta dal notebook cambia soltanto il sesto pannello,
il piano entropia--comprimibilità. Là l'asse verticale era lineare e copriva
l'intervallo da 0 a 0.81: i 57 documenti con $R_{gzip}$ fra 0.007 e 0.020 —
un quarto del totale — finivano schiacciati sulla linea dello zero, dando
l'impressione di un rapporto di compressione nullo, che non esiste. Con l'asse
logaritmico quei documenti si aprono e si vede che occupano un intervallo di
entropia molto ampio.

I punti sono inoltre colorati per temperatura anziché per modello, così il
percorso lungo $T$ annunciato dal titolo del pannello è effettivamente visibile;
la forma del marcatore continua a distinguere i cinque modelli.

I dati vengono letti dai CSV già prodotti dal notebook: la figura non ricalcola
nulla.

Uso:
    py figure_cap4_compressione.py             # in bozze/anteprima
    py figure_cap4_compressione.py --applica   # in Tesi/figure
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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


def stile_modello(m):
    return dict(color=COLORI_MODELLO[m], marker=MARKER_MODELLO[m])


def plot_vs_T(df, col, ylabel, titolo, ax, umano=None, logy=False, legenda=True):
    """Copiata dal notebook (cella 23): una curva per modello in funzione di T."""
    for m in MODELLI:
        d = df[df.modello == m]
        if col not in d.columns or d[col].notna().sum() == 0:
            continue
        g = d.groupby('temperatura')[col]
        mu, sd, n = g.mean(), g.std(), g.count()
        sd = sd.where(n > 1, 0.0).fillna(0.0)
        ax.errorbar(mu.index, mu.values, yerr=sd.values, capsize=3, ms=5,
                    lw=1.6, label=m, **stile_modello(m))
    if umano is not None and np.isfinite(umano):
        ax.axhline(umano, color=COLORE_UMANO, lw=1.9, ls='--',
                   label=f'umano, {N_TOK // 1000}k token')
    ax.set_xlabel('temperatura $T$')
    ax.set_ylabel(ylabel)
    ax.set_title(titolo, fontsize=10)
    if logy:
        ax.set_yscale('log')
    if legenda:
        ax.legend(fontsize=6.5)


def pannello_piano(ax, DF, UM):
    """Piano entropia--comprimibilità con asse verticale logaritmico e punti
    colorati per temperatura."""
    cmap = plt.get_cmap('viridis')
    tmin, tmax = DF.temperatura.min(), DF.temperatura.max()
    norm = plt.Normalize(tmin, tmax)

    for m in MODELLI:
        d = DF[DF.modello == m]
        ax.scatter(d.H_parola_norm, d.R_gzip, s=34, alpha=.85,
                   c=cmap(norm(d.temperatura)), marker=MARKER_MODELLO[m],
                   edgecolors='none')
    ax.scatter([UM['H_parola_norm']], [UM['R_gzip']], s=260, marker='*',
               color=COLORE_UMANO, zorder=6)
    ax.annotate('umano', (UM['H_parola_norm'], UM['R_gzip']),
                textcoords='offset points', xytext=(-6, 10),
                ha='right', fontsize=9)

    ax.set_yscale('log')
    ax.set_xlabel('entropia normalizzata a livello di parola')
    ax.set_ylabel(r'$R_{\mathrm{gzip}}$')
    ax.set_title('Entropia del vocabolario vs comprimibilità\n'
                 "(asse della Fig. 1A di [C], percorso lungo $T$)", fontsize=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cax = ax.inset_axes([0.06, 0.90, 0.34, 0.035])
    cb = ax.figure.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_label('temperatura $T$', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    voci = [Line2D([], [], ls='', marker=MARKER_MODELLO[m], color='#666666',
                   ms=6, label=m) for m in MODELLI]
    voci.append(Line2D([], [], ls='', marker='*', color=COLORE_UMANO, ms=12,
                       label='umano'))
    ax.legend(handles=voci, fontsize=6.5, loc='lower right')


def main():
    DF = pd.read_csv(RIS / 'metriche_per_documento.csv')
    UM = pd.to_numeric(pd.read_csv(RIS / 'baseline_umana.csv', index_col=0).iloc[:, 0],
                       errors='coerce')

    # Sei pannelli quadrati e uguali in griglia 2x3. La barra dei colori
    # dell'ultimo e' un inserto interno, cosi' non sottrae spazio all'asse e
    # tutti i riquadri restano della stessa dimensione.
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 10.4))
    plot_vs_T(DF, 'R_gzip', r'$R_{\mathrm{gzip}} = C(x)/|x|$',
              'Compression ratio [C] eq.(4)', axes[0, 0], UM['R_gzip'])
    plot_vs_T(DF, 'compr_condizionata', 'compressione condizionata',
              'Prevedibilità dal contesto [C]', axes[0, 1], UM['compr_condizionata'])
    plot_vs_T(DF, 'NCD_shuffle', 'NCD(originale, permutato)',
              "Contributo dell'ordine delle parole [C]", axes[0, 2], UM['NCD_shuffle'])
    plot_vs_T(DF, 'dist_rip_media', 'distanza media fra ripetizioni',
              'Struttura delle ripetizioni [C]', axes[1, 0], UM['dist_rip_media'],
              logy=True)
    plot_vs_T(DF, 'H_char_norm', 'entropia normalizzata dei caratteri',
              'Entropia a livello di carattere [C]', axes[1, 1], UM['H_char_norm'])
    pannello_piano(axes[1, 2], DF, UM)

    for a in axes.ravel():
        a.set_box_aspect(1)

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'cap4_compressione.{ext}')
    plt.close(fig)
    print(f"-> {OUT / 'cap4_compressione.png'}")

    # --- numeri citati nel testo ------------------------------------------
    bassi = DF[DF.R_gzip < 0.02]
    print(f"\nDocumenti con R_gzip < 0.02: {len(bassi)} su {len(DF)}")
    print(f"  R_gzip minimo assoluto      : {DF.R_gzip.min():.4f}")
    print(f"  loro entropia di parola     : {bassi.H_parola_norm.min():.3f} "
          f"– {bassi.H_parola_norm.max():.3f}")
    print(f"  loro numero di tipi         : {int(bassi.n_tipi.min())} "
          f"– {int(bassi.n_tipi.max())}")
    print(f"  loro TTR sulle parole       : {bassi.TTR_parole.min():.4f} "
          f"– {bassi.TTR_parole.max():.4f}")


if __name__ == '__main__':
    main()
