"""
Rifa' la Figura 5.3 e i numeri che ne dipendono escludendo dal livello delle
parole chiave le parole funzione che il filtro del notebook non intercetta.

PERCHE'
-------
Il filtro delle parole chiave ammette qualunque parola di almeno quattro
caratteri che non compaia nella lista di parole funzione del notebook. Quella
lista non contiene alcuni connettivi e attenuatori inglesi molto comuni, fra cui
"like", "amid", "including" e "though". In Wikipedia questo non ha conseguenze,
perche' quei termini non entrano mai fra le sette parole piu' frequenti di una
voce; in Grokipedia sì, e "like" vi entra in 29 voci su 39. Il livello delle
parole chiave di Grokipedia contiene quindi materiale che a rigore appartiene al
livello delle parole funzione, e poiche' i connettivi hanno burstiness bassa il
divario misurato a quel livello ne risulta gonfiato.

CRITERIO
--------
L'esclusione e' definita per CLASSE GRAMMATICALE, non a partire dai risultati:
preposizioni, congiunzioni subordinanti, avverbi connettivi e attenuatori,
quantificatori. La stessa lista si applica a tutti e quattro i corpora. Nessuna
parola di contenuto viene toccata: "economic", "military" e "northern" restano
al loro posto anche se popolano il quadrante basso-sinistra.

Il test interno della Figura 5.3A appaia ogni parola chiave con il proprio
controllo a frequenza appaiata secondo l'ordine in cui compaiono nel file.
Rimuovendo una parola chiave va quindi rimosso anche il suo controllo, o
l'appaiamento slitta: il programma lo fa esplicitamente.

LIMITE DICHIARATO
-----------------
Le parole rimosse non vengono sostituite dalla successiva in ordine di
frequenza, perche' sequenze.csv contiene solo i bersagli selezionati e la
sostituzione richiederebbe di rimisurare i testi. Le liste di parole chiave di
Grokipedia risultano quindi piu' corte di quelle di Wikipedia. E' una scelta
conservativa: toglie materiale al corpus che ne ha di piu' e non ne aggiunge.

USO
---
    py figure_cap5_senza_connettivi.py              # in bozze/anteprima
    py figure_cap5_senza_connettivi.py --applica    # in Tesi/figure
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import stats as sps
except ImportError:
    sps = None

BASE = Path(__file__).resolve().parents[2]
DATI = BASE / 'Wiki' / 'risultati_tre_corpora' / 'dati'
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

# --- parole funzione mancanti dalla lista del notebook ----------------------
# preposizioni e congiunzioni
STOP_EXTRA = set("""amid amidst among amongst including excluding regarding
concerning despite throughout toward towards within without beyond across
around along besides versus like unlike alongside
although though whereas whether unless since thereby whilst
however moreover furthermore therefore thus hence nevertheless nonetheless
meanwhile otherwise instead also still even just rather quite often sometimes
usually generally particularly especially largely mainly mostly notably
similarly likewise accordingly subsequently previously later earlier well
many much several various certain given another every
became become becomes remains remained included includes involving""".split())

plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
                     'font.size': 10, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.axisbelow': True, 'legend.frameon': True})


# ---------------------------------------------------------------------------
def filtra(A, escluse):
    """Toglie le parole chiave escluse e i controlli appaiati corrispondenti.

    L'appaiamento fra i due livelli e' posizionale: la i-esima keyword del
    documento va con il i-esimo controllo. Va quindi rimossa la coppia intera.
    """
    tieni = pd.Series(True, index=A.index)
    for _, doc in A.groupby(['corpus', 'titolo'], sort=False):
        k = doc[doc.tipo == 'keyword'].sort_index()
        m = doc[doc.tipo == 'appaiata'].sort_index()
        for i, (ik, w) in enumerate(zip(k.index, k.sequenza)):
            if str(w).lower() in escluse:
                tieni[ik] = False
                if i < len(m):
                    tieni[m.index[i]] = False
    return A[tieni].copy()


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return (max(c - h, 0.0), min(c + h, 1.0))


def test_interno(A, corpus, col):
    """Figura 5.3A: la keyword batte il proprio controllo a frequenza appaiata?"""
    sub = A[A.corpus == corpus]
    v = tot = 0
    for _, doc in sub.groupby('titolo'):
        k = doc[doc.tipo == 'keyword'].sort_index()
        m = doc[doc.tipo == 'appaiata'].sort_index()
        for ik, im in zip(k.index, m.index):
            a, b = A.at[ik, col], A.at[im, col]
            if np.isfinite(a) and np.isfinite(b):
                tot += 1
                v += int(a > b)
    lo, hi = wilson(v, tot)
    p = float(sps.binomtest(v, tot, 0.5).pvalue) if (sps and tot) else np.nan
    return dict(vittorie=v, confronti=tot, frazione=v / tot if tot else np.nan,
                ic_lo=lo, ic_hi=hi, p=p)


def per_doc(A, c, tipo, col):
    s = A[(A.corpus == c) & (A.tipo == tipo)]
    return s.groupby('titolo')[col].mean().dropna()


def livelli(A):
    righe = []
    for c in ORDINE:
        for tipo, _ in LIV:
            cv, ga = per_doc(A, c, tipo, 'cv_tau'), per_doc(A, c, tipo, 'gamma')
            righe.append(dict(corpus=c, tipo=tipo, n_doc=len(cv),
                              cv=cv.mean(), cv_sd=cv.std(),
                              gamma=ga.mean(), gamma_sd=ga.std()))
    return pd.DataFrame(righe)


def appaiato(A, c1, c2, tipo, col):
    a, b = per_doc(A, c1, tipo, col), per_doc(A, c2, tipo, col)
    idx = a.index.intersection(b.index)
    a, b = a[idx], b[idx]
    if len(a) < 5:
        return dict(n=len(a), mediana=np.nan, p=np.nan)
    p = float(sps.wilcoxon(a, b).pvalue) if sps else np.nan
    return dict(n=len(a), mediana=float(np.median(a - b)), p=p)


# ---------------------------------------------------------------------------
def figura(A, LEV, PT, suffisso, titolo_extra):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.9))

    ax = axes[0]
    y = np.arange(len(ORDINE))
    for i, c in enumerate(ORDINE):
        x = PT[c]
        err = np.array([[max(x['frazione'] - x['ic_lo'], 0)],
                        [max(x['ic_hi'] - x['frazione'], 0)]])
        ax.errorbar([x['frazione']], [i], xerr=err, fmt='o', ms=10, capsize=5,
                    color=COLORI[c], lw=2)
    ax.set_yticks(y)
    ax.set_yticklabels([ETICHETTE[c] for c in ORDINE], fontsize=9)
    ax.set_ylim(-.6, len(ORDINE) - .4)
    ax.set_xlim(0, 1)
    ax.axvline(.5, color='grey', ls=':', lw=1.4)
    ax.set_xlabel('frazione di coppie con keyword più bursty')
    ax.set_title("A) l'effetto di Altmann esiste?\n(0.5 = nessun effetto)", fontsize=10)

    ax = axes[1]
    tipi = [t for t, _ in LIV]
    xs = np.arange(len(tipi))
    for i, c in enumerate(ORDINE):
        s = LEV[LEV.corpus == c].set_index('tipo').reindex(tipi)
        ax.errorbar(xs + .07 * (i - 1.5), s.cv, yerr=s.cv_sd, marker='osD^'[i],
                    ms=7, lw=1.8, capsize=3.5, color=COLORI[c], label=ETICHETTE[c])
    ax.axhline(1, color='grey', ls=':', lw=1.2)
    ax.set_xticks(xs)
    ax.set_xticklabels([n for _, n in LIV], rotation=12, fontsize=8)
    ax.set_ylabel(r'$\sigma_\tau/\langle\tau\rangle$')
    ax.set_title('B) burstiness per livello linguistico', fontsize=10)
    ax.legend(fontsize=7.5)

    ax = axes[2]
    dati, et, cc = [], [], []
    rng = np.random.default_rng(20260814)
    for c in ORDINE:
        v = per_doc(A, c, 'keyword', 'cv_tau')
        if len(v):
            dati.append(v.values)
            et.append(f'{ETICHETTE[c]}\n({len(v)})')
            cc.append(c)
    bp = ax.boxplot(dati, tick_labels=et, showmeans=True, widths=.55,
                    patch_artist=True)
    for b, c in zip(bp['boxes'], cc):
        b.set(facecolor=COLORI[c], alpha=.30)
    for i, v in enumerate(dati):
        ax.scatter(np.full(len(v), i + 1) + rng.normal(0, .05, len(v)), v, s=14,
                   color='k', alpha=.45, zorder=4)
    ax.axhline(1, color='grey', ls=':', lw=1.2)
    ax.set_ylabel(r'$\sigma_\tau/\langle\tau\rangle$  delle keyword')
    ax.set_title('C) burstiness delle keyword, testo per testo', fontsize=10)
    ax.tick_params(axis='x', labelsize=8)

    fig.suptitle('Il protocollo di Altmann sui quattro corpora' + titolo_extra,
                 fontweight='bold')
    fig.tight_layout()
    nome = f'cap5_tre_corpora{suffisso}'
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{nome}.{ext}')
    plt.close(fig)
    print(f'-> {OUT / (nome + ".png")}')


def riassunto(A, eti):
    LEV = livelli(A)
    PT = {c: test_interno(A, c, 'cv_tau') for c in ORDINE}
    PTg = {c: test_interno(A, c, 'gamma') for c in ORDINE}
    print(f'\n===== {eti} =====')
    print('quota di confronti vinti dalle keyword (Fig. 5.3A):')
    for c in ORDINE:
        x, g = PT[c], PTg[c]
        print(f"  {ETICHETTE[c]:18} cv: {x['vittorie']:>3}/{x['confronti']:>3} = "
              f"{x['frazione']:.3f} [{x['ic_lo']:.3f}, {x['ic_hi']:.3f}]   "
              f"gamma: {g['frazione']:.3f}")
    print('\ncv per livello (Fig. 5.3B):')
    print(f"  {'livello':20}" + ''.join(f'{ETICHETTE[c][:15]:>17}' for c in ORDINE))
    for tipo, nome in LIV:
        v = [LEV[(LEV.corpus == c) & (LEV.tipo == tipo)].cv.iloc[0] for c in ORDINE]
        print(f'  {nome:20}' + ''.join(f'{x:>17.3f}' for x in v))
    print('\ndifferenza appaiata per livello (Wilcoxon, mediana delle differenze):')
    for c1 in ('grok_v01', 'grok_oggi'):
        for tipo, nome in LIV:
            r = appaiato(A, c1, 'wikipedia', tipo, 'cv_tau')
            print(f"  {c1:10} vs wikipedia  {nome:20} n={r['n']:>3}  "
                  f"{r['mediana']:+.3f}  p = {r['p']:.1e}")
    return LEV, PT


def main():
    A = pd.read_csv(DATI / 'sequenze.csv')
    A = A[np.isfinite(A.cv_tau)].copy()

    K = A[A.tipo == 'keyword']
    colpite = sorted(set(K.sequenza.str.lower()) & STOP_EXTRA)
    print('parole funzione trovate fra le parole chiave: ' + ', '.join(colpite))
    print('\nrighe di tipo keyword rimosse, per corpus:')
    t = K[K.sequenza.str.lower().isin(STOP_EXTRA)].groupby('corpus').size()
    tot = K.groupby('corpus').size()
    for c in ORDINE:
        print(f'  {ETICHETTE[c]:18} {int(t.get(c, 0)):>3} su {int(tot[c])} '
              f'({100*t.get(c, 0)/tot[c]:.1f}%)')

    LEV0, PT0 = riassunto(A, 'ORIGINALE')
    B = filtra(A, STOP_EXTRA)
    LEV1, PT1 = riassunto(B, 'SENZA LE PAROLE FUNZIONE')

    figura(A, LEV0, PT0, '_originale', '')
    figura(B, LEV1, PT1, '_senza_connettivi',
           '\nlivello delle parole chiave depurato dalle parole funzione')

    print('\n===== SPOSTAMENTI =====')
    for tipo, nome in LIV:
        for c in ORDINE:
            a = LEV0[(LEV0.corpus == c) & (LEV0.tipo == tipo)].cv.iloc[0]
            b = LEV1[(LEV1.corpus == c) & (LEV1.tipo == tipo)].cv.iloc[0]
            if abs(a - b) > 5e-4:
                print(f'  cv  {nome:20} {ETICHETTE[c]:18} {a:.3f} -> {b:.3f} '
                      f'({b-a:+.3f})')
    for c in ORDINE:
        a, b = PT0[c]['frazione'], PT1[c]['frazione']
        if abs(a - b) > 5e-4:
            print(f'  quota vinta {ETICHETTE[c]:18} {a:.3f} -> {b:.3f} ({b-a:+.3f})')


if __name__ == '__main__':
    main()
