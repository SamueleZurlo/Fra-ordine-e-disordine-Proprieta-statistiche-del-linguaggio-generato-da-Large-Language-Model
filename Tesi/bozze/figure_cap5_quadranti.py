"""
Lettura per quadranti del piano burstiness--correlazione (Figura 5.4).

Il capitolo commenta due dei quattro quadranti: quello in basso a destra, vuoto
in tutti i corpora, e quello in alto a sinistra, popolato. Restano fuori i due
che separano meglio la prosa umana da quella generata, cioe' l'alto a destra --
la posizione canonica delle parole chiave in Altmann et al. -- e il basso a
sinistra, dove una sequenza e' insieme piu' regolare del caso e priva di
correlazione a lungo raggio.

Il programma quantifica quel confronto, verifica che non sia un artefatto e
guarda dentro il quadrante basso a sinistra per stabilire quali parole lo
popolino. Legge soltanto sequenze.csv, prodotto dal notebook altmann_tre_corpora:
non ricalcola nulla.

USO
---
    py figure_cap5_quadranti.py              # in bozze/anteprima
    py figure_cap5_quadranti.py --applica    # in Tesi/figure
"""
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    from scipy import stats as sps
except ImportError:
    sps = None

BASE = Path(__file__).resolve().parents[2]
DATI = BASE / 'Wiki' / 'risultati_tre_corpora_v2' / 'dati'
OUT = (BASE / 'Tesi' / 'figure') if '--applica' in sys.argv \
    else (BASE / 'Tesi' / 'bozze' / 'anteprima')
OUT.mkdir(parents=True, exist_ok=True)

ORDINE = ['letterario', 'wikipedia', 'grok_v01', 'grok_oggi']
ETICHETTE = {'letterario': 'Guerra e pace', 'wikipedia': 'Wikipedia',
             'grok_v01': 'Grokipedia v0.1', 'grok_oggi': 'Grokipedia oggi'}
COLORI = {'letterario': '#333333', 'wikipedia': '#0072B2',
          'grok_v01': '#E69F00', 'grok_oggi': '#D55E00'}
QUAD = ['alto-dx', 'alto-sx', 'basso-dx', 'basso-sx']
COLQ = {'alto-dx': '#0072B2', 'alto-sx': '#9ecae1',
        'basso-dx': '#f0c8a0', 'basso-sx': '#D55E00'}
DESCR = {'alto-dx': 'bursty e correlata', 'alto-sx': 'regolare ma correlata',
         'basso-dx': 'bursty ma scorrelata', 'basso-sx': 'regolare e scorrelata'}

plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
                     'font.size': 10, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.axisbelow': True, 'legend.frameon': False,
                     'axes.spines.top': False, 'axes.spines.right': False})


def quadrante(r):
    if r.cv_tau >= 1:
        return 'alto-dx' if r.gamma >= 1 else 'basso-dx'
    return 'alto-sx' if r.gamma >= 1 else 'basso-sx'


def dal_titolo(r):
    """La parola compare nel titolo della voce? Regola oggettiva per separare il
    soggetto dell'articolo dal resto del lessico."""
    if r.corpus == 'letterario':
        return False
    t = re.findall(r"[a-z']+", str(r.titolo).lower())
    w = str(r.sequenza).lower().rstrip("'s")
    return any(w == x or w == x.rstrip("'s") for x in t)


def carica():
    A = pd.read_csv(DATI / 'sequenze.csv')
    S = A[np.isfinite(A.cv_tau) & np.isfinite(A.gamma)].copy()
    S['quadrante'] = S.apply(quadrante, axis=1)
    K = S[S.tipo == 'keyword'].copy()
    K['dal_titolo'] = K.apply(dal_titolo, axis=1)
    K['bs'] = K.quadrante == 'basso-sx'
    return S, K


def rapporto(S, K):
    print('Quota di PAROLE CHIAVE per quadrante:')
    t = pd.crosstab(K.corpus, K.quadrante, normalize='index').reindex(ORDINE)
    t = t.reindex(columns=QUAD).round(3)
    t['n'] = K.groupby('corpus').size().reindex(ORDINE)
    print(t.to_string())

    print('\nCalibrazione: gamma del null model A1 per corpus (atteso 1):')
    print(S.groupby('corpus').gamma_A1.agg(['mean', 'median']).reindex(ORDINE)
          .round(4).to_string())

    print('\nControllo sul numero di eventi: quota basso-sx per fascia di frequenza')
    K = K.copy()
    K['fascia'] = pd.cut(K.n_eventi, [14, 25, 35, 50, 80, 10 ** 6],
                         labels=['15-25', '26-35', '36-50', '51-80', '>80'])
    print(K.pivot_table(index='fascia', columns='corpus', values='bs',
                        aggfunc='mean', observed=True)
          .reindex(columns=ORDINE).round(3).to_string())

    if sps is not None:
        print('\nTest appaiato sulle voci (quota di keyword nel basso-sx):')
        piv = K.pivot_table(index='titolo', columns='corpus', values='bs',
                            aggfunc='mean').dropna(subset=['wikipedia', 'grok_oggi',
                                                           'grok_v01'])
        for c in ['grok_v01', 'grok_oggi']:
            d = piv[c] - piv['wikipedia']
            w = sps.wilcoxon(piv[c], piv['wikipedia'])
            print(f"  {c:10} mediana {d.median():+.3f}  media {d.mean():+.3f}  "
                  f"p = {w.pvalue:.1e}  voci in aumento {int((d>0).sum())}/{len(d)}")

    print('\nComposizione del quadrante basso-sx:')
    b = K[K.bs & (K.corpus != 'letterario')]
    t = b.groupby('corpus').dal_titolo.agg(['size', 'sum'])
    t.columns = ['totale', 'dal_titolo']
    t['non_dal_titolo'] = t.totale - t.dal_titolo
    print(t.reindex([c for c in ORDINE if c != 'letterario']).to_string())
    print('\n  parole non dal titolo, per numero di voci in cui compaiono:')
    for c in ['wikipedia', 'grok_v01', 'grok_oggi']:
        v = Counter(b[(b.corpus == c) & (~b.dal_titolo)].sequenza)
        print(f"    {c:11}: " + ', '.join(f'{w} ({n})' for w, n in v.most_common(8)))

    print('\n  presenza come keyword, su 39 voci:')
    piv = K[K.sequenza.isin(['like', 'amid', 'including', 'though'])].pivot_table(
        index='sequenza', columns='corpus', values='titolo', aggfunc='count',
        fill_value=0)
    print(piv.reindex(columns=[c for c in ORDINE if c in piv.columns])
          .fillna(0).astype(int).to_string())

    if sps is not None:
        print('\nIl divario centrale del capitolo sopravvive alla rimozione?')
        A2 = pd.read_csv(DATI / 'sequenze.csv')
        KK = A2[(A2.tipo == 'keyword') & np.isfinite(A2.cv_tau)]
        cattive = set(K[K.bs & ~K.dal_titolo].sequenza)
        for eti, sub in (('tutte le keyword', KK),
                         ('senza like/amid/including/though',
                          KK[~KK.sequenza.isin(['like', 'amid', 'including', 'though'])]),
                         (f'senza le {len(cattive)} basso-sx non-titolo',
                          KK[~KK.sequenza.isin(cattive)])):
            piv = sub.pivot_table(index='titolo', columns='corpus', values='cv_tau',
                                  aggfunc='mean').dropna(
                subset=['wikipedia', 'grok_oggi', 'grok_v01'])
            r = []
            for c in ['grok_v01', 'grok_oggi']:
                d = piv[c] - piv['wikipedia']
                r.append(f"{c}: {d.median():+.3f} (p={sps.wilcoxon(piv[c], piv['wikipedia']).pvalue:.1e})")
            print(f"  {eti:34} " + '   '.join(r))
    return K


def figura(K):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))

    # --- A: composizione per quadrante -------------------------------------
    ax = axes[0]
    t = pd.crosstab(K.corpus, K.quadrante, normalize='index').reindex(
        ORDINE).reindex(columns=QUAD).fillna(0)
    sinistra = np.zeros(len(ORDINE))
    y = np.arange(len(ORDINE))
    for q in QUAD:
        v = t[q].values
        ax.barh(y, v, left=sinistra, color=COLQ[q], edgecolor='white',
                linewidth=.8, label=f'{q}: {DESCR[q]}')
        for i, (a, b) in enumerate(zip(sinistra, v)):
            if b > 0.045:
                ax.text(a + b / 2, i, f'{100*b:.0f}%', ha='center', va='center',
                        fontsize=8.5,
                        color='white' if q in ('alto-dx', 'basso-sx') else '#333333')
        sinistra += v
    ax.set_yticks(y)
    ax.set_yticklabels([ETICHETTE[c] for c in ORDINE], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel('quota delle parole chiave')
    ax.set_title('A. I quattro quadranti, parole chiave\n'
                 "in Grokipedia il basso-sinistra è tre volte più popolato",
                 fontsize=10)
    ax.legend(fontsize=7.2, loc='lower center', bbox_to_anchor=(.5, -.30), ncol=2)
    ax.grid(axis='y', visible=False)

    # --- B: non e' un effetto della frequenza delle parole -----------------
    ax = axes[1]
    K2 = K.copy()
    bordi = [14, 25, 35, 50, 80, 10 ** 6]
    et = ['15-25', '26-35', '36-50', '51-80', '>80']
    K2['fascia'] = pd.cut(K2.n_eventi, bordi, labels=et)
    t = K2.pivot_table(index='fascia', columns='corpus', values='bs',
                       aggfunc='mean', observed=True).reindex(columns=ORDINE)
    x = np.arange(len(et))
    for i, c in enumerate(ORDINE):
        if c not in t.columns:
            continue
        ax.plot(x, t[c].reindex(et).values, marker='osD^'[i], ms=7, lw=1.8,
                color=COLORI[c], label=ETICHETTE[c])
    # il numero di parole per fascia, perche' la prima e' quasi vuota per
    # Wikipedia e il suo valore non va letto come gli altri
    n = K2.pivot_table(index='fascia', columns='corpus', values='bs',
                       aggfunc='size', observed=True).reindex(columns=ORDINE)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{e}\n(n={int(n.loc[e, "wikipedia"])} | '
                        f'{int(n.loc[e, "grok_oggi"])})' for e in et], fontsize=7.5)
    ax.set_xlabel('occorrenze della parola nel documento\n'
                  '(n = Wikipedia | Grokipedia oggi)', fontsize=9)
    ax.set_ylabel('quota nel quadrante basso-sinistra')
    ax.set_ylim(-0.02, None)
    ax.set_title('B. Non dipende dalla frequenza delle parole\n'
                 'il divario tiene in quattro fasce su cinque', fontsize=10)
    ax.legend(fontsize=7.5)

    for a in axes:
        a.set_box_aspect(1)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'cap5_quadranti.{ext}')
    plt.close(fig)
    print(f"\n-> {OUT / 'cap5_quadranti.png'}")


if __name__ == '__main__':
    S, K = carica()
    K = rapporto(S, K)
    figura(K)
