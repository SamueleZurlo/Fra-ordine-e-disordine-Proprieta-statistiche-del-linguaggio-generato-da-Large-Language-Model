"""
La riga di Wikipedia nella Figura 5.2: la parola "alexander" e' anomala?

DOMANDA
-------
Nella Fig. 5.2 la cumulata degli intervalli di "alexander" decade in modo molto
piu' ripido di quella di "natasha" e sembra assomigliare a quella di una
lettera. Questo programma stabilisce se sia un effetto di scala del grafico, una
proprieta' vera del testo, o una conseguenza di come la parola viene scelta.

METODO
------
Si ricostruiscono esattamente i quattro documenti rappresentativi del notebook
altmann_tre_corpora (stessa segmentazione del testo letterario, stessa pulizia
di Wikipedia e Grokipedia, stesso N_EFF) e si confrontano tre cose:

1. le cumulate RISCALATE per la media, P(>tau/<tau>), che tolgono l'effetto
   della diversa frequenza delle parole: due processi con la stessa forma ma
   frequenza diversa collassano sulla stessa curva;
2. il coefficiente di variazione rapportato a quello del null model A1 con lo
   stesso numero di eventi e la stessa lunghezza, perche' cv non vale 1 per un
   processo casuale discreto: per la lettera "e", con <tau> ~ 10, il valore
   casuale e' 0.95, non 1;
3. la posizione della parola scelta dentro la distribuzione di cv di tutte le
   keyword del suo corpus, che dice se il rappresentante e' rappresentativo.

USO
---
    py diagnosi_fig52_wikipedia.py              # figura in bozze/anteprima
    py diagnosi_fig52_wikipedia.py --applica    # figura in Tesi/figure
"""
import hashlib
import html as htmlmod
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[2]
WIKI = BASE / 'Wiki'
CACHE_WIKI = WIKI / 'cache_wikipedia'
CACHE_GROK = WIKI / 'cache_grokipedia'
CORPUS_V01 = WIKI / 'risultati_confronto' / 'corpus_v01'
CACHE_LIB = BASE / 'corpora_cache'
DATI = WIKI / 'risultati_tre_corpora_v2' / 'dati'
OUT = (BASE / 'Tesi' / 'figure') if '--applica' in sys.argv \
    else (BASE / 'Tesi' / 'bozze' / 'anteprima')
OUT.mkdir(parents=True, exist_ok=True)

N_EFF = 60_000
N_SEG_LETTERARI = 20
PROMPT_CHARS = 8262
START_PHRASE = "Well, Prince, so Genoa and Lucca"
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/2600/pg2600.txt"
WORD_CHAR = r"[^\W\d_]"

ETICHETTE = {'letterario': 'Guerra e pace', 'wikipedia': 'Wikipedia',
             'grok_v01': 'Grokipedia v0.1', 'grok_oggi': 'Grokipedia oggi'}
COLORI = {'letterario': '#333333', 'wikipedia': '#0072B2',
          'grok_v01': '#E69F00', 'grok_oggi': '#D55E00'}
ORDINE = ['letterario', 'wikipedia', 'grok_v01', 'grok_oggi']

plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.25,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False,
})

# ---------------------------------------------------------------------------
# ricostruzione dei testi, identica al notebook
# ---------------------------------------------------------------------------
APP = re.compile(r"^==+\s*(See also|References|Notes|Citations|Sources|Bibliography|"
                 r"Further reading|External links|Works cited|Footnotes|Explanatory notes|"
                 r"General sources)\s*==+\s*$", re.M | re.I)
INT = re.compile(r"^==+.*?==+\s*$", re.M)
TTS = re.compile(r'<span[^>]*data-tts-block="true"[^>]*>(.*?)</span>', re.S)
GS = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.S)
GE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.S)
ST = re.compile(r"^[ \t]*(?:BOOK\s+[A-Z]+[^\n]*|CHAPTER\s+[IVXLCDM\d]+[^\n]*|"
                r"(?:FIRST|SECOND)\s+EPILOGUE[^\n]*|EPILOGUE[^\n]*|CONTENTS[^\n]*|"
                r"PART\s+[IVXLCDM\d]+[^\n]*|APPENDIX[^\n]*|\d+)[ \t]*$", re.M)
_wc = {}


def pos_from_word(text, w):
    if w not in _wc:
        _wc[w] = re.compile(r"(?<!" + WORD_CHAR + r")" + re.escape(w) +
                            r"(?!" + WORD_CHAR + r")", re.IGNORECASE | re.UNICODE)
    return np.fromiter((m.start() for m in _wc[w].finditer(text)), dtype=np.int64)


def norm(t):
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\n[ \t]+\n", "\n\n", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def safe(t):
    return re.sub(r"[^A-Za-z0-9_]", "", t.replace(" ", "_"))[:60]


def _frag(h):
    h = re.sub(r"(?is)<(script|style|button|svg|nav|footer|aside)[^>]*>.*?</\1>", " ", h)
    h = re.sub(r"<[^>]+>", " ", h)
    h = htmlmod.unescape(h)
    h = re.sub(r"\[\d+\]", " ", h)
    h = re.sub(r"[ \t ]+", " ", h)
    return re.sub(r"\n\s*\n+", "\n\n", h).strip()


def testo(corpus, titolo):
    if corpus == 'wikipedia':
        tx = (CACHE_WIKI / (hashlib.sha256(titolo.encode()).hexdigest()[:14] + '.txt')
              ).read_bytes().decode('utf-8', 'replace')
        m = APP.search(tx)
        if m:
            tx = tx[:m.start()]
        return norm(INT.sub('', tx))
    if corpus == 'grok_v01':
        return norm((CORPUS_V01 / (safe(titolo) + '.txt')
                     ).read_bytes().decode('utf-8', 'replace'))
    if corpus == 'grok_oggi':
        p = (CACHE_GROK / (safe(titolo) + '.html')).read_bytes().decode('utf-8', 'replace')
        b = [x for x in (_frag(y) for y in TTS.findall(p)) if len(x) > 40]
        return norm('\n\n'.join(b))
    raise ValueError(corpus)


def segmenti_letterari():
    fn = CACHE_LIB / (hashlib.sha256(GUTENBERG_URL.encode()).hexdigest()[:12] + '.txt')
    raw = fn.read_bytes().decode('utf-8', 'replace')
    raw = raw.replace('\r\n', '\n').replace('\r', '\n')
    raw = raw[GS.search(raw).end():]
    raw = raw[:GE.search(raw).start()]
    raw = ST.sub('', raw[raw.find(START_PHRASE):])
    raw = re.sub(r'\n[ \t]+\n', '\n\n', raw)
    body = re.sub(r'\n{3,}', '\n\n', raw).strip()[PROMPT_CHARS:]
    return [body[s:s + N_EFF] for s in
            np.linspace(0, max(len(body) - N_EFF, 0), N_SEG_LETTERARI).astype(int)]


# ---------------------------------------------------------------------------
def cv(tau):
    return float(tau.std(ddof=1) / tau.mean())


def cv_A1(M, N, reps=300, seed=1):
    """cv atteso per M posizioni uniformi in N caratteri: NON vale 1, perche' il
    processo e' discreto e senza ripetizione. Per <tau> ~ 10 vale circa 0.95."""
    r = np.random.default_rng(seed)
    v = [cv(np.diff(np.sort(r.choice(N, size=M, replace=False)))) for _ in range(reps)]
    return float(np.mean(v)), float(np.std(v))


def cumulata(tau):
    s = np.sort(tau)
    return s, 1.0 - np.arange(len(s)) / len(s)


def main():
    A = pd.read_csv(DATI / 'sequenze.csv')
    SEG = segmenti_letterari()

    # gli stessi rappresentativi del notebook: documento e keyword a n_eventi massimo
    scelti = []
    for c in ORDINE:
        s = A[(A.corpus == c) & (A.tipo == 'keyword')]
        lab = s.loc[s.n_eventi.idxmax(), 'titolo']
        d = A[(A.corpus == c) & (A.titolo == lab)]
        kw = d[d.tipo == 'keyword'].sort_values('n_eventi', ascending=False).sequenza.iloc[0]
        t = (SEG[int(lab.split('_')[1])] if c == 'letterario' else testo(c, lab))[:N_EFF]
        scelti.append(dict(corpus=c, titolo=lab, kw=kw, testo=t))

    print(f"{'corpus':17} {'documento':22} {'parola':11} {'n':>4} {'<tau>':>7} "
          f"{'cv':>6} {'cv casuale':>11} {'rapporto':>9}")
    print('-' * 98)
    righe = []
    for s in scelti:
        pos = pos_from_word(s['testo'], s['kw'])
        tau = np.diff(pos)
        c0, sd0 = cv_A1(len(pos), N_EFF)
        rif = A[(A.corpus == s['corpus']) & (A.titolo == s['titolo']) &
                (A.sequenza == s['kw'])].cv_tau.iloc[0]
        assert abs(cv(tau) - rif) < 5e-3, (s['kw'], cv(tau), rif)
        righe.append(dict(**s, tau=tau, n=len(pos), cv=cv(tau), cv0=c0,
                          rapporto=cv(tau) / c0))
        print(f"{ETICHETTE[s['corpus']]:17} {s['titolo'][:22]:22} {s['kw']:11} "
              f"{len(pos):>4} {tau.mean():>7.1f} {cv(tau):>6.3f} "
              f"{c0:>7.3f}±{sd0:.3f} {cv(tau)/c0:>9.3f}")

    print()
    lettere = []
    for s in scelti:
        carr = np.array(list(s['testo'].lower()))
        tau = np.diff(np.flatnonzero(carr == 'e'))
        c0, _ = cv_A1(len(tau) + 1, N_EFF, reps=40)
        lettere.append(dict(corpus=s['corpus'], tau=tau, cv=cv(tau), cv0=c0,
                            rapporto=cv(tau) / c0))
        print(f"{ETICHETTE[s['corpus']]:17} {'':22} {'lettera e':11} "
              f"{len(tau)+1:>4} {tau.mean():>7.1f} {cv(tau):>6.3f} "
              f"{c0:>7.3f}       {cv(tau)/c0:>9.3f}")

    # --- la parola scelta e' rappresentativa del suo corpus? ---------------
    K = A[A.tipo == 'keyword'].dropna(subset=['cv_tau'])
    print("\ncv di TUTTE le keyword, per corpus:")
    print(K.groupby('corpus').cv_tau.describe()[['count', '25%', '50%', '75%']]
          .round(3).to_string())
    print("\npercentile della parola scelta dentro il suo corpus:")
    for r in righe:
        k = K[K.corpus == r['corpus']].cv_tau
        print(f"  {ETICHETTE[r['corpus']]:17} {r['kw']:11} cv = {r['cv']:.3f} "
              f"-> percentile {100*(k < r['cv']).mean():4.1f}  "
              f"(mediana del corpus {k.median():.3f})")

    print("\nkeyword dell'articolo scelto, in ordine di frequenza:")
    for r in righe:
        d = A[(A.corpus == r['corpus']) & (A.titolo == r['titolo']) &
              (A.tipo == 'keyword')].sort_values('n_eventi', ascending=False)
        voci = '  '.join(f"{x.sequenza}({int(x.n_eventi)}, cv={x.cv_tau:.2f})"
                         for x in d.itertuples())
        print(f"  {ETICHETTE[r['corpus']]:17} {voci}")

    figura(righe, lettere, K)


def figura(righe, lettere, K):
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.2))

    # --- A: cumulate come nella Fig. 5.2, scala assoluta -------------------
    ax = axes[0]
    for r in righe:
        x, y = cumulata(r['tau'])
        ax.plot(x, y, color=COLORI[r['corpus']], lw=1.8,
                label=f"{r['kw']} ({ETICHETTE[r['corpus']]})")
    x, y = cumulata(lettere[1]['tau'])
    ax.plot(x, y, color='#888888', lw=1.4, ls=':', label='lettera "e" (Wikipedia)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\tau$  [caratteri]')
    ax.set_ylabel(r'$P(>\tau)$')
    ax.set_title('A. Come nella Fig. 5.2\n'
                 'le curve differiscono anche solo per la frequenza', fontsize=10)
    ax.legend(fontsize=7)

    # --- B: le stesse curve riscalate per la media -------------------------
    ax = axes[1]
    for r in righe:
        x, y = cumulata(r['tau'] / r['tau'].mean())
        ax.plot(x, y, color=COLORI[r['corpus']], lw=1.8,
                label=f"{r['kw']} — cv={r['cv']:.2f}")
    x, y = cumulata(lettere[1]['tau'] / lettere[1]['tau'].mean())
    ax.plot(x, y, color='#888888', lw=1.4, ls=':',
            label=f"lettera \"e\" — cv={lettere[1]['cv']:.2f}")
    u = np.logspace(-2, 1.2, 200)
    ax.plot(u, np.exp(-u), color='k', lw=1.2, ls='--',
            label='esponenziale (casuale)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(3e-3, 1.5)
    ax.set_xlabel(r'$\tau / \langle\tau\rangle$')
    ax.set_ylabel(r'$P(>\tau)$')
    ax.set_title('B. Riscalate per la media\n'
                 '"alexander" sta con "freud", non con la lettera', fontsize=10)
    ax.legend(fontsize=7)

    # --- C: dove cade la parola scelta dentro il suo corpus ----------------
    ax = axes[2]
    for i, c in enumerate(ORDINE):
        v = K[K.corpus == c].cv_tau.values
        parts = ax.violinplot([v], positions=[i], widths=.75, showextrema=False)
        for b in parts['bodies']:
            b.set_facecolor(COLORI[c])
            b.set_alpha(.30)
        ax.plot([i - .25, i + .25], [np.median(v)] * 2, color=COLORI[c], lw=2)
        r = righe[i]
        ax.plot([i], [r['cv']], marker='*', ms=17, color=COLORI[c],
                markeredgecolor='k', markeredgewidth=.7, zorder=5)
        ax.annotate(r['kw'], (i, r['cv']), textcoords='offset points',
                    xytext=(9, -3), fontsize=8)
    ax.axhline(1.0, color='k', ls='--', lw=1.2)
    ax.annotate('cv = 1: casuale', (3.35, 1.0), textcoords='offset points',
                xytext=(0, 4), ha='right', fontsize=8)
    ax.set_yscale('log')
    ax.set_xticks(range(len(ORDINE)))
    ax.set_xticklabels([ETICHETTE[c] for c in ORDINE], rotation=18, ha='right',
                       fontsize=8)
    ax.set_ylabel(r'$\sigma_\tau/\langle\tau\rangle$  delle keyword')
    ax.set_title('C. La stella e\' la parola scelta per la Fig. 5.2\n'
                 'nei tre corpora enciclopedici non e\' rappresentativa',
                 fontsize=10)

    for a in axes:
        a.set_box_aspect(1)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'cap5_diagnosi_keyword.{ext}')
    plt.close(fig)
    print(f"\n-> {OUT / 'cap5_diagnosi_keyword.png'}")


if __name__ == '__main__':
    main()
