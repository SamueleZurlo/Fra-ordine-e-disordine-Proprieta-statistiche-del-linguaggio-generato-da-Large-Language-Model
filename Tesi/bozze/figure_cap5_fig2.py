"""
Rifa' la Figura 5.2 (analogo della Fig. 2 di Altmann et al.) cambiando la regola
con cui si sceglie la parola rappresentativa.

PERCHE'
-------
Il notebook sceglieva il documento con la keyword piu' ricorrente e, dentro di
esso, la keyword con piu' occorrenze. In un romanzo quella regola pesca il nome
di un personaggio, che compare solo nelle scene in cui il personaggio c'e'; in
un'enciclopedia pesca il soggetto della voce, che ricorre in quasi ogni frase.
La figura finiva cosi' per confrontare le due code opposte delle rispettive
distribuzioni: "natasha" stava al 96esimo percentile di burstiness del corpus
letterario, "alexander" all'1.8esimo di Wikipedia e "freud" al 3.3esimo di
Grokipedia. Il sottotitolo annunciava code larghe e gamma_A2 vicino a gamma, e
la cosa valeva per una riga su quattro.

La regola nuova sceglie, in ciascun corpus, la keyword con cv piu' vicino alla
MEDIANA del corpus, fra quelle con almeno N_MIN occorrenze perche' la cumulata
sia leggibile. La figura illustra cosi' il caso tipico invece di due estremi
opposti, e resta confrontabile riga per riga.

Tutto il resto -- pulizia dei testi, N_EFF, lag, null model A1 e A2, finestra di
fit -- e' identico al notebook altmann_tre_corpora.

USO
---
    py figure_cap5_fig2.py              # in bozze/anteprima
    py figure_cap5_fig2.py --applica    # sovrascrive Tesi/figure/cap5_altmann_fig2.png
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
APPLICA = '--applica' in sys.argv
OUT = (BASE / 'Tesi' / 'figure') if APPLICA else (BASE / 'Tesi' / 'bozze' / 'anteprima')
OUT.mkdir(parents=True, exist_ok=True)

# --- parametri identici al notebook ----------------------------------------
N_EFF = 60_000
N_SEG_LETTERARI = 20
PROMPT_CHARS = 8262
START_PHRASE = "Well, Prince, so Genoa and Lucca"
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/2600/pg2600.txt"
LAG_MIN, LAG_MAX, N_LAGS = 4, 4000, 40
FIT_RANGE = (100, 2000)
N_NULL_REPS = 3
RANDOM_SEED = 20260814
WORD_CHAR = r"[^\W\d_]"
LAGS = np.unique(np.logspace(np.log10(LAG_MIN), np.log10(LAG_MAX), N_LAGS).astype(int))

# --- parametro della regola nuova ------------------------------------------
N_MIN = 40      # occorrenze minime perche' la cumulata sia leggibile

ORDINE = ['letterario', 'wikipedia', 'grok_v01', 'grok_oggi']
ETICHETTE = {'letterario': 'Guerra e pace', 'wikipedia': 'Wikipedia',
             'grok_v01': 'Grokipedia v0.1', 'grok_oggi': 'Grokipedia oggi'}

plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
                     'font.size': 10, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.axisbelow': True, 'legend.frameon': True})

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


def testo(corpus, titolo, SEG):
    if corpus == 'letterario':
        return SEG[int(titolo.split('_')[1])]
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
    p = (CACHE_GROK / (safe(titolo) + '.html')).read_bytes().decode('utf-8', 'replace')
    b = [x for x in (_frag(y) for y in TTS.findall(p)) if len(x) > 40]
    return norm('\n\n'.join(b))


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
# nucleo di misura, identico al notebook
# ---------------------------------------------------------------------------
def seme(*p):
    h = hashlib.sha256("|".join(map(str, p)).encode("utf-8")).hexdigest()
    return (int(h[:8], 16) ^ RANDOM_SEED) % (2 ** 32)


def transport_sigma2(pos, N):
    x = np.zeros(N, dtype=np.float64)
    if len(pos):
        x[pos[pos < N]] = 1.0
    C = np.concatenate(([0.0], np.cumsum(x)))
    out = np.full(len(LAGS), np.nan)
    for i, t in enumerate(LAGS):
        if t >= N // 4:
            continue
        out[i] = (C[t:] - C[:-t]).var()
    return out


def fit_gamma(s2, lo, hi, min_pts=5):
    s2 = np.asarray(s2, float)
    m = np.isfinite(s2) & (s2 > 0) & (LAGS >= lo) & (LAGS <= hi)
    if m.sum() < min_pts:
        return np.nan
    X, Y = np.log10(LAGS[m].astype(float)), np.log10(s2[m])
    return float(np.polyfit(X, Y, 1)[0])


def null_A1(pos, N, r):
    M = len(pos)
    return np.sort(r.choice(N, size=min(M, N), replace=False)) if M else pos


def null_A2(pos, N, r):
    tau = np.diff(pos)
    if len(tau) < 2:
        return pos
    tau = r.permutation(tau)
    new = np.concatenate(([pos[0]], pos[0] + np.cumsum(tau)))
    return new[new < N]


def transport_A(pos, N, r, n_null=N_NULL_REPS):
    s2 = transport_sigma2(pos, N)
    a1 = np.zeros_like(s2)
    a2 = np.zeros_like(s2)
    for _ in range(n_null):
        a1 += transport_sigma2(null_A1(pos, N, r), N) / n_null
        a2 += transport_sigma2(null_A2(pos, N, r), N) / n_null
    return dict(s2=s2, s2_A1=a1, s2_A2=a2, tau=np.diff(pos))


# ---------------------------------------------------------------------------
def scegli(A, corpus):
    """La keyword con cv piu' vicino alla mediana del corpus, fra quelle con
    almeno N_MIN occorrenze. La mediana di riferimento e' calcolata sulla stessa
    popolazione da cui si sceglie, cosi' il criterio e' interno e verificabile."""
    K = A[(A.corpus == corpus) & (A.tipo == 'keyword')].dropna(subset=['cv_tau'])
    pool = K[K.n_eventi >= N_MIN]
    if not len(pool):
        pool = K
    med = pool.cv_tau.median()
    r = pool.loc[(pool.cv_tau - med).abs().idxmin()]
    perc = 100 * (K.cv_tau < r.cv_tau).mean()
    return dict(titolo=r.titolo, kw=r.sequenza, cv=float(r.cv_tau),
                n=int(r.n_eventi), mediana=float(med), percentile=perc,
                n_pool=len(pool), n_tot=len(K))


def main():
    A = pd.read_csv(DATI / 'sequenze.csv')
    SEG = segmenti_letterari()

    scelte, CURVE = {}, {}
    print(f"regola nuova: keyword con cv piu' vicino alla mediana del corpus, "
          f"fra quelle con n >= {N_MIN}\n")
    print(f"{'corpus':17} {'documento':24} {'parola':13} {'n':>4} {'cv':>6} "
          f"{'mediana':>8} {'perc.':>6}")
    for c in ORDINE:
        s = scegli(A, c)
        t = testo(c, s['titolo'], SEG)[:N_EFF]
        d = A[(A.corpus == c) & (A.titolo == s['titolo'])]
        let = d[d.tipo == 'lettera'].sort_values('n_eventi', ascending=False).sequenza.iloc[0]
        r = np.random.default_rng(seme(c, s['titolo'], 'curve'))
        carr = np.array(list(t.lower()))
        pos_kw = pos_from_word(t, s['kw'])
        # controllo: il cv ricalcolato deve coincidere con quello del CSV
        cvv = float(np.diff(pos_kw).std(ddof=1) / np.diff(pos_kw).mean())
        assert abs(cvv - s['cv']) < 5e-3, (s['kw'], cvv, s['cv'])
        CURVE[c] = dict(lettera=let, keyword=s['kw'], titolo=s['titolo'],
                        percentile=s['percentile'],
                        L=transport_A(np.flatnonzero(carr == let), N_EFF, r),
                        K=transport_A(pos_kw, N_EFF, r))
        scelte[c] = s
        print(f"{ETICHETTE[c]:17} {s['titolo'][:24]:24} {s['kw']:13} {s['n']:>4} "
              f"{s['cv']:>6.2f} {s['mediana']:>8.2f} {s['percentile']:>5.1f}%")

    fig, axes = plt.subplots(len(ORDINE), 4, figsize=(16.5, 3.5 * len(ORDINE)),
                             squeeze=False)
    for i, c in enumerate(ORDINE):
        cur = CURVE[c]
        for j, (chiave, nome) in enumerate([('L', f'lettera "{cur["lettera"]}"'),
                                            ('K', f'parola "{cur["keyword"]}"')]):
            d = cur[chiave]
            ax = axes[i][2 * j]
            s = np.sort(d['tau'])
            ax.plot(s, 1.0 - np.arange(len(s)) / len(s), color='k', lw=1.8)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'$\tau$')
            ax.set_ylabel(r'$P(>\tau)$')
            cv = d['tau'].std(ddof=1) / d['tau'].mean()
            extra = (f"  ({cur['percentile']:.0f}° percentile del corpus)"
                     if chiave == 'K' else '')
            ax.set_title(f'{ETICHETTE[c]} — {nome}\n'
                         + r'$\sigma_\tau/\langle\tau\rangle$ = ' + f'{cv:.2f}{extra}',
                         fontsize=9)
            ax = axes[i][2 * j + 1]
            for arr, kw in [(d['s2'], dict(color='k', lw=1.8, marker='o', ms=2.5,
                                           label='originale')),
                            (d['s2_A1'], dict(color='#0072B2', lw=1.2, ls='--',
                                              label='A1')),
                            (d['s2_A2'], dict(color='#D55E00', lw=1.2, ls='-.',
                                              label='A2'))]:
                m = np.isfinite(arr) & (arr > 0)
                ax.plot(LAGS[m], np.asarray(arr)[m], **kw)
            g = fit_gamma(d['s2'], *FIT_RANGE)
            g2 = fit_gamma(d['s2_A2'], *FIT_RANGE)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.axvspan(*FIT_RANGE, color='grey', alpha=.10)
            ax.set_xlabel('$t$')
            ax.set_ylabel(r'$\sigma_X^2(t)$')
            ax.set_title(r'$\hat\gamma$ = ' + f'{g:.2f}   '
                         + r'$\hat\gamma_{A2}$ = ' + f'{g2:.2f}', fontsize=9)
            ax.legend(fontsize=6.5)
    fig.suptitle('Analogo della Fig. 2 di Altmann et al. sui quattro corpora\n'
                 'lettera: coda rapida e $\\hat\\gamma_{A2}\\simeq1$ — parola chiave '
                 'di burstiness mediana: coda larga e $\\hat\\gamma_{A2}\\simeq\\hat\\gamma$',
                 fontweight='bold')
    fig.tight_layout()
    nome = 'cap5_altmann_fig2' if APPLICA else 'cap5_altmann_fig2_nuova'
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{nome}.{ext}')
    plt.close(fig)
    print(f"\n-> {OUT / (nome + '.png')}")


if __name__ == '__main__':
    main()
