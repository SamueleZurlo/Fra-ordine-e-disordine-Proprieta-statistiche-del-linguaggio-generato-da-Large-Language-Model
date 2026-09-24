"""
Ricostruisce il confronto fra le due lunghezze di analisi del §5.10.

PERCHE' ESISTE QUESTO FILE
--------------------------
La cella del notebook altmann_tre_corpora che disegna la Figura sulla robustezza
non calcola nulla: legge confronto_N_eff/confronto.json, prodotto a suo tempo da
uno script che non fa piu' parte del progetto. Rieseguire il notebook con la
lista di parole funzione estesa lascia quindi quella figura ferma ai valori
vecchi, e il §5.10 finirebbe per contraddire il §5.3. Questo programma
ricostruisce il JSON mancante.

COME
----
Le funzioni di misura non vengono riscritte: vengono prese dai notebook
eseguendone le celle di definizione, cosi' la catena e' identica a quella del
resto del capitolo. La lista di parole funzione viene estesa nel sorgente prima
dell'esecuzione, con lo stesso meccanismo di rigenera_cap5_stoplist.py. L'unica
cosa scritta qui e' il ciclo sulle due lunghezze.

USO
---
    py confronto_neff.py
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rigenera_cap5_stoplist import STOP_EXTRA, celle_codice   # noqa: E402

BASE = Path(__file__).resolve().parents[2]
WIKI = BASE / 'Wiki'
OUT = WIKI / 'confronto_N_eff_v2'
OUT.mkdir(parents=True, exist_ok=True)

LUNGHEZZE = [60_000, 80_000]
ORDINE = ['letterario', 'wikipedia', 'grok_v01', 'grok_oggi']
LIV = ['lettera', 'funzione', 'appaiata', 'keyword']
SOGLIA_DEN = 0.05        # come nella cella della quota di burstiness
MIN_TAU_SEQ = 20         # come nel notebook della coda
QUANT_XMIN = 90


def prepara_misura():
    """Celle di configurazione, import, nucleo di misura e caricamento corpora."""
    celle = celle_codice(WIKI / 'altmann_tre_corpora.ipynb')
    ns = {'__name__': '__main__', 'display': lambda x: print(x),
          'STOP_EXTRA_INIETTATA': STOP_EXTRA}
    for i in (0, 1, 2, 3):        # config, palette+import, nucleo, corpora
        exec(compile(celle[i], f'<tre_corpora#{i}>', 'exec'), ns)
    assert 'analizza' in ns and 'TESTI' in ns
    print(f"nucleo pronto | STOPWORDS = {len(ns['STOPWORDS'])} parole "
          f"| titoli = {len(ns['TESTI'])}")
    return ns, celle


def prepara_coda():
    """Solo le definizioni del fit troncato, prese dal notebook della coda."""
    celle = celle_codice(WIKI / 'esponente_coda.ipynb')
    ns = {'__name__': '__main__', 'display': lambda x: print(x),
          'STOP_EXTRA_INIETTATA': STOP_EXTRA}
    for i in (0, 1):
        exec(compile(celle[i], f'<coda#{i}>', 'exec'), ns)
    # la cella con fit_tronc: si riconosce dal nome della funzione
    for i, src in enumerate(celle):
        if 'def fit_tronc' in src:
            exec(compile(src, f'<coda#{i}>', 'exec'), ns)
            break
    assert 'fit_tronc' in ns
    print('fit della coda pronto')
    return ns


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return (max(c - h, 0.0), min(c + h, 1.0))


def test_interno(A, corpus, col):
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
    return [v, tot, v / tot if tot else float('nan'), lo, hi]


def per_doc(A, c, tipo, col):
    s = A[(A.corpus == c) & (A.tipo == tipo)]
    return s.groupby('titolo')[col].mean().dropna()


def coppia(A, tipo):
    a = per_doc(A, 'grok_oggi', tipo, 'cv_tau')
    b = per_doc(A, 'wikipedia', tipo, 'cv_tau')
    idx = a.index.intersection(b.index)
    a, b = a[idx], b[idx]
    if len(a) < 5:
        return [len(a), float('nan'), float('nan'), float('nan')]
    return [len(a), float(np.median(a - b)), float((a > b).mean()),
            float(sps.wilcoxon(a, b).pvalue)]


def quota(A, c, tipo):
    s = A[(A.corpus == c) & (A.tipo == tipo)]
    d = s.groupby('titolo')[['gamma', 'gamma_A2']].mean().dropna()
    d = d[d.gamma - 1 > SOGLIA_DEN]
    if len(d) < 5:
        return float('nan')
    q = ((d.gamma_A2 - 1) / (d.gamma - 1)).clip(-0.5, 1.5)
    return float(q.median())


def coda_keyword(A, docs_testi, ns, nsc, N):
    """mu e tau_c sul pool delle keyword di ciascun corpus, come nel §5.7."""
    fuori = {}
    for c in ORDINE:
        pool = []
        s = A[(A.corpus == c) & (A.tipo == 'keyword')]
        for (tit, seq), _ in s.groupby(['titolo', 'sequenza']):
            t = docs_testi[(c, tit)][:N]
            pos = ns['pos_from_word'](t, seq)
            pos = pos[pos < N]
            if len(pos) < MIN_TAU_SEQ + 1:
                continue
            tau = np.diff(pos).astype(float)
            pool.append(tau / tau.mean())
        if not pool:
            continue
        x0 = np.concatenate(pool)
        xmin = float(np.percentile(x0, QUANT_XMIN))
        x = x0[x0 >= xmin]
        if len(x) < 150:
            continue
        mu, tc, _ = nsc['fit_tronc'](x, xmin)
        fuori[c] = (mu, tc, int(len(x0)))
    return fuori


def misura(ns, celle, N, nsc):
    ns['N_EFF'] = N
    exec(compile(celle[4], '<tre_corpora#4>', 'exec'), ns)   # USABILI e SEG
    usabili, SEG, TESTI = ns['USABILI'], ns['SEG'], ns['TESTI']
    print(f'  N = {N:,}: {len(usabili)} terne, {len(SEG)} segmenti letterari')

    testi = {}
    rec = []
    for c in ('wikipedia', 'grok_v01', 'grok_oggi'):
        for t in usabili:
            testi[(c, t)] = TESTI[t][c]
            rec += ns['analizza'](TESTI[t][c], dict(corpus=c, titolo=t))
    for i, s in enumerate(SEG):
        lab = f'wrnpc_{i:02d}'
        testi[('letterario', lab)] = s
        rec += ns['analizza'](s, dict(corpus='letterario', titolo=lab))
    A = pd.DataFrame(rec)
    A.to_csv(OUT / f'sequenze_N{N//1000}k.csv', index=False)

    R = {'N_EFF': N, 'n_terne': len(usabili)}
    for c in ORDINE:
        for col in ('cv_tau', 'gamma'):
            R[f'test_{c}_{col}'] = test_interno(A, c, col)
    for tipo in LIV:
        R[f'pair_{tipo}'] = coppia(A, tipo)
    for c in ORDINE:
        for tipo in LIV:
            R[f'q_{c}_{tipo}'] = quota(A, c, tipo)
            for col in ('cv_tau', 'gamma', 'gamma_A2'):
                v = per_doc(A, c, tipo, col)
                R[f'{col}_{c}_{tipo}'] = float(v.mean()) if len(v) else float('nan')
    for c, (mu, tc, n) in coda_keyword(A, testi, ns, nsc, N).items():
        R[f'mu_{c}'], R[f'tauc_{c}'], R[f'ntau_{c}'] = mu, tc, n
    return R


def main():
    # i notebook usano Path.cwd() per costruire i propri percorsi: vanno eseguiti
    # dalla cartella Wiki, non da qui
    cwd = Path.cwd()
    os.chdir(WIKI)
    try:
        ns, celle = prepara_misura()
        nsc = prepara_coda()
        fuori = {}
        for N in LUNGHEZZE:
            fuori[str(N)] = misura(ns, celle, N, nsc)
    finally:
        os.chdir(cwd)
    (OUT / 'confronto.json').write_text(json.dumps(fuori, indent=1), encoding='utf-8')
    print(f'\n-> {OUT / "confronto.json"}')

    a, b = fuori[str(LUNGHEZZE[0])], fuori[str(LUNGHEZZE[1])]
    print(f'\nterne: {a["n_terne"]} -> {b["n_terne"]}')
    print('\n1. TEST INTERNO (frazione vinta dalle keyword)')
    for c in ORDINE:
        for col in ('cv_tau', 'gamma'):
            print(f'  {c:12}{col:9}{a[f"test_{c}_{col}"][2]:>8.2f}'
                  f'{b[f"test_{c}_{col}"][2]:>8.2f}')
    print('\n2. DIVARIO GROKIPEDIA OGGI - WIKIPEDIA (cv)')
    for tipo in LIV:
        x, y = a[f'pair_{tipo}'], b[f'pair_{tipo}']
        print(f'  {tipo:20}{x[1]:>+9.3f}{x[3]:>10.1e}{y[1]:>+9.3f}{y[3]:>10.1e}')
    print('\n3. QUOTA DI BURSTINESS q SULLE KEYWORD')
    for c in ORDINE:
        print(f'  {c:12}{a[f"q_{c}_keyword"]:>8.2f}{b[f"q_{c}_keyword"]:>8.2f}')
    d60 = a['q_wikipedia_keyword'] - a['q_grok_oggi_keyword']
    d80 = b['q_wikipedia_keyword'] - b['q_grok_oggi_keyword']
    print(f'  differenza Wikipedia - Grokipedia: {d60:+.2f} a 60k, {d80:+.2f} a 80k')
    print('\n4. CODA DELLE KEYWORD')
    for c in ORDINE:
        if f'mu_{c}' in a:
            print(f'  {c:12} mu {a[f"mu_{c}"]:>6.2f} -> {b[f"mu_{c}"]:>6.2f}   '
                  f'tau_c {a[f"tauc_{c}"]:>6.1f} -> {b[f"tauc_{c}"]:>6.1f}')


if __name__ == '__main__':
    main()
