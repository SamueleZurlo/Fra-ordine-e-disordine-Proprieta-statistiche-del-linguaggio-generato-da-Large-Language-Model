"""
Controprova dell'esponente di coda su romanzi diversi da Guerra e pace.

DOMANDA
-------
Sul livello delle parole chiave il corpus letterario dà mu = 1.76, fuori dalla
banda 2 < mu < 3 in cui vale l'Eq. 8 di Altmann et al., e la sua curva scende
salendo nella gerarchia mentre quelle enciclopediche salgono. Il comportamento
e' di Guerra e pace o della prosa letteraria in generale?

METODO
------
Due romanzi inglesi di lunghezza confrontabile vengono trattati esattamente come
Guerra e pace: stessa pulizia Gutenberg, stessa segmentazione in 20 finestre di
N_EFF caratteri, stessa selezione dei bersagli con la lista di parole funzione
estesa, stesso pool di intervalli normalizzati e stesso fit troncato. Le
funzioni sono prese dai notebook, non riscritte.

Il confronto viene fatto anche al variare del quantile che definisce x_min,
perche' il §5.8 mostra che l'ordinamento fra corpora dipende da quella scelta:
se il romanzo sta sotto gli altri a ogni quantile, il fenomeno e' reale; se
l'ordine si inverte, e' un artefatto della soglia.

USO
---
    py controllo_letterario.py
"""
import os
import re
import ssl
import sys
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rigenera_cap5_stoplist import STOP_EXTRA, celle_codice   # noqa: E402

BASE = Path(__file__).resolve().parents[2]
WIKI = BASE / 'Wiki'
CACHE = BASE / 'corpora_cache'
CACHE.mkdir(exist_ok=True)

N_EFF = 60_000
N_SEG = 20
MIN_TAU_SEQ = 20
QUANTILI = (75, 80, 85, 90, 95)

# romanzi inglesi lunghi abbastanza da dare 20 finestre non sovrapposte
LIBRI = {
    'Moby Dick': 'https://www.gutenberg.org/cache/epub/2701/pg2701.txt',
    'Middlemarch': 'https://www.gutenberg.org/cache/epub/145/pg145.txt',
    'David Copperfield': 'https://www.gutenberg.org/cache/epub/766/pg766.txt',
}

GS = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.S)
GE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.S)
ST = re.compile(r"^[ \t]*(?:BOOK\s+[A-Z]+[^\n]*|CHAPTER\s+[IVXLCDM\d]+[^\n]*|"
                r"(?:FIRST|SECOND)\s+EPILOGUE[^\n]*|EPILOGUE[^\n]*|CONTENTS[^\n]*|"
                r"PART\s+[IVXLCDM\d]+[^\n]*|APPENDIX[^\n]*|\d+)[ \t]*$", re.M)


def ctx_ssl():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def scarica(nome, url):
    f = CACHE / (re.sub(r'\W+', '_', nome).lower() + '.txt')
    if not f.exists():
        req = urllib.request.Request(url, headers={'User-Agent': 'research/1.0'})
        with urllib.request.urlopen(req, timeout=90, context=ctx_ssl()) as r:
            f.write_bytes(r.read())
        print(f'  scaricato {nome}')
    return f.read_bytes().decode('utf-8', errors='replace')


def segmenti(raw):
    """Stessa preparazione di Guerra e pace nel notebook a tre corpora."""
    raw = raw.replace('\r\n', '\n').replace('\r', '\n')
    m = GS.search(raw)
    if m:
        raw = raw[m.end():]
    m = GE.search(raw)
    if m:
        raw = raw[:m.start()]
    raw = ST.sub('', raw)
    raw = re.sub(r'\n[ \t]+\n', '\n\n', raw)
    body = re.sub(r'\n{3,}', '\n\n', raw).strip()
    passo = max(len(body) - N_EFF, 0)
    return body, [body[s:s + N_EFF] for s in
                  np.linspace(0, passo, N_SEG).astype(int)]


def prepara():
    celle = celle_codice(WIKI / 'altmann_tre_corpora.ipynb')
    ns = {'__name__': '__main__', 'display': lambda x: print(x),
          'STOP_EXTRA_INIETTATA': STOP_EXTRA}
    for i in (0, 1, 2):
        exec(compile(celle[i], f'<tre#{i}>', 'exec'), ns)
    ns['N_EFF'] = N_EFF
    celle_c = celle_codice(WIKI / 'esponente_coda.ipynb')
    nsc = {'__name__': '__main__', 'display': lambda x: print(x),
           'STOP_EXTRA_INIETTATA': STOP_EXTRA}
    for i in (0, 1):
        exec(compile(celle_c[i], f'<coda#{i}>', 'exec'), nsc)
    for i, src in enumerate(celle_c):
        if 'def fit_tronc' in src:
            exec(compile(src, f'<coda#{i}>', 'exec'), nsc)
            break
    return ns, nsc


def pool_keyword(ns, segs, nome):
    """Intervalli delle parole chiave, normalizzati per la propria media."""
    fuori, n_seq = [], 0
    for i, t in enumerate(segs):
        rec = ns['analizza'](t, dict(corpus=nome, titolo=f'seg_{i:02d}'))
        for r in rec:
            if r['tipo'] != 'keyword':
                continue
            pos = ns['pos_from_word'](t[:N_EFF], r['sequenza'])
            pos = pos[pos < N_EFF]
            if len(pos) < MIN_TAU_SEQ + 1:
                continue
            tau = np.diff(pos).astype(float)
            fuori.append(tau / tau.mean())
            n_seq += 1
    return np.concatenate(fuori), n_seq


def main():
    cwd = Path.cwd()
    os.chdir(WIKI)
    try:
        ns, nsc = prepara()
        print(f"nucleo pronto | STOPWORDS = {len(ns['STOPWORDS'])}\n")
        risultati = {}
        for nome, url in LIBRI.items():
            body, segs = segmenti(scarica(nome, url))
            sovrapp = max(0, N_EFF - (len(body) - N_EFF) / (N_SEG - 1))
            x0, n_seq = pool_keyword(ns, segs, nome)
            print(f'{nome}: corpo {len(body):,} caratteri, {n_seq} sequenze, '
                  f'{len(x0):,} intervalli, sovrapposizione fra finestre '
                  f'{sovrapp:,.0f} caratteri')
            riga = {}
            for q in QUANTILI:
                xmin = float(np.percentile(x0, q))
                x = x0[x0 >= xmin]
                if len(x) < 150:
                    continue
                mu, tc, _ = nsc['fit_tronc'](x, xmin)
                riga[q] = (mu, tc, len(x))
            risultati[nome] = riga
    finally:
        os.chdir(cwd)

    print('\n' + '=' * 78)
    print('ESPONENTE DI CODA SULLE PAROLE CHIAVE, al variare del quantile di x_min')
    print('=' * 78)
    print(f"{'testo':22}" + ''.join(f'{q:>10}' for q in QUANTILI))
    for nome, r in risultati.items():
        print(f'{nome:22}' + ''.join(f'{r[q][0]:>10.2f}' if q in r else f'{"--":>10}'
                                     for q in QUANTILI))
    print('\nper confronto, dai risultati del capitolo (stessa procedura):')
    print(f"{'Guerra e pace':22}" + f"{1.65:>10.2f}{1.75:>10.2f}{1.75:>10.2f}"
          f"{1.76:>10.2f}{2.11:>10.2f}")
    print(f"{'Wikipedia':22}" + f"{1.65:>10.2f}{1.85:>10.2f}{1.99:>10.2f}"
          f"{2.37:>10.2f}{2.57:>10.2f}")
    print('\ntau_c corrispondente:')
    print(f"{'testo':22}" + ''.join(f'{q:>10}' for q in QUANTILI))
    for nome, r in risultati.items():
        print(f'{nome:22}' + ''.join(f'{r[q][1]:>10.1f}' if q in r else f'{"--":>10}'
                                     for q in QUANTILI))


if __name__ == '__main__':
    main()
