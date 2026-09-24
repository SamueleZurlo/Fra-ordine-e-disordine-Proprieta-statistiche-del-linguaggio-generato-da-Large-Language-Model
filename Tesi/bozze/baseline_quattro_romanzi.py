# -*- coding: utf-8 -*-
"""Quanto dipende T* dal fatto che la baseline umana e' UN testo solo?

MOTIVO
------
Le Tabelle S1--S11 di Altmann et al. mostrano che la variabilita' fra testi al
livello di parola e' grande (media delle sette sequenze piu' bursty: da 1.60 su
The Voyage of the Beagle a 4.38 su Guerra e pace) e che Guerra e pace e' il piu'
estremo dei dieci libri. Sei degli otto criteri del Capitolo 4 scelgono T*
minimizzando la distanza da una baseline misurata su quel solo testo.

METODO
------
Le stesse funzioni del notebook dello sweep vengono applicate a tre altri
romanzi inglesi, alla stessa lunghezza in token e con lo stesso tokenizzatore.
Si ottiene cosi' una dispersione fra testi per ciascuna grandezza della
baseline. Poi, per ciascuno dei quattro romanzi, si ricalcola T* per i sei
criteri che dipendono dalla baseline e si guarda se la mediana per modello
cambia.

Il criterio D_s va ricalcolato per intero, perche' confronta ogni documento
generato con il testo umano: si rifa' contro ciascun romanzo.

USO
---
    py baseline_quattro_romanzi.py
"""
import io
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

BASE = Path(__file__).resolve().parents[2]
SWEEP = BASE / 'Analisi Sweep di temperature'
NB = SWEEP / 'analisi_sweep_temperature.ipynb'
CACHE = BASE / 'corpora_cache'
CACHE.mkdir(exist_ok=True)

# gli stessi tre romanzi usati come controllo nel capitolo 5
ALTRI = {
    'Moby Dick': 'https://www.gutenberg.org/cache/epub/2701/pg2701.txt',
    'Middlemarch': 'https://www.gutenberg.org/cache/epub/145/pg145.txt',
    'David Copperfield': 'https://www.gutenberg.org/cache/epub/766/pg766.txt',
}

GS = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.S)
GE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.S)

# le sei grandezze su cui il Capitolo 4 sceglie T* per prossimita' all'umano
CRITERI = [('D_s_umano', 'D_s minima dal testo umano'),
           ('R_gzip', 'R_gzip piu vicino'),
           ('alpha', 'alpha di Zipf piu vicino'),
           ('R', 'R piu vicino'),
           ('beta', 'gamma_H (Heaps) piu vicino'),
           ('alpha_DFA', 'alpha_DFA piu vicino')]


def celle(nb_path):
    d = json.loads(Path(nb_path).read_text(encoding='utf-8'))
    return [''.join(c['source']) for c in d['cells'] if c['cell_type'] == 'code']


def scarica(nome, url):
    f = CACHE / (re.sub(r'\W+', '_', nome).lower() + '.txt')
    if not f.exists():
        req = urllib.request.Request(url, headers={'User-Agent': 'research/1.0'})
        with urllib.request.urlopen(req, timeout=90) as r:
            f.write_bytes(r.read())
    raw = f.read_bytes().decode('utf-8', errors='replace')
    m = GS.search(raw)
    if m:
        raw = raw[m.end():]
    m = GE.search(raw)
    if m:
        raw = raw[:m.start()]
    return raw.strip()


def main():
    cwd = Path.cwd()
    os.chdir(SWEEP)
    try:
        cs = celle(NB)
        ns = {'__name__': '__main__', 'display': lambda *a, **k: None}
        # 0-2 setup e configurazione, 7-11 le funzioni di misura
        for i in (0, 1, 2, 7, 8, 9, 10, 11):
            exec(compile(cs[i], f'<sweep#{i}>', 'exec'), ns)
        CFG, ENC = ns['CFG'], None
        import tiktoken
        ENC = tiktoken.get_encoding(CFG.TOKENIZER)
        N_TOK = CFG.N_TOK
        print(f'tokenizzatore {CFG.TOKENIZER}, N_TOK = {N_TOK}\n')

        # --- baseline di Guerra e pace, gia' misurata dal notebook ----------
        wp = pd.read_csv(SWEEP / 'Risultati Sweep' / 'baseline_umana.csv',
                         index_col=0)['0']
        base = {'Guerra e pace': {k: float(wp[k]) for k in
                                  ['alpha', 'MAPE', 'beta', 'R', 'alpha_DFA',
                                   'R_gzip', 'H_bit_char']}}

        # --- le stesse misure sugli altri tre --------------------------------
        testi = {}
        for nome, url in ALTRI.items():
            body = scarica(nome, url)
            ids = ENC.encode(body[:3_000_000], disallowed_special=())[:N_TOK]
            tok = [str(i) for i in ids]
            txt = ENC.decode(ids)
            par = re.findall(r"[a-zA-Z']+", txt.lower())
            testi[nome] = txt
            a_c, mape = ns['zipf_cumulativo_fit'](tok)
            base[nome] = dict(
                alpha=a_c, MAPE=mape, beta=ns['heaps_beta'](tok),
                R=ns['descrittore_R'](tok),
                alpha_DFA=ns['dfa_alpha'](ns['serie_rango_caratteri'](txt)),
                R_gzip=ns['gzip_ratio'](txt),
                H_bit_char=ns['tasso_entropia'](txt[:2 * CFG.KL_CHUNK]))
            print(f'{nome:20} misurato ({len(txt):,} caratteri)')

        B = pd.DataFrame(base).T
        print('\n' + '=' * 78)
        print('BASELINE UMANA SU QUATTRO ROMANZI, stessa lunghezza e procedura')
        print('=' * 78)
        print(B.round(4).to_string())
        print('\ndispersione fra testi:')
        for c in B.columns:
            v = B[c].astype(float)
            print(f'  {c:12s} media {v.mean():7.4f}  sd {v.std(ddof=1):7.4f}  '
                  f'min {v.min():7.4f}  max {v.max():7.4f}  '
                  f'escursione/media {100*(v.max()-v.min())/abs(v.mean()):5.1f}%')

        # --- D_s di ogni documento generato contro ciascun romanzo -----------
        DF = pd.read_csv(SWEEP / 'Risultati Sweep' / 'metriche_per_documento.csv')
        print(f'\n{len(DF)} documenti generati letti dal notebook')

        # per ricalcolare D_s serve il testo di ciascun documento: si rilegge il
        # dataset con le celle 3-6 del notebook
        for i in (3, 4, 5, 6):
            exec(compile(cs[i], f'<sweep#{i}>', 'exec'), ns)
        DOCS = ns['DOCS']
        chiave = {(r['_modello'], r['temperature'], r.get('sample_id', j)): r['_testo']
                  for j, r in enumerate(DOCS)}
        testi['Guerra e pace'] = ns['UMANO_TXT']

        Ds = {}
        for nome, ref in testi.items():
            col = []
            for r in DF.itertuples():
                t = chiave.get((r.modello, r.temperatura, r.sample_id))
                col.append(np.nan if t is None else
                           ns['kl_simmetrica'](t[:CFG.KL_CHUNK], ref[:CFG.KL_CHUNK]))
            Ds[nome] = col
            print(f'  D_s ricalcolata contro {nome}')

        # --- T* per ciascun criterio e ciascuna baseline ---------------------
        MOD = sorted(DF.modello.unique())
        righe = []
        for nome in base:
            for col, eti in CRITERI:
                d = DF.copy()
                if col == 'D_s_umano':
                    d['_v'] = Ds[nome]
                    bersaglio = 0.0
                else:
                    d['_v'] = d[col]
                    bersaglio = float(base[nome][col])
                for m in MOD:
                    s = d[d.modello == m].groupby('temperatura')['_v'].mean()
                    s = s.dropna()
                    if not len(s):
                        continue
                    righe.append(dict(baseline=nome, criterio=eti, modello=m,
                                      T=float((s - bersaglio).abs().idxmin())))
        T = pd.DataFrame(righe)

        print('\n' + '=' * 78)
        print('T* PER CRITERIO E PER BASELINE')
        print('=' * 78)
        for m in MOD:
            print(f'\n--- {m} ---')
            p = T[T.modello == m].pivot(index='criterio', columns='baseline',
                                        values='T')
            print(p.to_string())
        print('\n' + '=' * 78)
        print('MEDIANA DEI SEI CRITERI, per modello e per baseline')
        print('=' * 78)
        med = T.groupby(['modello', 'baseline']).T.median().unstack()
        print(med.to_string())
    finally:
        os.chdir(cwd)


if __name__ == '__main__':
    main()
