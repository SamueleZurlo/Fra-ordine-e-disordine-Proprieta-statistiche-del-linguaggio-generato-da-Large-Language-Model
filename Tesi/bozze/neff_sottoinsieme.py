# -*- coding: utf-8 -*-
"""Aggiunge a confronto.json il blocco '60000_35': le stesse misure a 60k
calcolate sul sottoinsieme di titoli che sopravvive anche a 80k.

Il confronto del §5.9 metteva 39 terne a 60k contro 35 a 80k, cosicche' lo
spostamento del divario poteva venire dalla composizione del campione anziche'
dalla lunghezza. Con questo blocco le due lunghezze si confrontano sugli stessi
titoli.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BASE = Path(__file__).resolve().parents[2]
DIR = BASE / 'Wiki' / 'confronto_N_eff_v2'

a = pd.read_csv(DIR / 'sequenze_N60k.csv')
b = pd.read_csv(DIR / 'sequenze_N80k.csv')
R = json.loads((DIR / 'confronto.json').read_text(encoding='utf-8'))

TIPI = ['lettera', 'funzione', 'appaiata', 'keyword']
CORPORA = ['letterario', 'wikipedia', 'grok_v01', 'grok_oggi']

# titoli presenti a entrambe le lunghezze, corpus per corpus
comuni = {c: set(a[a.corpus == c].titolo) & set(b[b.corpus == c].titolo)
          for c in CORPORA}
sub = a[[t in comuni[c] for c, t in zip(a.corpus, a.titolo)]]

fuori = {}
fuori['N_EFF'] = 60000
fuori['n_terne'] = len(comuni['wikipedia'])

for c in CORPORA:
    for t in TIPI:
        s = sub[(sub.corpus == c) & (sub.tipo == t)]
        fuori[f'cv_tau_{c}_{t}'] = float(s.cv_tau.mean())
        fuori[f'gamma_{c}_{t}'] = float(s.gamma.mean())
        fuori[f'gamma_A2_{c}_{t}'] = float(s.gamma_A2.mean())


def per_titolo(df, corpus, tipo):
    s = df[(df.corpus == corpus) & (df.tipo == tipo)]
    return s.groupby('titolo')['cv_tau'].mean()


for t in TIPI:
    w = per_titolo(sub, 'wikipedia', t)
    g = per_titolo(sub, 'grok_oggi', t)
    idx = sorted(set(w.index) & set(g.index))
    d = (g[idx] - w[idx]).values
    fuori[f'pair_{t}'] = [len(d), float(np.median(d)), float(np.mean(d > 0)),
                          float(wilcoxon(d).pvalue)]

R['60000_35'] = fuori
(DIR / 'confronto.json').write_text(json.dumps(R, indent=1), encoding='utf-8')

print(f"blocco 60000_35 scritto, n_terne={fuori['n_terne']}")
for t in TIPI:
    n, m, _, p = fuori[f'pair_{t}']
    n8, m8, _, p8 = R['80000'][f'pair_{t}']
    print(f'  {t:10s} 60k/35 {m:+.4f} (p={p:.3g})   80k/35 {m8:+.4f} (p={p8:.3g})')
