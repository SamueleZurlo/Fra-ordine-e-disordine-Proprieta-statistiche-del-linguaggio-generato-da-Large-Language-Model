# -*- coding: utf-8 -*-
"""Perche' i controlli appaiati di Grokipedia hanno J < 0 e quelli di
Wikipedia J > 0?

Tre spiegazioni possibili, che vanno separate:

  (a) ARTEFATTO DI QUANTIZZAZIONE. verifica_J.py mostra che J non e' invariante
      per dilatazione quando <tau> e' piccolo. Se i controlli dei due corpora
      hanno <tau> molto diversi, una parte del divario e' strumentale.

  (b) COMPOSIZIONE. I controlli sono parole diverse nei due corpora: il divario
      potrebbe misurare quali parole vengono scelte, non come sono distribuite.

  (c) STILE. Le stesse parole sono distribuite in modo diverso.

Il test che separa (b) da (c) e' il confronto appaiato PER PAROLA sulle parole
usate come controllo in entrambi i corpora.

Nessun file della tesi viene modificato.
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu

from autoanalisi import WIKI    # noqa: E402

D = pd.read_csv(WIKI / 'risultati_entropia_v2' / 'dati' /
                'entropia_sequenze.csv')
CORP = ['letterario', 'wikipedia', 'grok_v01', 'grok_oggi']
ETI = {'letterario': 'Guerra e pace', 'wikipedia': 'Wikipedia',
       'grok_v01': 'Grokipedia v0.1', 'grok_oggi': 'Grokipedia oggi'}

print('=' * 78)
print('IL FATTO DA SPIEGARE')
print('=' * 78)
for tp in ['keyword', 'appaiata']:
    print(f'\n{tp}:')
    for c in CORP:
        s = D[(D.corpus == c) & (D.tipo == tp) & D.J.notna()]
        pd_ = s.groupby('titolo')['J'].mean()
        print(f'  {ETI[c]:18} J = {pd_.mean():+.3f}   '
              f'n_eventi mediano {s.n_eventi.median():5.0f}   '
              f'<tau> mediano {s.mean_tau.median():7.1f}   '
              f'{len(s)} sequenze')

# ------------------------------------------------------------------ (a)
print('\n' + '=' * 78)
print('(a) ARTEFATTO DI QUANTIZZAZIONE?')
print('=' * 78)
print('verifica_J.py: J e\' distorto verso lo zero quando <tau> < ~30.')
A = D[(D.tipo == 'appaiata') & D.J.notna()]
print(f"\n{'corpus':18}{'<tau> min':>11}{'q25':>9}{'mediana':>9}"
      f"{'q75':>9}{'% con <tau> < 30':>18}")
for c in CORP:
    s = A[A.corpus == c]
    q = s.mean_tau.quantile([.25, .5, .75])
    print(f'{ETI[c]:18}{s.mean_tau.min():>11.1f}{q[.25]:>9.1f}'
          f'{q[.5]:>9.1f}{q[.75]:>9.1f}{(s.mean_tau < 30).mean():>17.0%}')

# la dipendenza di J da <tau> dentro ciascun corpus
print('\ncorrelazione di rango fra J e <tau>, dentro il livello appaiata:')
for c in CORP:
    s = A[A.corpus == c]
    rho = s[['J', 'mean_tau']].corr(method='spearman').iloc[0, 1]
    print(f'  {ETI[c]:18} rho = {rho:+.3f}  (n={len(s)})')

# ------------------------------------------------------------------ (b)
print('\n' + '=' * 78)
print('(b) COMPOSIZIONE: quali parole sono scelte come controllo')
print('=' * 78)
for c in CORP:
    s = A[A.corpus == c]
    print(f'\n{ETI[c]} — {s.sequenza.nunique()} parole distinte, '
          f'le 12 piu\' usate:')
    for w, n in Counter(s.sequenza).most_common(12):
        j = s[s.sequenza == w].J.mean()
        print(f'    {w:14} in {n:2d} voci   J = {j:+.3f}')

comuni = set(A[A.corpus == 'wikipedia'].sequenza) & \
    set(A[A.corpus == 'grok_oggi'].sequenza)
print(f'\nparole usate come controllo in ENTRAMBI i corpora: {len(comuni)}')
print('  ' + ', '.join(sorted(comuni)[:40]))

# ------------------------------------------------------------------ (c)
print('\n' + '=' * 78)
print('(c) STILE: la stessa parola, nei due corpora')
print('=' * 78)
righe = []
for w in sorted(comuni):
    a = A[(A.corpus == 'wikipedia') & (A.sequenza == w)]
    b = A[(A.corpus == 'grok_oggi') & (A.sequenza == w)]
    if len(a) < 3 or len(b) < 3:
        continue
    righe.append(dict(parola=w, n_wiki=len(a), n_grok=len(b),
                      J_wiki=a.J.mean(), J_grok=b.J.mean(),
                      dJ=b.J.mean() - a.J.mean(),
                      tau_wiki=a.mean_tau.median(),
                      tau_grok=b.mean_tau.median()))
C = pd.DataFrame(righe).sort_values('dJ')
if len(C):
    print(f"{'parola':14}{'n wiki':>8}{'n grok':>8}{'J wiki':>9}"
          f"{'J grok':>9}{'differenza':>12}")
    for r in C.itertuples():
        print(f'{r.parola:14}{r.n_wiki:>8}{r.n_grok:>8}{r.J_wiki:>9.3f}'
              f'{r.J_grok:>9.3f}{r.dJ:>+12.3f}')
    d = C.dJ.values
    p = wilcoxon(d).pvalue if len(d) >= 6 else float('nan')
    print(f'\nsu {len(C)} parole in comune: differenza mediana '
          f'{np.median(d):+.3f} bit, p = {p:.3g}')
    print('se la differenza sopravvive a parola fissa, e\' stile, non '
          'composizione.')

# ---------------------------------------------- quanto pesa la composizione
print('\n' + '=' * 78)
print('QUANTO DEL DIVARIO E\' COMPOSIZIONE E QUANTO E\' STILE')
print('=' * 78)
w = A[A.corpus == 'wikipedia']
g = A[A.corpus == 'grok_oggi']
tot = g.groupby('titolo').J.mean().mean() - w.groupby('titolo').J.mean().mean()
print(f'divario totale (media per documento):        {tot:+.3f} bit')
if len(C):
    print(f'divario a parola fissa (parole in comune):   {np.mean(C.dJ):+.3f} bit')
    print(f'residuo attribuibile alla composizione:      '
          f'{tot - np.mean(C.dJ):+.3f} bit')

# ------------------------------------------- che tipo di parole sono
print('\n' + '=' * 78)
print('CHE COSA SONO QUELLE PAROLE')
print('=' * 78)
print('Il §5.7 dice che i controlli sono in larga parte lessico di servizio.')
print('Se lo sono in entrambi i corpora, il divario non e\' spiegato da questo.\n')
STOP_EN = set("""a about above after again against all am an and any are as at
be because been before being below between both but by can cannot could did do
does doing down during each few for from further had has have having he her here
hers herself him himself his how i if in into is it its itself me more most my
myself no nor not of off on once only or other ought our ours ourselves out over
own same she should so some such than that the their theirs them themselves then
there these they this those through to too under until up very was we were what
when where which while who whom why with would you your yours yourself
""".split())
for c in ['wikipedia', 'grok_v01', 'grok_oggi']:
    s = A[A.corpus == c]
    q = s.sequenza.map(lambda x: x in STOP_EN)
    print(f'{ETI[c]:18} quota di controlli che sono stopword inglesi: '
          f'{q.mean():.0%}   J medio: servizio {s[q].J.mean():+.3f}, '
          f'altro {s[~q].J.mean():+.3f}')
