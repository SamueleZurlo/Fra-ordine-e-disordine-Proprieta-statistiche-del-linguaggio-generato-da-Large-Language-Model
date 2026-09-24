# -*- coding: utf-8 -*-
"""Quali sono gli esponenti DFA dei cinque documenti a T = 0.4?

Il §4.4 riporta l'intervallo 0.16--0.66, il §6.8 l'intervallo 0.06--0.72.
Uno dei due e' sbagliato. Il capitolo usa il detrending di ordine 2, quindi la
fonte corretta e' dfa_ordine2.csv; per completezza si riporta anche l'ordine 1,
che e' quello della versione precedente del capitolo.
"""
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                              errors='replace')

import pandas as pd

RIS = (Path(__file__).resolve().parents[2] /
       'Analisi Sweep di temperature' / 'Risultati Sweep')

O2 = pd.read_csv(RIS / 'dfa_ordine2.csv')
O1 = pd.read_csv(RIS / 'metriche_per_documento.csv')
print('ordine 2:', list(O2.columns))
print('ordine 1:', [c for c in O1.columns if 'DFA' in c or c in
                    ('modello', 'temperatura', 'sample_id')], '\n')

col2 = [c for c in O2.columns if 'alpha' in c.lower() or 'dfa' in c.lower()]
print('colonne di esponente in dfa_ordine2.csv:', col2, '\n')

for eti, D, col in [('ORDINE 2 (usato nel capitolo)', O2, col2[0]),
                    ('ordine 1 (versione precedente)', O1, 'alpha_DFA')]:
    d = D[D.temperatura == 0.4]
    print('=' * 68)
    print(f'{eti}   T = 0.4')
    print('=' * 68)
    print(f"{'modello':22}{'n':>4}{'min':>9}{'max':>9}{'media':>9}{'sd':>9}")
    for m in sorted(d.modello.unique()):
        s = d[d.modello == m][col].dropna()
        if not len(s):
            continue
        print(f'{m:22}{len(s):>4}{s.min():>9.3f}{s.max():>9.3f}'
              f'{s.mean():>9.3f}{s.std(ddof=1) if len(s) > 1 else float("nan"):>9.3f}')
    tutti = d[col].dropna()
    print(f'{"TUTTI I MODELLI":22}{len(tutti):>4}{tutti.min():>9.3f}'
          f'{tutti.max():>9.3f}{tutti.mean():>9.3f}{tutti.std(ddof=1):>9.3f}')
    # il modello con l'escursione maggiore fra i propri cinque documenti
    esc = {m: (d[d.modello == m][col].max() - d[d.modello == m][col].min())
           for m in d.modello.unique() if d[d.modello == m][col].notna().sum() > 1}
    if esc:
        peggio = max(esc, key=esc.get)
        s = d[d.modello == peggio][col].dropna()
        print(f'\nmaggiore escursione fra documenti dello stesso modello: '
              f'{peggio}')
        print(f'   {len(s)} documenti, da {s.min():.3f} a {s.max():.3f}')
        print('   valori:', ', '.join(f'{v:.3f}' for v in sorted(s)))
    print()

# la deviazione standard fra documenti, citata come "dieci volte"
print('=' * 68)
print('DISPERSIONE FRA DOCUMENTI IN FUNZIONE DELLA TEMPERATURA (ordine 2)')
print('=' * 68)
g = O2.groupby('temperatura')[col2[0]].agg(['mean', 'std', 'min', 'max', 'size'])
print(g.round(3).to_string())
