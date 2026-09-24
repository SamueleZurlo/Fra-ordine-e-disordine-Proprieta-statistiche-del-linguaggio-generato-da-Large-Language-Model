# -*- coding: utf-8 -*-
"""Aggiunge mu e tau_c al blocco '60000_35' di confronto.json.

neff_sottoinsieme.py aveva ricalcolato sulle 35 terne comuni tutte le misure
del confronto tranne il fit della coda, cosicche' il §5.9 dichiarava di
riportare a 60000 caratteri valori calcolati sulle stesse 35 terne mentre i tre
esponenti che cita venivano ancora dal blocco a 39. Questo programma rifa' il
fit sul sottoinsieme, riusando coda_keyword di confronto_neff perche' la catena
resti identica a quella del resto del capitolo.

USO
---
    py neff_coda_sottoinsieme.py
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import confronto_neff as cn

DIR = cn.WIKI / 'confronto_N_eff_v2'
N = 60_000


def main():
    a = pd.read_csv(DIR / 'sequenze_N60k.csv')
    b = pd.read_csv(DIR / 'sequenze_N80k.csv')
    comuni = {c: set(a[a.corpus == c].titolo) & set(b[b.corpus == c].titolo)
              for c in cn.ORDINE}
    print('titoli comuni: ' + ', '.join(
        '%s %d' % (c, len(comuni[c])) for c in cn.ORDINE))

    cwd = Path.cwd()
    os.chdir(cn.WIKI)
    try:
        ns, celle = cn.prepara_misura()
        nsc = cn.prepara_coda()
        ns['N_EFF'] = N
        exec(compile(celle[4], '<tre_corpora#4>', 'exec'), ns)
        USABILI, SEG, TESTI = ns['USABILI'], ns['SEG'], ns['TESTI']

        testi = {}
        for c in ('wikipedia', 'grok_v01', 'grok_oggi'):
            for t in USABILI:
                testi[(c, t)] = TESTI[t][c]
        for i, s in enumerate(SEG):
            testi[('letterario', 'wrnpc_%02d' % i)] = s

        # stesse righe di sequenze_N60k.csv, ristrette ai titoli comuni
        sub = a[[t in comuni[c] for c, t in zip(a.corpus, a.titolo)]]
        fuori = cn.coda_keyword(sub, testi, ns, nsc, N)
    finally:
        os.chdir(cwd)

    R = json.loads((DIR / 'confronto.json').read_text(encoding='utf-8'))
    blocco = R['60000_35']
    print()
    print('%-12s %-16s %-16s' % ('corpus', 'mu  39 -> 35', 'tau_c  39 -> 35'))
    for c in cn.ORDINE:
        if c not in fuori:
            print('%-12s fit non riuscito' % c)
            continue
        mu, tc, n = fuori[c]
        print('%-12s %6.2f -> %6.2f   %6.1f -> %6.1f   (n=%d)'
              % (c, R['60000']['mu_' + c], mu,
                 R['60000']['tauc_' + c], tc, n))
        blocco['mu_' + c], blocco['tauc_' + c], blocco['ntau_' + c] = mu, tc, n
    (DIR / 'confronto.json').write_text(json.dumps(R, indent=1), encoding='utf-8')
    print()
    print('-> ' + str(DIR / 'confronto.json'))


if __name__ == '__main__':
    main()
