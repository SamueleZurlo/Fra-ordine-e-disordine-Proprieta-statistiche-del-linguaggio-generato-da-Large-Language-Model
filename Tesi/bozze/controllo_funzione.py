# -*- coding: utf-8 -*-
"""Il romanzo e' l'unico corpus il cui fit sulle PAROLE FUNZIONE non degenera.

DOMANDA
-------
Nel §5.6 il corpus letterario e' l'unico dei quattro che alle parole funzione
conserva un tratto di coda a legge di potenza: tau_c = 5.3 contro 1.7--1.9 dei
tre corpora enciclopedici, cioe' sopra la soglia TAUC_MIN = 3 sotto la quale il
modello troncato degenera in una esponenziale. E' un tratto di Guerra e pace o
della prosa letteraria?

METODO
------
Identico a controllo_letterario.py, che risponde alla stessa domanda sul livello
delle parole chiave, ma il pool e' costruito sulle parole funzione. Stessi tre
romanzi, stessa segmentazione, stesse funzioni prese dai notebook.

USO
---
    py controllo_funzione.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from controllo_letterario import (LIBRI, N_EFF, MIN_TAU_SEQ, QUANTILI, WIKI,
                                  scarica, segmenti, prepara)  # noqa: E402

TAUC_MIN = 3.0          # soglia di degenerazione usata nel notebook della coda
QUANT_PRINCIPALE = 90   # quantile con cui e' costruita la Figura 5.8


def pool_tipo(ns, segs, nome, tipo):
    fuori, n_seq, parole = [], 0, set()
    for i, t in enumerate(segs):
        rec = ns['analizza'](t, dict(corpus=nome, titolo=f'seg_{i:02d}'))
        for r in rec:
            if r['tipo'] != tipo:
                continue
            pos = ns['pos_from_word'](t[:N_EFF], r['sequenza'])
            pos = pos[pos < N_EFF]
            if len(pos) < MIN_TAU_SEQ + 1:
                continue
            tau = np.diff(pos).astype(float)
            fuori.append(tau / tau.mean())
            parole.add(r['sequenza'])
            n_seq += 1
    return np.concatenate(fuori), n_seq, sorted(parole)


def main():
    cwd = Path.cwd()
    os.chdir(WIKI)
    try:
        ns, nsc = prepara()
        print(f"nucleo pronto | STOPWORDS = {len(ns['STOPWORDS'])}\n")
        ris = {}
        for nome, url in LIBRI.items():
            _, segs = segmenti(scarica(nome, url))
            x0, n_seq, parole = pool_tipo(ns, segs, nome, 'funzione')
            print(f'{nome}: {n_seq} sequenze, {len(x0):,} intervalli')
            print(f'   parole funzione selezionate: {", ".join(parole)}')
            riga = {}
            for q in QUANTILI:
                xmin = float(np.percentile(x0, q))
                x = x0[x0 >= xmin]
                if len(x) < 150:
                    continue
                mu, tc, _ = nsc['fit_tronc'](x, xmin)
                riga[q] = (mu, tc, len(x))
            ris[nome] = riga
    finally:
        os.chdir(cwd)

    print('\n' + '=' * 78)
    print('PAROLE FUNZIONE: mu e tau_c al variare del quantile di x_min')
    print('=' * 78)
    for etichetta, idx in (('mu', 0), ('tau_c', 1)):
        print(f'\n{etichetta}')
        print(f"{'testo':22}" + ''.join(f'{q:>10}' for q in QUANTILI))
        for nome, r in ris.items():
            print(f'{nome:22}' + ''.join(
                f'{r[q][idx]:>10.2f}' if q in r else f'{"--":>10}'
                for q in QUANTILI))

    print(f'\ndegenerazione al quantile {QUANT_PRINCIPALE} '
          f'(tau_c < {TAUC_MIN} => mu non interpretabile):')
    for nome, r in ris.items():
        if QUANT_PRINCIPALE in r:
            mu, tc, _ = r[QUANT_PRINCIPALE]
            stato = 'DEGENERA' if tc < TAUC_MIN else 'non degenera'
            print(f'  {nome:22} mu = {mu:5.2f}   tau_c = {tc:5.1f}   {stato}')
    print('\ndal capitolo, stessa procedura, stesso quantile:')
    print('  Guerra e pace          mu =  2.56   tau_c =   5.3   non degenera')
    print('  Wikipedia / Grokipedia                tau_c = 1.7--1.9   DEGENERA')


if __name__ == '__main__':
    main()
