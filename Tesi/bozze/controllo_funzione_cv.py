# -*- coding: utf-8 -*-
"""Coefficiente di variazione delle parole funzione sui tre romanzi di controllo.

Il fit troncato su quel livello e' rumoroso (controllo_funzione.py), quindi la
domanda si ripete sulla grandezza piu' stabile: le parole funzione della prosa
letteraria sono piu' bursty di quelle enciclopediche, o e' un tratto di
Guerra e pace?
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from controllo_letterario import (LIBRI, N_EFF, WIKI, scarica, segmenti,
                                  prepara)  # noqa: E402

RIFERIMENTO = {'Guerra e pace': 1.192, 'Wikipedia': 1.033,
               'Grokipedia v0.1': 0.967, 'Grokipedia oggi': 0.979}


def main():
    cwd = Path.cwd()
    os.chdir(WIKI)
    try:
        ns, _ = prepara()
        print(f"nucleo pronto | STOPWORDS = {len(ns['STOPWORDS'])}\n")
        out = {}
        for nome, url in LIBRI.items():
            _, segs = segmenti(scarica(nome, url))
            per_tipo = {}
            for i, t in enumerate(segs):
                for r in ns['analizza'](t, dict(corpus=nome,
                                                titolo=f'seg_{i:02d}')):
                    per_tipo.setdefault(r['tipo'], []).append(r['cv_tau'])
            out[nome] = {k: float(np.nanmean(v)) for k, v in per_tipo.items()}
    finally:
        os.chdir(cwd)

    LIV = ['lettera', 'funzione', 'appaiata', 'keyword']
    print('=' * 66)
    print('COEFFICIENTE DI VARIAZIONE MEDIO PER LIVELLO')
    print('=' * 66)
    print(f"{'testo':22}" + ''.join(f'{l:>13}' for l in LIV))
    for nome, d in out.items():
        print(f'{nome:22}' + ''.join(
            f"{d[l]:>13.3f}" if l in d else f"{'--':>13}" for l in LIV))
    print('\ndal capitolo (parole funzione):')
    for k, v in RIFERIMENTO.items():
        print(f'  {k:22}{v:>7.3f}')


if __name__ == '__main__':
    main()
