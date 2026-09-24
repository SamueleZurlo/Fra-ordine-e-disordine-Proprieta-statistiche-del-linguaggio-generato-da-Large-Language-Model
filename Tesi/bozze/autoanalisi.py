# -*- coding: utf-8 -*-
"""Il protocollo della Figura 5.4 applicato al testo della tesi stessa.

Nessun file della tesi viene modificato: i sorgenti .tex sono letti in sola
lettura, ripuliti della marcatura e trattati come un corpus qualsiasi.

DIFFERENZE RISPETTO AI CORPORA DEL CAPITOLO 5
---------------------------------------------
La lista chiusa di parole funzione e' quella italiana, non quella inglese: e'
l'unico parametro che cambia, e cambia perche' deve. Tutto il resto -- i
quattro livelli, N_EFF, i lag, la finestra di fit, i due null model, la regola
di selezione dei bersagli -- viene preso dal notebook senza riscritture.

USO
---
    py autoanalisi.py
"""
import io
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                              errors='replace')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rigenera_cap5_stoplist import celle_codice  # noqa: E402

RAD = Path(__file__).resolve().parent.parent
BASE = RAD.parent
WIKI = BASE / 'Wiki'
OUT = RAD / 'bozze' / 'anteprima'
OUT.mkdir(parents=True, exist_ok=True)

CAPITOLI = ['capitolo1_introduzione.tex', 'capitolo2_quadro_teorico.tex',
            'capitolo3_materiali_metodi.tex', 'capitolo4_sweep_temperatura.tex',
            'capitolo5_burstiness.tex', 'capitolo6_discussione_conclusioni.tex']

# ---------------------------------------------------------------- stoplist IT
STOP_IT = set("""
a ad affinche agli ai al alcuna alcune alcuni alcuno all alla alle allo altre altri
altro anche ancora anzi appena avanti avendo avere avesse avessero avete aveva
avevano avevo avra avranno avrebbe avrebbero avro avuta avute avuti avuto
c che chi ci cio cioe circa coi col colui come con contro cosa cosi cui
d da dagli dai dal dall dalla dalle dallo degli dei del dell della delle dello
dentro di dopo dove dovrebbe due dunque durante
e ebbe ebbero ecco ed egli entrambi era erano ero esse essendo essere essi
fa fanno fara farebbe fatto fin finche fino fosse fossero foste fra fu furono
gia gli grazie
ha hai hanno ho
i il in infatti inoltre insieme intanto invece io
la le lei li lo loro lui
ma me meno mentre mi mia mie miei mio molta molte molti molto
ne nei nel nell nella nelle nello nemmeno neppure nessun nessuna nessuno niente
no noi non nondimeno nostra nostre nostri nostro nulla
o od oltre ogni ognuna ognuno oppure ora ossia ovvero ovunque
per percio perche piu poco poi poiche presso prima pur purche
qua quale quali qualunque quando quanta quante quanti quanto quasi quel quella
quelle quelli quello questa queste questi questo qui quindi
sara saranno sarebbe sarebbero saremo sarei se sebbene secondo sei sempre senza
si sia siamo siano siate siete solo sono sopra sotto sta stanno stata state
stati stato stesse stessi stesso stia su sua sue sugli sui sul sull sulla sulle
sullo suo suoi
tale tali tanto te tra tranne troppo tu tua tue tuoi tuo tutta tutte tutti tutto
un una uno
va vale verso vi via voi vostra vostre vostri vostro
""".split())


# ------------------------------------------------------------ estrazione testo
MATE = [
    (re.compile(r'\\begin\{(equation|align|gather|tabular|table|figure|'
                r'itemize|enumerate|verbatim|lstlisting)\*?\}.*?'
                r'\\end\{\1\*?\}', re.S), ' '),
    (re.compile(r'\$\$.*?\$\$', re.S), ' '),
    (re.compile(r'\$[^$]*\$'), ' '),
    (re.compile(r'\\\[.*?\\\]', re.S), ' '),
]
COMANDI_CON_TESTO = ('emph', 'textbf', 'textit', 'textsc', 'section',
                     'subsection', 'paragraph', 'chapter', 'caption', 'num')
VIA = re.compile(r'\\(?:cite|ref|eqref|label|includegraphics|input|vspace|'
                 r'hspace|newcommand|renewcommand|usepackage|documentclass|'
                 r'bibitem|url|href|footnote)\s*(\[[^\]]*\])?\s*(\{[^{}]*\})*')


def spoglia(s):
    """Toglie la marcatura e lascia la prosa."""
    s = re.sub(r'(?<!\\)%.*', '', s)                 # commenti
    for rx, sub in MATE:
        s = rx.sub(sub, s)
    s = VIA.sub(' ', s)
    # comandi di cui si tiene l'argomento
    for _ in range(6):
        s = re.sub(r'\\(' + '|'.join(COMANDI_CON_TESTO) + r')\*?\{([^{}]*)\}',
                   r'\2', s)
    s = re.sub(r'\\[a-zA-Z]+\*?', ' ', s)            # comandi residui
    s = s.replace('~', ' ').replace('--', '-')
    s = re.sub(r'[{}\\&]', ' ', s)
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n\s*\n+', '\n\n', s)
    return s.strip()


def testo_tesi():
    pezzi = []
    for n in CAPITOLI:
        pezzi.append(spoglia((RAD / n).read_text(encoding='utf-8')))
    return '\n\n'.join(pezzi)


# ------------------------------------------------------------- nucleo di misura
def nucleo():
    celle = celle_codice(WIKI / 'altmann_tre_corpora.ipynb')
    ns = {'__name__': '__main__', 'display': lambda *a, **k: None,
          'STOP_EXTRA_INIETTATA': set()}
    for i in (0, 1, 2):
        exec(compile(celle[i], f'<tre#{i}>', 'exec'), ns)
    ns['STOPWORDS'] = STOP_IT           # l'unica sostituzione
    return ns


def main():
    testo = testo_tesi()
    print(f'testo della tesi ripulito: {len(testo):,} caratteri, '
          f'{len(testo.split()):,} parole\n')
    print('inizio:', testo[:150].replace('\n', ' '), '\n')

    cwd = Path.cwd()
    os.chdir(WIKI)
    try:
        ns = nucleo()
        N_EFF = ns['N_EFF']
        # stessa regola del riferimento letterario nel notebook:
        #   SEG = [BODY[s:s+N_EFF] for s in linspace(0, len-N_EFF, N_SEG)]
        N_SEG = 20
        inizi = np.linspace(0, max(len(testo) - N_EFF, 0), N_SEG).astype(int)
        passo = int(np.diff(inizi).mean()) if len(inizi) > 1 else 0
        sovr = max(0, N_EFF - passo)
        print(f'N_EFF = {N_EFF:,} | {N_SEG} finestre, passo {passo:,} caratteri')
        print(f'sovrapposizione fra finestre contigue: {sovr:,} caratteri '
              f'({sovr/N_EFF:.0%} della finestra)')
        print(f'STOPWORDS italiane: {len(STOP_IT)}\n')
        n_seg = N_SEG
        rec = []
        for i, a in enumerate(inizi):
            seg = testo[a:a + N_EFF]
            rec += ns['analizza'](seg, dict(corpus='tesi',
                                            titolo=f'seg_{i:02d}'))
        print(f'{len(rec)} sequenze misurate')
    finally:
        os.chdir(cwd)

    import pandas as pd
    D = pd.DataFrame(rec)
    D.to_csv(OUT / 'autoanalisi_tesi.csv', index=False)

    # ------------------------------------------------------------- il piano
    TIPI = [('lettera', 'lettere', 'o', '#888888'),
            ('funzione', 'parole funzione', 's', '#0072B2'),
            ('appaiata', 'controlli appaiati', '^', '#E69F00'),
            ('keyword', 'parole chiave', 'D', '#D55E00')]
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    for tipo, nome, mk, col in TIPI:
        s = D[D.tipo == tipo]
        ax.scatter(s.cv_tau, s.gamma, s=42, marker=mk, color=col,
                   edgecolors='k', linewidths=.4, alpha=.8, label=nome)
    ax.axvline(1.0, color='grey', ls='--', lw=1.2)
    ax.axhline(1.0, color='grey', ls='--', lw=1.2)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\sigma_\tau/\langle\tau\rangle$')
    ax.set_ylabel(r'$\hat\gamma$')
    ax.set_title('Il piano della Figura 5.4, misurato sul testo della tesi\n'
                 f'{n_seg} finestre da {N_EFF//1000}k caratteri (sovrapposte), '
                 f'{len(D)} sequenze', fontsize=11)
    ax.legend(fontsize=8, loc='lower right')
    # etichette dei quadranti in coordinate d'asse, per non deformare i limiti
    for fx, fy, t, ha, va in [(.02, .97, 'alto-sx: correlate, non bursty', 'left', 'top'),
                              (.98, .97, 'alto-dx: canonico', 'right', 'top'),
                              (.02, .03, 'basso-sx: regolari e scorrelate', 'left', 'bottom'),
                              (.98, .03, 'basso-dx: atteso vuoto', 'right', 'bottom')]:
        ax.text(fx, fy, t, transform=ax.transAxes, fontsize=8.5,
                color='#555555', ha=ha, va=va, style='italic')
    # le sole parole chiave fuori dal quadrante canonico vengono nominate
    for r in D[(D.tipo == 'keyword') &
               ((D.cv_tau < 1) | (D.gamma < 1))].itertuples():
        ax.annotate(r.sequenza, (r.cv_tau, r.gamma), fontsize=8,
                    xytext=(6, -3), textcoords='offset points', color='#D55E00')
    fig.tight_layout()
    fig.savefig(OUT / 'autoanalisi_piano.png', dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------- i quadranti
    print('\n' + '=' * 74)
    print('COMPOSIZIONE PER QUADRANTE (soglie: cv = 1, gamma = 1)')
    print('=' * 74)
    print(f"{'livello':20}{'alto-dx':>9}{'alto-sx':>9}{'basso-sx':>10}"
          f"{'basso-dx':>10}{'n':>6}")
    for tipo, nome, _, _ in TIPI:
        s = D[D.tipo == tipo]
        if not len(s):
            continue
        ad = ((s.cv_tau >= 1) & (s.gamma >= 1)).mean()
        asx = ((s.cv_tau < 1) & (s.gamma >= 1)).mean()
        bsx = ((s.cv_tau < 1) & (s.gamma < 1)).mean()
        bd = ((s.cv_tau >= 1) & (s.gamma < 1)).mean()
        print(f'{nome:20}{ad:>8.1%}{asx:>9.1%}{bsx:>10.1%}{bd:>10.1%}{len(s):>6}')

    print('\n' + '=' * 74)
    print('PAROLE CHIAVE NEL QUADRANTE IN BASSO A SINISTRA')
    print('=' * 74)
    k = D[D.tipo == 'keyword']
    bsx = k[(k.cv_tau < 1) & (k.gamma < 1)]
    print(f'{len(bsx)} sequenze su {len(k)} '
          f'({len(bsx)/max(len(k),1):.1%})\n')
    if len(bsx):
        print(f"{'parola':18}{'segmento':10}{'n':>6}{'cv':>8}{'gamma':>8}")
        for r in bsx.sort_values('cv_tau').itertuples():
            print(f'{r.sequenza:18}{r.titolo:10}{r.n_eventi:>6}'
                  f'{r.cv_tau:>8.3f}{r.gamma:>8.3f}')
        print('\nconteggio per parola (in quante finestre vi ricade):')
        for w, n in Counter(bsx.sequenza).most_common():
            print(f'  {w:20} {n}')

    print('\n' + '=' * 74)
    print('ANGOLO IN BASSO A DESTRA (bursty ma scorrelate): deve essere vuoto')
    print('=' * 74)
    bd = D[(D.cv_tau > 1.5) & (D.gamma < 1.1)]
    print(f'{len(bd)} sequenze su {len(D[D.cv_tau > 1.5])} con cv > 1.5')
    if len(bd):
        print(bd[['sequenza', 'tipo', 'cv_tau', 'gamma']].to_string(index=False))

    print('\n' + '=' * 74)
    print('MEDIE PER LIVELLO (confrontabili con la Tabella del §5.3)')
    print('=' * 74)
    print(f"{'livello':20}{'cv':>9}{'gamma':>9}{'gamma_A1':>10}{'gamma_A2':>10}")
    for tipo, nome, _, _ in TIPI:
        s = D[D.tipo == tipo]
        if not len(s):
            continue
        print(f'{nome:20}{s.cv_tau.mean():>9.3f}{s.gamma.mean():>9.3f}'
              f'{s.gamma_A1.mean():>10.3f}{s.gamma_A2.mean():>10.3f}')

    print('\n' + '=' * 74)
    print('LA TESI ACCANTO AI QUATTRO CORPORA DEL CAPITOLO 5')
    print('=' * 74)
    liv = pd.read_csv(BASE / 'Wiki' / 'risultati_tre_corpora_v2' /
                      'dati' / 'livelli.csv')
    ETI = {'letterario': 'Guerra e pace', 'wikipedia': 'Wikipedia',
           'grok_v01': 'Grokipedia v0.1', 'grok_oggi': 'Grokipedia oggi'}
    print(f"{'corpus':20}{'lettere':>10}{'funzione':>10}"
          f"{'appaiati':>10}{'keyword':>10}")
    for c in ['letterario', 'wikipedia', 'grok_v01', 'grok_oggi']:
        r = []
        for t in ['lettera', 'funzione', 'appaiata', 'keyword']:
            v = liv[(liv.corpus == c) & (liv.tipo == t)]['cv']
            r.append(float(v.iloc[0]) if len(v) else float('nan'))
        print(f'{ETI[c]:20}' + ''.join(f'{x:>10.3f}' for x in r))
    r = [D[D.tipo == t].cv_tau.mean()
         for t in ['lettera', 'funzione', 'appaiata', 'keyword']]
    print(f'{"QUESTA TESI":20}' + ''.join(f'{x:>10.3f}' for x in r))

    print('\n' + '=' * 74)
    print('BERSAGLI SELEZIONATI, per segmento')
    print('=' * 74)
    for t in sorted(D.titolo.unique()):
        s = D[D.titolo == t]
        for tipo, nome, _, _ in TIPI[1:]:
            w = list(s[s.tipo == tipo].sequenza)
            print(f'  {t}  {nome:20} {", ".join(w)}')
        print()

    print(f'-> {OUT / "autoanalisi_piano.png"}')


if __name__ == '__main__':
    main()
