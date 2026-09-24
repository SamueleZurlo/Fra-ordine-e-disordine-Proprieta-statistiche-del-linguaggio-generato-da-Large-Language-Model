"""
Riesegue i notebook del Capitolo 5 con la lista di parole funzione estesa.

PERCHE'
-------
Il filtro che separa parole chiave e parole funzione ammette fra le prime
qualunque parola di almeno quattro caratteri assente dalla lista di parole
funzione del notebook. Quella lista non contiene alcuni connettivi e attenuatori
inglesi molto comuni: "like", "amid", "including", "though", "also", "rather",
"since", "although", "without", "later". In Wikipedia la cosa non ha
conseguenze, perche' quei termini non entrano fra le sette parole piu' frequenti
di una voce; in Grokipedia sì, e "like" vi entra in 29 voci su 39. Il livello
delle parole chiave di Grokipedia conteneva quindi materiale che appartiene al
livello delle parole funzione.

Estendere la lista NON e' una rimozione: le parole passano dal livello delle
parole chiave a quello delle parole funzione, e al loro posto sale la parola di
contenuto successiva in ordine di frequenza. Tutti i livelli conservano cosi' la
stessa numerosita' in tutti i corpora, cosa che una semplice esclusione non
garantirebbe. La lista e' definita per classe grammaticale e si applica
identica a tutti e quattro i corpora.

COME
----
I notebook non vengono modificati. Il programma ne estrae le celle di codice,
le esegue in sequenza in uno spazio dei nomi condiviso e, subito dopo la cella
che definisce STOPWORDS, vi aggiunge l'estensione. Le cartelle di output sono
deviate su nomi con suffisso "_v2", cosi' i risultati esistenti restano
intatti finche' non si decide di adottarli.

USO
---
    py rigenera_cap5_stoplist.py                # tutti i notebook
    py rigenera_cap5_stoplist.py tre_corpora    # solo quello indicato
"""
import io
import json
import sys
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.axes

# Dalla matplotlib 3.9 il parametro `labels` di boxplot si chiama `tick_labels`,
# e dalla 3.11 il vecchio nome non e' piu' accettato. I notebook usano ormai il
# nome nuovo; la traduzione resta per chi rieseguisse una loro copia precedente.
_boxplot_orig = matplotlib.axes.Axes.boxplot


def _boxplot(self, *a, **k):
    if 'labels' in k and 'tick_labels' not in k:
        k['tick_labels'] = k.pop('labels')
    return _boxplot_orig(self, *a, **k)


matplotlib.axes.Axes.boxplot = _boxplot

BASE = Path(__file__).resolve().parents[2]
WIKI = BASE / 'Wiki'
LOG = BASE / 'Tesi' / 'bozze' / 'log_rigenerazione'
LOG.mkdir(parents=True, exist_ok=True)
SUFFISSO = '_v2'

# --- l'estensione, per classe grammaticale ---------------------------------
STOP_EXTRA = set("""amid amidst among amongst including excluding regarding
concerning despite throughout toward towards within without beyond across
around along besides versus like unlike alongside
although though whereas whether unless since thereby whilst
however moreover furthermore therefore thus hence nevertheless nonetheless
meanwhile otherwise instead also still even just rather quite often sometimes
usually generally particularly especially largely mainly mostly notably
similarly likewise accordingly subsequently previously later earlier well
many much several various certain given another every
became become becomes remains remained included includes involving""".split())

NOTEBOOK = {
    'tre_corpora': 'altmann_tre_corpora.ipynb',
    'entropia': 'entropia_intervalli.ipynb',
    'coda': 'esponente_coda.ipynb',
    'wikipedia': 'altmann_wikipedia.ipynb',
}
# eseguire in quest'ordine: esponente_coda legge sequenze.csv del primo
ORDINE = ['tre_corpora', 'entropia', 'coda', 'wikipedia']


MARCA = '\nSTOPWORDS = STOPWORDS | STOP_EXTRA_INIETTATA\n'


def inietta(src):
    """Aggiunge l'estensione subito dopo l'assegnazione di STOPWORDS.

    Non basta estendere la lista dopo l'esecuzione della cella: in tre notebook
    su quattro la stessa cella che definisce STOPWORDS esegue anche il ciclo di
    selezione dei bersagli, e la modifica arriverebbe a selezione avvenuta.
    """
    i = src.find('STOPWORDS = set(')
    if i < 0:
        return src, False
    j = src.find('""".split())', i)
    if j < 0:
        return src, False
    j += len('""".split())')
    return src[:j] + MARCA + src[j:], True


def celle_codice(path):
    nb = json.loads(Path(path).read_text(encoding='utf-8'))
    fuori, fatto = [], False
    for c in nb['cells']:
        if c['cell_type'] != 'code':
            continue
        righe = [r for r in c['source']
                 if not r.lstrip().startswith(('%', '!'))]
        src = ''.join(righe)
        if not fatto:
            src, fatto = inietta(src)
        fuori.append(src)
    if not fatto:
        raise SystemExit(f'[!] {Path(path).name}: STOPWORDS non trovata, '
                         'la lista non sarebbe stata estesa')
    return fuori


def dev(p):
    """risultati_x -> risultati_x_v2, lasciando fermo il resto del percorso."""
    p = Path(p)
    parti = list(p.parts)
    for i, q in enumerate(parti):
        if q.startswith('risultati_') and not q.endswith(SUFFISSO):
            parti[i] = q + SUFFISSO
    return Path(*parti)


def esegui(chiave):
    path = WIKI / NOTEBOOK[chiave]
    celle = celle_codice(path)
    ns = {'__name__': '__main__', 'display': lambda x: print(x),
          'STOP_EXTRA_INIETTATA': STOP_EXTRA}
    stato = {'stop': False, 'out': False, 'seq': False}
    cwd = Path.cwd()
    buf = io.StringIO()
    t0 = time.time()
    print(f'\n=== {path.name}: {len(celle)} celle di codice ===', flush=True)
    import os
    os.chdir(WIKI)
    try:
        with redirect_stdout(buf):
            for i, src in enumerate(celle):
                try:
                    exec(compile(src, f'<{path.name}#{i}>', 'exec'), ns)
                except Exception:
                    print(f'\n[!] cella {i} fallita:\n{traceback.format_exc()}')
                    raise
                # --- patch, una volta sola ciascuna ------------------------
                if not stato['stop'] and isinstance(ns.get('STOPWORDS'), set):
                    stato['stop'] = True
                    print(f'[patch] STOPWORDS estesa nel sorgente: '
                          f'{len(ns["STOPWORDS"])} parole (cella {i})')
                if not stato['out'] and isinstance(ns.get('OUT_DIR'), Path):
                    vecchio = ns['OUT_DIR']
                    ns['OUT_DIR'] = dev(vecchio)
                    for d in (ns['OUT_DIR'], ns['OUT_DIR'] / 'figure',
                              ns['OUT_DIR'] / 'dati'):
                        d.mkdir(parents=True, exist_ok=True)
                    stato['out'] = True
                    print(f'[patch] OUT_DIR: {vecchio.name} -> {ns["OUT_DIR"].name}')
                if not stato['seq'] and isinstance(ns.get('SEQUENZE'), Path):
                    ns['SEQUENZE'] = dev(ns['SEQUENZE'])
                    stato['seq'] = True
                    print(f'[patch] SEQUENZE -> {ns["SEQUENZE"]}')
                print(f'--- fine cella {i} ({time.time()-t0:.0f}s) ---')
        esito = 'OK'
    except Exception as e:
        esito = f'FALLITO: {type(e).__name__}: {e}'
    finally:
        os.chdir(cwd)
        f = LOG / f'{chiave}.log'
        f.write_text(buf.getvalue(), encoding='utf-8')
    print(f'{path.name}: {esito}  ({time.time()-t0:.0f}s)  log -> {f}', flush=True)
    return esito == 'OK'


if __name__ == '__main__':
    voluti = sys.argv[1:] or ORDINE
    for k in voluti:
        if k not in NOTEBOOK:
            print(f'sconosciuto: {k}')
            continue
        if not esegui(k):
            print(f'interrotto su {k}')
            break
