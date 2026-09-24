"""
Lunghezza del ciclo ripetuto in fase di stallo, contro l'entropia di parola.

MOTIVAZIONE
-----------
Il sesto pannello della Figura 4.8 (piano entropia--comprimibilita', asse della
Fig. 1A di [C]) mostra una nuvola di documenti a comprimibilita' bassissima,
R_gzip fra 0.007 e 0.020, che pero' si distende su un intervallo di entropia di
parola molto ampio, da 0.28 a 0.85. Il capitolo attribuisce questa dispersione
al fatto che le due grandezze misurano cose diverse: il rapporto di compressione
dipende dal FATTO che la sequenza si ripeta, l'entropia normalizzata dalla forma
della distribuzione di frequenza sul vocabolario che il ciclo contiene. Un ciclo
lungo e vario e un ciclo corto e povero sarebbero ugualmente comprimibili ma con
entropie molto diverse.

Questo programma misura direttamente la grandezza che quell'argomento invoca
senza averla mai misurata -- la lunghezza del ciclo -- e ne verifica le due
conseguenze:

    (i)  L_ciclo deve spiegare la posizione ORIZZONTALE nel pannello, cioe'
         correlare con l'entropia di parola;
    (ii) L_ciclo NON deve spiegare la posizione VERTICALE, cioe' essere
         scorrelata da R_gzip.

METODO
------
Il testo e' preparato esattamente come nel pannello: primi N_TOK = 20000 token
cl100k_base, decodifica, parole = re.findall(r'\\w+', sub(r'\\d', ' ', lower)).
L'entropia e' ricalcolata con la stessa funzione del notebook, non letta dal
CSV, cosi' il confronto e' interno a questo programma; il CSV del notebook serve
solo da controllo incrociato in coda al rapporto.

Il periodo si misura sulla sequenza di parole. Il periodo fondamentale e' il
PIU' PICCOLO sfasamento p per cui l'accordo w[i] == w[i-p], misurato sugli
ultimi 20*p simboli del documento, sta sopra SOGLIA. La finestra e' locale e
commisurata al periodo candidato per due ragioni opposte: su tutto il testo la
prosa che precede il ciclo diluirebbe la media, mentre su un suffisso lungo un
ciclo che cambia lentamente variante scende sotto soglia al suo periodo vero e
la supera a un multiplo, dove le varianti tornano in fase, restituendo un alias.
Prendere il piu' piccolo p, e non quello di copertura massima, scarta a sua
volta i multipli.

L'inizio del ciclo si determina poi risalendo dalla fine finche' l'accordo
mediato su una finestra corta resta sopra soglia.

La soglia 0.90 anziche' 1.0 tollera i cicli non esattamente verbatim (una parola
che cambia a ogni giro). L'accordo atteso per caso in prosa inglese e' circa
0.05, quindi non c'e' rischio di falsi positivi.

USO
---
    py analisi_ciclo_stallo.py                 # figura in bozze/anteprima
    py analisi_ciclo_stallo.py --applica       # figura in Tesi/figure
    py analisi_ciclo_stallo.py --ricalcola     # ignora la cache delle misure
"""
import collections
import gzip
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from scipy import stats as sps
except ImportError:
    sps = None

# ---------------------------------------------------------------------------
# 1. Configurazione: identica a quella del notebook per tutto cio' che tocca
#    la preparazione del testo.
# ---------------------------------------------------------------------------
RADICE = Path(__file__).resolve().parents[2]
RIS = RADICE / 'Analisi Sweep di temperature' / 'Risultati Sweep'
OUT = (RADICE / 'Tesi' / 'figure') if '--applica' in sys.argv \
    else (RADICE / 'Tesi' / 'bozze' / 'anteprima')
OUT.mkdir(parents=True, exist_ok=True)
CSV = RIS / 'ciclo_stallo.csv'

TOKENIZER = 'cl100k_base'
N_TOK = 20000            # troncamento comune del capitolo
MIN_TOK = 2000           # sotto questa soglia il documento e' escluso

MODELLI = {
    'Qwen2.5-1.5B':    'sweep_qwen2.5-1.5b.jsonl',
    'Qwen2.5-3B':      'sweep_qwen2.5-3b.jsonl',
    'Qwen2.5-14B':     'sweep_qwen2.5-14b.jsonl',
    'Qwen3.5-2B-Base': 'sweep_qwen3.5-2b-base.jsonl',
    'Qwen3.5-4B-Base': 'sweep_qwen3.5-4b-base.jsonl',
}
ESCLUDI = ('_short', '_failed', '.bak')

MARKER_MODELLO = {'Qwen2.5-1.5B': 'o', 'Qwen2.5-3B': 's', 'Qwen2.5-14B': '^',
                  'Qwen3.5-2B-Base': 'D', 'Qwen3.5-4B-Base': 'v'}
COLORE_UMANO = '#000000'

# --- parametri della misura del ciclo --------------------------------------
SOGLIA = 0.90         # accordo minimo perche' il suffisso sia considerato ciclico
MIN_PERIODI = 3       # periodi interi richiesti come prova della periodicita'
COPERTURA_MIN = 0.20  # frazione minima del documento occupata dal ciclo
P_CAP = 5000          # periodo massimo esplorato, in parole
GIRI_TEST = 20        # giri di ciclo su cui si verifica un periodo candidato
FINESTRA_MIN = 200    # parole minime della finestra di verifica

plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.25,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False,
})


# ---------------------------------------------------------------------------
# 2. Preparazione del testo e entropia: copiate dal notebook senza modifiche
# ---------------------------------------------------------------------------
def parole_con_posizione(testo):
    """Le stesse parole di parole() del notebook, ma con l'offset di ciascuna.

    parole() applica lower() e sostituisce ogni cifra con UNO spazio: entrambe
    le trasformazioni conservano la lunghezza, quindi gli offset trovati sul
    testo trasformato valgono anche sull'originale. L'unica eccezione sono i
    pochi caratteri Unicode che si allungano sotto lower(); il caso viene
    verificato e, se si presenta, la lunghezza in caratteri e' marcata NaN.
    """
    t = re.sub(r'\d', ' ', testo.lower())
    coerente = (len(t) == len(testo))
    m = list(re.finditer(r'\w+', t))
    off = np.array([x.start() for x in m], dtype=np.int64)
    return [x.group() for x in m], off, coerente


def entropia_normalizzata(seq):
    """[C] sec.4.1.1, identica alla cella 15 del notebook: entropia di Shannon
    sulle frequenze, normalizzata su log(numero di tipi)."""
    if not len(seq):
        return np.nan
    c = collections.Counter(seq)
    p = np.array(list(c.values()), dtype=float) / len(seq)
    if len(c) < 2:
        return 0.0
    return float(-(p * np.log(p)).sum() / np.log(len(c)))


def gzip_ratio(testo):
    """[C] eq.(4): R(x) = C(x)/|x| con gzip sui byte UTF-8."""
    b = testo.encode('utf-8')
    return len(gzip.compress(b, 9)) / len(b) if b else np.nan


# ---------------------------------------------------------------------------
# 3. Misura del periodo del ciclo
# ---------------------------------------------------------------------------
def accordo_locale(codici, p, giri=GIRI_TEST, minimo=FINESTRA_MIN):
    """Accordo a sfasamento p misurato SOLO in fondo al documento.

    La finestra e' commisurata al periodo candidato: gli ultimi giri*p simboli,
    e comunque non meno di `minimo`. Misurare l'accordo su tutto il documento
    sarebbe sbagliato in due modi opposti. Su un suffisso lungo, un ciclo che
    cambia lentamente variante scende sotto soglia al suo periodo vero e la
    supera invece a un multiplo, dove le varianti tornano in fase: il periodo
    riportato diventa un alias (13 parole lette come 273). Su tutto il testo,
    la prosa che precede il ciclo diluisce la media.
    """
    n = len(codici)
    if p >= n:
        return 0.0
    w = int(min(n - p, max(minimo, giri * p)))
    a = codici[n - w:]
    if len(a) <= p:
        return 0.0
    return float(np.mean(a[p:] == a[:-p]))


def inizio_ciclo(codici, p, soglia=SOGLIA):
    """Indice in cui il ciclo di periodo p comincia davvero.

    copertura_periodica() usa una media sull'intero suffisso, quindi tollera che
    l'accordo sia nullo su un tratto iniziale purche' la media complessiva resti
    alta: con un ciclo che occupa il 96% del testo, il criterio si estende
    all'indietro fino a inglobare anche la prosa che lo precede. Per l'inizio
    serve invece un criterio LOCALE: si risale dalla fine finche' l'accordo
    mediato su una finestra corta resta sopra soglia, e ci si ferma all'ultimo
    punto in cui scende.
    """
    m = (codici[p:] == codici[:-p]).astype(float)   # m[j] riguarda la posizione j+p
    w = int(min(500, max(50, 2 * p)))
    if m.size < w:
        return len(codici) - m.size
    mv = np.convolve(m, np.ones(w) / w, mode='valid')   # mv[j] copre j..j+w-1
    sotto = np.nonzero(mv < soglia)[0]
    if not sotto.size:
        return p
    return int(min(len(codici), sotto[-1] + w + p))


def misura_ciclo(parole_doc):
    """Periodo fondamentale del ciclo terminale, o None se il documento non
    stalla. Restituisce (p, k, inizio), con inizio = indice di partenza del
    ciclo nella lista di parole."""
    n = len(parole_doc)
    if n < 100:
        return None
    codici = np.unique(np.asarray(parole_doc), return_inverse=True)[1].astype(np.int32)
    p_max = min(P_CAP, n // MIN_PERIODI)
    if p_max < 1:
        return None

    # il periodo fondamentale e' il PIU' PICCOLO p che regge in fondo al testo:
    # i suoi multipli reggono altrettanto bene e vanno scartati
    p = None
    for q in range(1, p_max + 1):
        if n >= (MIN_PERIODI + 1) * q and accordo_locale(codici, q) >= SOGLIA:
            p = q
            break
    if p is None:
        return None
    s = inizio_ciclo(codici, p)
    k = n - s - p                      # posizioni effettivamente cicliche
    if k < (MIN_PERIODI - 1) * p or (n - s) < COPERTURA_MIN * n:
        return None
    return p, k, s


# ---------------------------------------------------------------------------
# 4. Raccolta delle misure su tutti i documenti
# ---------------------------------------------------------------------------
def carica_documenti():
    import tiktoken
    enc = tiktoken.get_encoding(TOKENIZER)
    righe = []
    for etichetta, pattern in MODELLI.items():
        cand = [p for p in sorted(RADICE.glob(pattern))
                if not any(x in p.name for x in ESCLUDI)]
        if not cand:
            print(f"  [ ] {etichetta:16s} nessun file per '{pattern}'")
            continue
        f = max(cand, key=lambda p: p.stat().st_mtime)
        print(f"  [x] {etichetta:16s} {f.name}")
        with open(f, encoding='utf-8') as fh:
            for i, linea in enumerate(fh):
                linea = linea.strip()
                if not linea:
                    continue
                r = json.loads(linea)
                ids = enc.encode(r['generated_text'], disallowed_special=())[:N_TOK]
                if len(ids) < MIN_TOK:
                    continue
                righe.append(dict(modello=etichetta,
                                  temperatura=r['temperature'],
                                  sample_id=r.get('sample_id', i),
                                  testo=enc.decode(ids)))
    return righe


def misura_tutto(righe):
    out = []
    t0 = time.time()
    for j, r in enumerate(righe):
        par, off, coerente = parole_con_posizione(r['testo'])
        d = dict(modello=r['modello'], temperatura=r['temperatura'],
                 sample_id=r['sample_id'], n_parole=len(par),
                 H_parola_norm=entropia_normalizzata(par),
                 R_gzip=gzip_ratio(r['testo']))
        res = misura_ciclo(par)
        if res is None:
            d.update(stallo=False, L_ciclo=np.nan, L_ciclo_char=np.nan,
                     V_ciclo=np.nan, H_ciclo=np.nan, frazione_stallo=np.nan,
                     accordo=np.nan, estratto='')
        else:
            p, k, s = res
            blocco = par[s:s + p]
            if coerente and (s + p) < len(off):
                a, b = int(off[s]), int(off[s + p])
                lc, testo_ciclo = b - a, r['testo'][a:b]
            else:
                lc, testo_ciclo = np.nan, ' '.join(blocco)
            cod = np.unique(np.asarray(par), return_inverse=True)[1][s:]
            acc = float(np.mean(cod[p:] == cod[:-p])) if len(cod) > p else np.nan
            d.update(stallo=True, L_ciclo=p, L_ciclo_char=lc,
                     V_ciclo=len(set(blocco)), H_ciclo=entropia_normalizzata(blocco),
                     frazione_stallo=(k + p) / len(par), accordo=acc,
                     estratto=re.sub(r'\s+', ' ', testo_ciclo)[:400])
        out.append(d)
        print(f"  {j+1}/{len(righe)}  ({time.time()-t0:.0f}s)", end='\r')
    print()
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# 5. Figura
# ---------------------------------------------------------------------------
def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if sps is None or m.sum() < 5:
        return np.nan, np.nan, int(m.sum())
    r, p = sps.spearmanr(x[m], y[m])
    return float(r), float(p), int(m.sum())


def spearman_a_T_fissa(df, col):
    """Correlazione dentro ciascuna temperatura, poi combinata.

    Sia L_ciclo sia l'entropia calano al crescere di T, quindi una correlazione
    calcolata su tutti i documenti insieme potrebbe essere prodotta dalla sola
    temperatura. Qui il legame si misura fra documenti generati alla STESSA
    temperatura, cosi' quel canale e' chiuso; le correlazioni per gruppo si
    combinano con la trasformata z di Fisher pesata sui gradi di liberta'.
    """
    zs, ws, per_gruppo = [], [], []
    for T, g in df.groupby('temperatura'):
        r, p, n = spearman(g.L_ciclo, g[col])
        if not np.isfinite(r) or n < 6 or abs(r) >= 1:
            continue
        per_gruppo.append((T, r, n))
        zs.append(np.arctanh(r))
        ws.append(n - 3)
    if not zs:
        return np.nan, np.nan, per_gruppo
    z = float(np.average(zs, weights=ws))
    se = 1.0 / np.sqrt(np.sum(ws))
    p = 2 * sps.norm.sf(abs(z) / se) if sps is not None else np.nan
    return float(np.tanh(z)), float(p), per_gruppo


def testo_rho(x, y):
    r, p, n = spearman(x, y)
    if not np.isfinite(r):
        return ''
    ps = 'p < 10^{-6}' if p < 1e-6 else f'p = {p:.2g}'
    return rf'Spearman $\rho = {r:+.2f}$, ${ps}$  ($n = {n}$)'


def figura(DF, UM):
    st = DF[DF.stallo].copy()
    ns = DF[~DF.stallo]
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(DF.temperatura.min(), DF.temperatura.max())

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.6))

    # --- A: il pannello di riferimento, con lo stallo evidenziato -----------
    ax = axes[0]
    for m, mk in MARKER_MODELLO.items():
        d = ns[ns.modello == m]
        ax.scatter(d.H_parola_norm, d.R_gzip, s=30, marker=mk,
                   facecolors='none', edgecolors=cmap(norm(d.temperatura)),
                   linewidths=1.0)
        d = st[st.modello == m]
        ax.scatter(d.H_parola_norm, d.R_gzip, s=34, marker=mk,
                   c=cmap(norm(d.temperatura)), edgecolors='none', alpha=.9)
    ax.scatter([UM['H_parola_norm']], [UM['R_gzip']], s=260, marker='*',
               color=COLORE_UMANO, zorder=6)
    ax.annotate('umano', (UM['H_parola_norm'], UM['R_gzip']),
                textcoords='offset points', xytext=(-6, 10), ha='right', fontsize=9)
    ax.set_yscale('log')
    ax.set_xlabel('entropia normalizzata a livello di parola')
    ax.set_ylabel(r'$R_{\mathrm{gzip}}$')
    ax.set_title('A. Il piano della Fig. 4.8, sesto pannello\n'
                 f'pieno = in stallo ({len(st)}), vuoto = non in stallo ({len(ns)})',
                 fontsize=10)
    voci = [Line2D([], [], ls='', marker=mk, color='#666666', ms=6, label=m)
            for m, mk in MARKER_MODELLO.items()]
    voci.append(Line2D([], [], ls='', marker='*', color=COLORE_UMANO, ms=12,
                       label='umano'))
    ax.legend(handles=voci, fontsize=6.5, loc='lower right')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cax = ax.inset_axes([0.06, 0.90, 0.34, 0.035])
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_label('temperatura $T$', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # --- B e C: cosa spiega la lunghezza del ciclo --------------------------
    for ax, col, ylab, logy, titolo in (
            (axes[1], 'H_parola_norm', 'entropia normalizzata a livello di parola',
             False, "B. Il ciclo spiega l'asse orizzontale"),
            (axes[2], 'R_gzip', r'$R_{\mathrm{gzip}}$',
             True, 'C. Il ciclo non spiega quello verticale')):
        for m, mk in MARKER_MODELLO.items():
            d = st[st.modello == m]
            ax.scatter(d.L_ciclo, d[col], s=40, marker=mk,
                       c=cmap(norm(d.temperatura)), edgecolors='none', alpha=.9)
        ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')
        # mediana per ottava di lunghezza del ciclo
        bordi = 2.0 ** np.arange(0, np.ceil(np.log2(max(st.L_ciclo.max(), 2))) + 1)
        cen, med = [], []
        for lo, hi in zip(bordi[:-1], bordi[1:]):
            g = st[(st.L_ciclo >= lo) & (st.L_ciclo < hi)]
            if len(g) >= 3:
                cen.append(np.sqrt(lo * hi))
                med.append(g[col].median())
        if len(cen) >= 2:
            ax.plot(cen, med, color='#b2182b', lw=2.0, marker='o', ms=4,
                    zorder=8, label='mediana per ottava')
        ax.axhline(UM[col], color=COLORE_UMANO, lw=1.6, ls='--', label='umano')
        ax.legend(fontsize=7.5, loc='lower right')
        ax.set_xlabel(r'$L_{\mathrm{ciclo}}$  (parole per giro)')
        ax.set_ylabel(ylab)
        rp, pp, _ = spearman_a_T_fissa(st, col)
        sotto = (rf'a $T$ fissata: $\rho = {rp:+.2f}$, '
                 + ('$p < 10^{-6}$' if pp < 1e-6 else f'$p = {pp:.2g}$'))
        ax.set_title(titolo + '\n' + testo_rho(st.L_ciclo, st[col]) + '\n' + sotto,
                     fontsize=9.5)

    for a in axes:
        a.set_box_aspect(1)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'cap4_ciclo_entropia.{ext}')
    plt.close(fig)
    print(f"-> {OUT / 'cap4_ciclo_entropia.png'}")


# ---------------------------------------------------------------------------
# 6. Esecuzione
# ---------------------------------------------------------------------------
def baseline_umana():
    """Il testo umano di riferimento, trattato esattamente come i generati."""
    import tiktoken
    enc = tiktoken.get_encoding(TOKENIZER)
    cand = [RADICE / 'Analisi Sweep di temperature' / 'cache' / 'war_and_peace.txt',
            RADICE / 'corpora_cache' / 'f6c229e9b581.txt']
    f = next((c for c in cand if c.exists()), None)
    if f is None:
        raise SystemExit('Testo umano di riferimento non trovato.')
    testo = f.read_text(encoding='utf-8', errors='replace')
    i = testo.find('CHAPTER I')          # salta l'intestazione Gutenberg
    if i > 0:
        testo = testo[i:]
    testo = enc.decode(enc.encode(testo, disallowed_special=())[:N_TOK])
    par, _, _ = parole_con_posizione(testo)
    res = misura_ciclo(par)
    return dict(H_parola_norm=entropia_normalizzata(par), R_gzip=gzip_ratio(testo),
                stallo=res is not None, L_ciclo=(res[0] if res else np.nan))


def rapporto(DF, UM):
    st = DF[DF.stallo]
    print('\n' + '=' * 72)
    print(f"Documenti in stallo: {len(st)} su {len(DF)} ({100*len(st)/len(DF):.0f}%)")
    if not len(st):
        return
    print(f"  accordo medio dentro il ciclo : {st.accordo.mean():.3f}")
    print(f"  frazione ciclica del documento: mediana {st.frazione_stallo.median():.2f}, "
          f"minimo {st.frazione_stallo.min():.2f}")
    print(f"  L_ciclo (parole)              : min {int(st.L_ciclo.min())}, "
          f"mediana {int(st.L_ciclo.median())}, max {int(st.L_ciclo.max())}")
    ok = st.L_ciclo_char.notna()
    if ok.any():
        print(f"  L_ciclo (caratteri)           : min {int(st.L_ciclo_char[ok].min())}, "
              f"mediana {int(st.L_ciclo_char[ok].median())}, "
              f"max {int(st.L_ciclo_char[ok].max())}")

    print('\nDistribuzione dello stallo per temperatura:')
    t = DF.groupby('temperatura').agg(n=('stallo', 'size'), in_stallo=('stallo', 'sum'))
    t['quota'] = (t.in_stallo / t.n).round(2)
    t['L_mediana'] = st.groupby('temperatura').L_ciclo.median()
    t['H_mediana'] = st.groupby('temperatura').H_parola_norm.median().round(3)
    print(t.to_string())

    if sps is not None and len(st) > 5:
        print('\nCorrelazioni di rango sui documenti in stallo:')
        for col, nome in (('H_parola_norm', 'entropia di parola'),
                          ('R_gzip', 'R_gzip'),
                          ('V_ciclo', 'tipi distinti nel ciclo')):
            r, p, n = spearman(st.L_ciclo, st[col])
            rp, pp, gruppi = spearman_a_T_fissa(st, col)
            print(f"  L_ciclo vs {nome:24s} rho = {r:+.3f} (p = {p:.1e})   "
                  f"a T fissa: rho = {rp:+.3f} (p = {pp:.1e})")
            print('       per temperatura: '
                  + '  '.join(f'T={T:.1f}:{r_:+.2f}(n={n_})' for T, r_, n_ in gruppi))

        # quanto della dispersione orizzontale resta dopo aver tolto log L
        x = np.log10(st.L_ciclo.astype(float))
        y = st.H_parola_norm.astype(float)
        b = np.polyfit(x, y, 1)
        res = y - np.polyval(b, x)
        print(f"\n  Retta H = {b[0]:+.3f} log10(L_ciclo) {b[1]:+.3f}: "
              f"R^2 = {1 - res.var()/y.var():.2f}")
        print(f"  dispersione di H: sd = {y.std():.3f} -> {res.std():.3f} "
              f"a L_ciclo fissato")
        # controllo sull'altro asse: R_gzip a L_ciclo fissato resta larghissimo
        alti = st[st.L_ciclo >= 10]
        r, p, n = spearman(alti.L_ciclo, alti.R_gzip)
        print(f"  R_gzip vs L_ciclo per L >= 10 parole: rho = {r:+.3f}, "
              f"p = {p:.2f}, n = {n}")
        for lo, hi in ((8, 16), (16, 32), (32, 64)):
            g = st[(st.L_ciclo >= lo) & (st.L_ciclo < hi)]
            if len(g) >= 5:
                print(f"    L in [{lo:3d},{hi:3d}): R_gzip da {g.R_gzip.min():.4f} "
                      f"a {g.R_gzip.max():.4f} ({g.R_gzip.max()/g.R_gzip.min():.0f}x), "
                      f"H da {g.H_parola_norm.min():.2f} a {g.H_parola_norm.max():.2f}")

    bassi = DF[DF.R_gzip < 0.02]
    bs = bassi[bassi.stallo]
    print(f"\nLa nuvola R_gzip < 0.02 citata nel capitolo: {len(bassi)} documenti, "
          f"{len(bs)} in stallo")
    print(f"  entropia   da {bassi.H_parola_norm.min():.2f} a {bassi.H_parola_norm.max():.2f}")
    if len(bs):
        print(f"  L_ciclo    da {int(bs.L_ciclo.min())} a {int(bs.L_ciclo.max())} parole")
        if sps is not None and len(bs) > 5:
            r, p, _ = spearman(bs.L_ciclo, bs.H_parola_norm)
            print(f"  dentro la sola nuvola: rho(L_ciclo, H) = {r:+.3f}, p = {p:.2e}")
            r, p, _ = spearman(bs.L_ciclo, bs.R_gzip)
            print(f"                         rho(L_ciclo, R) = {r:+.3f}, p = {p:.2e}")

    print('\nEstremi, con il ciclo effettivamente ripetuto:')
    for eti, r in (('ciclo piu corto', st.loc[st.L_ciclo.idxmin()]),
                   ('ciclo piu lungo', st.loc[st.L_ciclo.idxmax()])):
        print(f"\n  {eti}: {r.modello}  T={r.temperatura}  "
              f"L={int(r.L_ciclo)} parole ({int(r.L_ciclo_char)} car.)  V={int(r.V_ciclo)} tipi  "
              f"H={r.H_parola_norm:.3f}  R_gzip={r.R_gzip:.4f}")
        print(f'    "{str(r.estratto)[:220]}"')

    rif_path = RIS / 'metriche_per_documento.csv'
    if not rif_path.exists():
        return
    rif = pd.read_csv(rif_path)
    j = DF.merge(rif[['modello', 'temperatura', 'sample_id', 'H_parola_norm', 'R_gzip']],
                 on=['modello', 'temperatura', 'sample_id'], suffixes=('', '_rif'))
    if len(j):
        dh = (j.H_parola_norm - j.H_parola_norm_rif).abs().max()
        dr = (j.R_gzip - j.R_gzip_rif).abs().max()
        print(f"\nControllo contro metriche_per_documento.csv su {len(j)} documenti: "
              f"scarto massimo H = {dh:.2e}, R_gzip = {dr:.2e}")


def main():
    if CSV.exists() and '--ricalcola' not in sys.argv:
        DF = pd.read_csv(CSV)
        print(f"Misure lette da {CSV.name} ({len(DF)} documenti). "
              f"Usa --ricalcola per rifarle.")
    else:
        print('Caricamento dei dataset:')
        righe = carica_documenti()
        print(f"Documenti validi: {len(righe)}\nMisura del ciclo:")
        DF = misura_tutto(righe)
        DF.to_csv(CSV, index=False)
        print(f"-> {CSV}")

    UM = baseline_umana()
    print(f"\nBaseline umana: H_parola = {UM['H_parola_norm']:.3f}, "
          f"R_gzip = {UM['R_gzip']:.3f}, in stallo = {UM['stallo']}")

    figura(DF, UM)
    rapporto(DF, UM)


if __name__ == '__main__':
    main()
