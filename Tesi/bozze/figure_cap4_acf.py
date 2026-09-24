"""
Rifà la figura dell'autocorrelazione del Capitolo 4.

La versione precedente sovrapponeva dodici curve nello stesso riquadro ed era
illeggibile. Qui ogni condizione ha il proprio pannello: la riga superiore
riporta l'ACF, quella inferiore il modulo della sua trasformata di Fourier in
scala logaritmica, con la soglia di periodicità disegnata esplicitamente.

Le funzioni sono copiate senza modifiche dal notebook
'Analisi Sweep di temperature/analisi_sweep_temperature.ipynb' (celle 17, 42,
43) così che le curve coincidano con quelle già usate per le tabelle.

Uso:
    py figure_cap4_acf.py            # solo Qwen2.5-3B, in bozze/anteprima
    py figure_cap4_acf.py --tutti    # tutti e cinque i modelli
    py figure_cap4_acf.py --applica  # scrive in Tesi/figure
"""
import gzip, json, re, sys, time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tiktoken

# ---------------------------------------------------------------- percorsi
RADICE = Path(__file__).resolve().parents[2]          # 'Dataset totale'
SWEEP = RADICE / 'Analisi Sweep di temperature'
CACHE = SWEEP / 'cache'
APPLICA = '--applica' in sys.argv
OUT = (RADICE / 'Tesi' / 'figure') if APPLICA else (RADICE / 'Tesi' / 'bozze' / 'anteprima')
OUT.mkdir(parents=True, exist_ok=True)

MODELLI = {
    'Qwen2.5-1.5B':    'sweep_qwen2.5-1.5b.jsonl',
    'Qwen2.5-3B':      'sweep_qwen2.5-3b.jsonl',
    'Qwen2.5-14B':     'sweep_qwen2.5-14b.jsonl',
    'Qwen3.5-2B-Base': 'sweep_qwen3.5-2b-base.jsonl',
    'Qwen3.5-4B-Base': 'sweep_qwen3.5-4b-base.jsonl',
}
ESCLUDI = ('_short', '_failed', '.bak')

N_TOK = 20000
SEED = 20260817
SHORT_LAGS = np.arange(1, 101)
SOGLIA_FFT = 3.0

# Temperature mostrate: due dentro la fase periodica, l'ultima periodica,
# la prima oltre la transizione, e una temperatura alta.
T_MOSTRATE = [0.5, 0.8, 0.9, 1.0, 1.3]

plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.25,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False,
})

COLORI_T = ['#e6194B', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4',
            '#f032e6', '#469990', '#9A6324', '#808000', '#000075', '#808080']
TUTTE_T = [round(0.4 + i * 0.1, 1) for i in range(12)]


def colore_T(T):
    return COLORI_T[TUTTE_T.index(round(float(T), 1)) % len(COLORI_T)]


# ------------------------------------------------ funzioni prese dal notebook
def parole(testo):
    return re.findall(r'\w+', re.sub(r'\d', ' ', testo.lower()))


def traiettoria(parole_doc, emb, dim):
    """[F] sec.2.3: ogni parola diventa il suo vettore GloVe; il sistema viene
    centrato sottraendo la media sull'intero testo."""
    M = np.zeros((len(parole_doc), dim), dtype=np.float32)
    noti = np.zeros(len(parole_doc), dtype=bool)
    for i, w in enumerate(parole_doc):
        v = emb.get(w)
        if v is not None:
            M[i] = v
            noti[i] = True
    if noti.any():
        M[noti] -= M[noti].mean(0)
        M[~noti] = 0.0
    return M, noti


def acf_coseno(M, lags):
    """[F] eq.(6): media della similarità coseno fra i vettori a distanza tau."""
    norme = np.linalg.norm(M, axis=1)
    out = np.full(len(lags), np.nan)
    N = len(M)
    for j, tau in enumerate(lags):
        tau = int(tau)
        if tau < 1 or tau >= N:
            continue
        den = norme[:-tau] * norme[tau:]
        num = np.einsum('ij,ij->i', M[:-tau], M[tau:])
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
        out[j] = float(c.mean())
    return out


def parametro_periodicita(acf):
    """[F] sec.2.5: massimo modulo della DFT dell'ACF, escluso il primo
    coefficiente."""
    a = np.nan_to_num(np.asarray(acf, float) - np.nanmean(acf))
    if len(a) < 4:
        return np.nan
    F = np.abs(np.fft.rfft(a))
    return float(F[1:].max()) if len(F) > 1 else np.nan


def spettro(acf):
    a = np.nan_to_num(np.asarray(acf, float) - np.nanmean(acf))
    return np.abs(np.fft.rfft(a))


# ---------------------------------------------------------------- caricamento
def carica_glove():
    f = CACHE / 'glove-wiki-gigaword-100.gz'
    if not f.exists():
        sys.exit(f"Manca {f}: eseguire prima il notebook dello sweep.")
    t0 = time.time()
    emb = {}
    with gzip.open(f, 'rt', encoding='utf-8') as fh:
        fh.readline()
        for line in fh:
            p = line.rstrip().split(' ')
            emb[p[0]] = np.asarray(p[1:], dtype=np.float32)
    dim = len(next(iter(emb.values())))
    print(f"GloVe: {len(emb):,} parole, dimensione {dim}, "
          f"caricato in {time.time()-t0:.0f}s")
    return emb, dim


def carica_documenti(modello):
    cand = [p for p in sorted(RADICE.glob(MODELLI[modello]))
            if not any(x in p.name for x in ESCLUDI)]
    if not cand:
        sys.exit(f"Nessun dataset per {modello}")
    f = max(cand, key=lambda p: p.stat().st_mtime)
    docs = []
    with open(f, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def testo_umano():
    p = CACHE / 'war_and_peace.txt'
    if not p.exists():
        sys.exit(f"Manca {p}")
    return p.read_text(encoding='utf-8')


# ---------------------------------------------------------------- la figura
def curve_modello(modello, docs, enc, emb, dim):
    """Per ogni temperatura mostrata sceglie il campione la cui periodicità è
    la mediana di quella temperatura: un esemplare tipico, non il più vistoso."""
    scelte = {}
    for T in T_MOSTRATE:
        cand = [r for r in docs if abs(r['temperature'] - T) < 1e-9]
        if not cand:
            continue
        risultati = []
        for r in cand:
            ids = enc.encode(r['generated_text'], disallowed_special=())[:N_TOK]
            par = parole(enc.decode(ids))
            M, noti = traiettoria(par, emb, dim)
            if len(M) < 500 or not noti.any():
                continue
            ac = acf_coseno(M, SHORT_LAGS)
            rng = np.random.default_rng(SEED + int(r.get('sample_id', 0)))
            ac_null = acf_coseno(M[rng.permutation(len(M))], SHORT_LAGS)
            risultati.append(dict(
                sample=r.get('sample_id'), acf=ac,
                fft=spettro(ac), max_fft=parametro_periodicita(ac),
                rumore=float(np.nanmax(np.abs(ac_null))),
                copertura=float(noti.mean())))
        if not risultati:
            continue
        risultati.sort(key=lambda d: d['max_fft'])
        scelte[T] = risultati[len(risultati) // 2]
        s = scelte[T]
        print(f"  T={T:.1f}  campione {s['sample']}  "
              f"max|FFT|={s['max_fft']:.2f}  copertura={s['copertura']:.3f}  "
              f"(mediana su {len(risultati)} campioni)")
    return scelte


def curva_umana(txt, enc, emb, dim):
    ids = enc.encode(txt, disallowed_special=())[:N_TOK]
    par = parole(enc.decode(ids))
    M, noti = traiettoria(par, emb, dim)
    ac = acf_coseno(M, SHORT_LAGS)
    rng = np.random.default_rng(SEED)
    ac_null = acf_coseno(M[rng.permutation(len(M))], SHORT_LAGS)
    d = dict(acf=ac, fft=spettro(ac), max_fft=parametro_periodicita(ac),
             rumore=float(np.nanmax(np.abs(ac_null))), copertura=float(noti.mean()))
    print(f"  umano      max|FFT|={d['max_fft']:.2f}  copertura={d['copertura']:.3f}")
    return d


def disegna(modello, scelte, umano, nomefile):
    """Griglia 4x3: ogni condizione occupa una colonna con l'ACF sopra e il suo
    spettro sotto. La scala verticale dell'ACF è propria di ciascun pannello,
    perché l'ampiezza cambia di tre ordini di grandezza; il valore di picco è
    scritto dentro il riquadro. Lo spettro usa invece una scala logaritmica
    comune, così i sei casi sono direttamente confrontabili."""
    celle = [(f'$T = {T:.1f}$', colore_T(T), scelte[T])
             for T in T_MOSTRATE if T in scelte]
    celle.append(('testo umano', '#000000', umano))

    ncol = 3
    nblocchi = int(np.ceil(len(celle) / ncol))
    fig, axes = plt.subplots(2 * nblocchi, ncol,
                             figsize=(3.15 * ncol, 2.55 * 2 * nblocchi))
    axes = np.atleast_2d(axes)

    for k, (etichetta, colore, d) in enumerate(celle):
        blocco, j = divmod(k, ncol)
        a, b = axes[2 * blocco, j], axes[2 * blocco + 1, j]

        # --- autocorrelazione -------------------------------------------
        amp = float(np.nanmax(np.abs(d['acf'])))
        basso = max(abs(float(np.nanmin(d['acf']))), d['rumore'])
        alto = max(float(np.nanmax(d['acf'])), d['rumore'])
        a.axhspan(-d['rumore'], d['rumore'], color='#c8c8c8', alpha=.55,
                  lw=0, zorder=0)
        a.axhline(0, color='k', lw=.7, zorder=1)
        a.plot(SHORT_LAGS, d['acf'], color=colore, lw=1.0, zorder=2)
        a.set_ylim(-1.15 * basso, 1.55 * alto)
        a.set_title(etichetta, fontsize=10.5)
        a.set_xlabel(r'lag $\tau$ [parole]')
        a.set_ylabel(r'$C(\tau)$')
        a.text(.97, .95, rf"$\max|C| = {amp:.3f}$", transform=a.transAxes,
               ha='right', va='top', fontsize=8.5)

        # --- spettro ------------------------------------------------------
        F = d['fft']
        b.axhline(SOGLIA_FFT, color='#555555', ls='--', lw=1.0, zorder=1)
        if j == 0:
            b.text(48, SOGLIA_FFT * 1.35, 'soglia di periodicità',
                   ha='right', va='bottom', fontsize=8, color='#555555')
        b.plot(np.arange(1, len(F)), F[1:], color=colore, lw=1.0, zorder=2)
        b.set_yscale('log')
        b.set_ylim(3e-3, 1e2)
        b.set_xlim(0, 50)
        b.set_xlabel('frequenza')
        b.set_ylabel(r'$|\mathrm{FFT}|$')
        b.text(.97, .93, rf"$\max|\mathrm{{FFT}}| = {d['max_fft']:.2f}$",
               transform=b.transAxes, ha='right', va='top', fontsize=8.5)
        b.text(.03, .93, f"copertura {d['copertura']:.2f}",
               transform=b.transAxes, ha='left', va='top',
               fontsize=8, color='#555555')

    # riquadri avanzati, se le condizioni non riempiono la griglia
    for k in range(len(celle), nblocchi * ncol):
        blocco, j = divmod(k, ncol)
        axes[2 * blocco, j].axis('off')
        axes[2 * blocco + 1, j].axis('off')


    fig.suptitle(f'Autocorrelazione della traiettoria di embedding — {modello}',
                 fontsize=11.5, y=1.002)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{nomefile}.{ext}')
    plt.close(fig)
    print(f"  -> {OUT / (nomefile + '.png')}")


def main():
    enc = tiktoken.get_encoding('cl100k_base')
    emb, dim = carica_glove()
    umano = None

    elenco = list(MODELLI) if '--tutti' in sys.argv else ['Qwen2.5-3B']
    for modello in elenco:
        print(f"\n{modello}")
        docs = carica_documenti(modello)
        scelte = curve_modello(modello, docs, enc, emb, dim)
        if umano is None:
            umano = curva_umana(testo_umano(), enc, emb, dim)
        nome = 'cap4_acf_' + modello.replace('.', '').replace('-', '_')
        disegna(modello, scelte, umano, nome)


if __name__ == '__main__':
    main()
