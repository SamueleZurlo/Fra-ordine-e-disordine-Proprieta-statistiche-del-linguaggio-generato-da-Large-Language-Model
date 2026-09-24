"""
Rifa' le figure della DFA con detrending di ordine 2.

PERCHE'
-------
Nella versione di ordine 1 il detrending rimuove, dentro ogni finestra, soltanto
una retta. Alle basse temperature i documenti non sono stazionari: molti aprono
con prosa varia e chiudono in un ciclo di poche parole, e la composizione dei
caratteri cambia di conseguenza lungo il testo. Un gradino di quel tipo non e'
una retta, sopravvive al detrending lineare e viene letto come fluttuazione
crescente, cioe' gonfia l'esponente. Fra i 48 documenti periodici su oltre il
95% della loro lunghezza la correlazione fra la deriva di composizione e
l'esponente di ordine 1 vale rho = +0.93.

Il detrending quadratico rimuove anche la curvatura e quindi buona parte di quel
contributo. La domanda a cui la figura risponde e' se i punti sotto la pendenza
1/2 -- la saturazione dovuta alla periodicita' -- sopravvivono al cambio di
ordine. Devono sopravvivere: la saturazione non e' un andamento da detrendizzare,
e' l'assenza di diffusione.

COSA PRODUCE
------------
    cap4_dfa_<modello>_o2.png/pdf   una figura per modello, stesso impianto
                                     della versione di ordine 1
    cap4_dfa_ordini.png/pdf          confronto diretto fra i due ordini
    dfa_ordine2.csv                  esponenti per documento

USO
---
    py figure_cap4_dfa_o2.py                # in bozze/anteprima
    py figure_cap4_dfa_o2.py --applica      # in Tesi/figure
    py figure_cap4_dfa_o2.py --ricalcola    # ignora la cache
"""
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RADICE = Path(__file__).resolve().parents[2]
RIS = RADICE / 'Analisi Sweep di temperature' / 'Risultati Sweep'
CACHE = RADICE / 'Analisi Sweep di temperature' / 'cache'
OUT = (RADICE / 'Tesi' / 'figure') if '--applica' in sys.argv \
    else (RADICE / 'Tesi' / 'bozze' / 'anteprima')
OUT.mkdir(parents=True, exist_ok=True)
CSV = RIS / 'dfa_ordine2.csv'

TOKENIZER = 'cl100k_base'
N_TOK, MIN_TOK = 20000, 2000
DFA_LI, DFA_LF, DFA_N = 100, 10000, 12
N_SHUFFLE = 2
SEED = 20260817

MODELLI = {
    'Qwen2.5-1.5B':    'sweep_qwen2.5-1.5b.jsonl',
    'Qwen2.5-3B':      'sweep_qwen2.5-3b.jsonl',
    'Qwen2.5-14B':     'sweep_qwen2.5-14b.jsonl',
    'Qwen3.5-2B-Base': 'sweep_qwen3.5-2b-base.jsonl',
    'Qwen3.5-4B-Base': 'sweep_qwen3.5-4b-base.jsonl',
}
ESCLUDI = ('_short', '_failed', '.bak')

# palette del notebook (cella 3), cosi' le figure restano confrontabili
COLORI_T = ['#e6194B', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4',
            '#f032e6', '#469990', '#9A6324', '#808000', '#000075', '#808080']
MARKER_T = ['o', 's', '^', 'v', 'D', 'P', 'X', '*', '<', '>', 'h', 'p']
COLORI_MODELLO = {'Qwen2.5-1.5B': '#1b9e77', 'Qwen2.5-3B': '#d95f02',
                  'Qwen2.5-14B': '#7570b3', 'Qwen3.5-2B-Base': '#e7298a',
                  'Qwen3.5-4B-Base': '#66a61e'}
MARKER_MODELLO = {'Qwen2.5-1.5B': 'o', 'Qwen2.5-3B': 's', 'Qwen2.5-14B': '^',
                  'Qwen3.5-2B-Base': 'D', 'Qwen3.5-4B-Base': 'v'}
COLORE_UMANO = '#000000'

plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.25,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False,
})


# ---------------------------------------------------------------------------
# DFA: identica alla cella 14 del notebook, con l'ordine come parametro
# ---------------------------------------------------------------------------
def serie_rango_caratteri(testo):
    c = collections.Counter(testo)
    rank = {ch: i + 1 for i, (ch, _) in enumerate(c.most_common())}
    return np.fromiter((rank[ch] for ch in testo), dtype=np.float64, count=len(testo))


def _dfa_F(serie, L, ordine=1):
    x = np.cumsum(serie - serie.mean())
    nseg = len(x) // L
    if nseg < 1:
        return np.nan
    seg = x[:nseg * L].reshape(nseg, L)
    t = np.linspace(-1, 1, L)
    A = np.vander(t, N=ordine + 1, increasing=True)
    coef, *_ = np.linalg.lstsq(A, seg.T, rcond=None)
    return float(np.sqrt(np.mean((seg.T - A @ coef) ** 2)))


def dfa(serie, ordine=1):
    dl = (np.log2(DFA_LF) - np.log2(DFA_LI)) / (DFA_N - 1)
    Ls = np.unique(np.round(DFA_LI * 2 ** (dl * np.arange(DFA_N)))).astype(int)
    Ls = Ls[Ls <= max(len(serie) // 4, DFA_LI)]
    F = np.array([_dfa_F(serie, int(L), ordine) for L in Ls])
    m = np.isfinite(F) & (F > 0)
    return Ls[m], F[m]


def dfa_alpha(serie, ordine=1):
    L, F = dfa(serie, ordine)
    if len(L) < 3:
        return np.nan
    return float(np.polyfit(np.log(L), np.log(F), 1)[0])


def dfa_alpha_mescolato(serie, rng, ordine=1, n=N_SHUFFLE):
    out = []
    for _ in range(n):
        s = np.array(serie, copy=True)
        rng.shuffle(s)
        out.append(dfa_alpha(s, ordine))
    return float(np.nanmean(out))


# ---------------------------------------------------------------------------
# Dati: stessa pipeline del notebook, compresa la ricostruzione del testo umano
# ---------------------------------------------------------------------------
def carica():
    import tiktoken
    enc = tiktoken.get_encoding(TOKENIZER)
    docs = []
    for etichetta, pattern in MODELLI.items():
        cand = [p for p in sorted(RADICE.glob(pattern))
                if not any(x in p.name for x in ESCLUDI)]
        if not cand:
            continue
        f = max(cand, key=lambda p: p.stat().st_mtime)
        with open(f, encoding='utf-8') as fh:
            for i, linea in enumerate(fh):
                linea = linea.strip()
                if not linea:
                    continue
                r = json.loads(linea)
                ids = enc.encode(r['generated_text'], disallowed_special=())[:N_TOK]
                if len(ids) < MIN_TOK:
                    continue
                docs.append(dict(modello=etichetta, temperatura=r['temperature'],
                                 sample_id=r.get('sample_id', i),
                                 testo=enc.decode(ids),
                                 sha=r.get('prompt_sha256_16'),
                                 lung=r.get('prompt_length_chars')))
    return docs, enc


def testo_umano(docs, enc, max_offset=400000):
    """La continuazione umana del romanzo, ricostruita come nella cella 9 del
    notebook: si localizza il prompt nel testo di Guerra e pace tramite il suo
    hash, e si prende cio' che viene dopo."""
    p = CACHE / 'war_and_peace.txt'
    if not p.exists():
        p = RADICE / 'corpora_cache' / 'f6c229e9b581.txt'
    romanzo = p.read_bytes().decode('utf-8', errors='replace')
    voluti = {(d['sha'], d['lung']) for d in docs if d['sha'] and d['lung']}
    testa = romanzo[:max_offset + max(L for _, L in voluti) + 10]
    byte_di = np.zeros(len(testa) + 1, dtype=np.int64)
    byte_di[1:] = np.cumsum([len(ch.encode('utf-8')) for ch in testa])
    grezzo = testa.encode('utf-8')
    fine = 0
    for h, L in voluti:
        for off in range(min(max_offset, len(testa) - L)):
            if hashlib.sha256(grezzo[byte_di[off]:byte_di[off + L]]).hexdigest()[:16] == h:
                fine = max(fine, off + L)
                break
    if not fine:
        raise SystemExit('Prompt non localizzato nel romanzo.')
    t = romanzo[fine:]
    for marker in ('*** END OF', '***END OF', 'End of the Project Gutenberg'):
        j = t.find(marker)
        if j != -1:
            t = t[:j]
            break
    return enc.decode(enc.encode(t, disallowed_special=())[:N_TOK])


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def stile_T(T, tutte):
    i = list(np.round(tutte, 3)).index(round(float(T), 3))
    return dict(color=COLORI_T[i % len(COLORI_T)], marker=MARKER_T[i % len(MARKER_T)])


def figura_modello(m, esempi, umano_txt, D, UM, temperature, ordine=2):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5))

    ax = axes[0]
    for T in sorted(esempi):
        L, F = dfa(serie_rango_caratteri(esempi[T]), ordine)
        if len(L):
            ax.loglog(L, F / F[0], lw=1.3, ms=3.5, markevery=max(1, len(L) // 12),
                      label=f'$T$={T:g}', **stile_T(T, temperature))
    Lu, Fu = dfa(serie_rango_caratteri(umano_txt), ordine)
    if len(Lu):
        ax.loglog(Lu, Fu / Fu[0], color=COLORE_UMANO, lw=2.4, ls='--',
                  label='umano', zorder=10)
    sref = np.array([DFA_LI, DFA_LI * 30], dtype=float)
    ax.loglog(sref, (sref / sref[0]) ** 0.5, 'k:', lw=1.5,
              label=r'$\alpha_{\mathrm{DFA}}=0.5$')
    ax.set_xlabel('scala $L$ [caratteri]')
    ax.set_ylabel('$F(L)$ normalizzata')
    ax.set_title(f'DFA sui caratteri, detrending di ordine {ordine} — {m}',
                 fontsize=10)
    ax.legend(fontsize=6.5, ncol=2)

    ax = axes[1]
    d = D[D.modello == m]
    g = d.groupby('temperatura')
    mu, sd, n = g['alpha_o2'].mean(), g['alpha_o2'].std(), g['alpha_o2'].count()
    ax.errorbar(mu.index, mu.values, yerr=sd.where(n > 1, 0).fillna(0).values,
                capsize=3, ms=5, lw=1.6, label='originale',
                color=COLORI_MODELLO[m], marker=MARKER_MODELLO[m])
    gm = g['alpha_o2_mescolato'].mean()
    ax.plot(gm.index, gm.values, 's--', ms=4, color='grey',
            label='mescolato (null model)')
    ax.axhline(0.5, color='k', ls=':', lw=1.2)
    ax.axhline(UM['alpha_o2'], color=COLORE_UMANO, ls='--', lw=1.9, label='umano')
    ax.set_xlabel('temperatura $T$')
    ax.set_ylabel(r'$\alpha_{\mathrm{DFA}}$  (ordine 2)')
    ax.set_title(f'Esponente DFA di ordine {ordine} — {m}', fontsize=10)
    ax.legend(fontsize=7.5)

    fig.tight_layout()
    nome = f"cap4_dfa_{m.replace('.', '').replace('-', '_')}_o2"
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{nome}.{ext}')
    plt.close(fig)
    print(f"-> {OUT / (nome + '.png')}")


def figura_confronto(D, UM):
    """I due ordini a confronto: dove cambia la lettura e dove no."""
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.0))

    ax = axes[0]
    for col, lab, c, ls in (('alpha_o1', 'ordine 1 (lineare)', '#2F4E7E', '-'),
                            ('alpha_o2', 'ordine 2 (quadratico)', '#C1502E', '--')):
        g = D.groupby('temperatura')[col]
        mu, sd = g.mean(), g.std().fillna(0)
        ax.plot(mu.index, mu.values, ls=ls, marker='o', ms=5, lw=1.7, color=c,
                label=lab)
        ax.fill_between(mu.index, mu - sd, mu + sd, color=c, alpha=.12)
    ax.axhline(UM['alpha_o1'], color='#2F4E7E', ls=':', lw=1.4)
    ax.axhline(UM['alpha_o2'], color='#C1502E', ls=':', lw=1.4)
    ax.axhline(0.5, color='grey', ls=':', lw=1)
    ax.set_xlabel('temperatura $T$')
    ax.set_ylabel(r'$\alpha_{\mathrm{DFA}}$')
    ax.set_title('A. Media e dispersione ai due ordini\n'
                 '(punteggiate: baseline umana)', fontsize=10)
    ax.legend(fontsize=7.5)

    ax = axes[1]
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(D.temperatura.min(), D.temperatura.max())
    for m, mk in MARKER_MODELLO.items():
        d = D[D.modello == m]
        ax.scatter(d.alpha_o1, d.alpha_o2, s=32, marker=mk, alpha=.85,
                   c=cmap(norm(d.temperatura)), edgecolors='none')
    lim = [min(D.alpha_o1.min(), D.alpha_o2.min()) - .05,
           max(D.alpha_o1.max(), D.alpha_o2.max()) + .05]
    ax.plot(lim, lim, 'k--', lw=1.2)
    ax.axhline(0.5, color='grey', ls=':', lw=1)
    ax.axvline(0.5, color='grey', ls=':', lw=1)
    ax.scatter([UM['alpha_o1']], [UM['alpha_o2']], s=240, marker='*',
               color=COLORE_UMANO, zorder=6)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r'$\alpha_{\mathrm{DFA}}$, ordine 1')
    ax.set_ylabel(r'$\alpha_{\mathrm{DFA}}$, ordine 2')
    ax.set_title('B. Documento per documento\n'
                 f'sotto $1/2$: {int((D.alpha_o1 < .5).sum())} a ordine 1, '
                 f'{int((D.alpha_o2 < .5).sum())} a ordine 2', fontsize=10)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cax = ax.inset_axes([0.06, 0.92, 0.34, 0.030])
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_label('temperatura $T$', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax = axes[2]
    g = D.groupby('temperatura')
    for col, lab, c, ls in (('alpha_o1', 'ordine 1 (lineare)', '#2F4E7E', '-'),
                            ('alpha_o2', 'ordine 2 (quadratico)', '#C1502E', '--')):
        q = g[col].apply(lambda s: (s < 0.5).mean())
        ax.plot(q.index, q.values, ls=ls, marker='o', ms=5, lw=1.7, color=c,
                label=lab)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel('temperatura $T$')
    ax.set_ylabel(r'quota di documenti con $\alpha_{\mathrm{DFA}} < 1/2$')
    ax.set_title('C. I punti sotto la pendenza $1/2$\n'
                 'non spariscono cambiando ordine: aumentano', fontsize=10)
    ax.legend(fontsize=7.5)

    for a in axes:
        a.set_box_aspect(1)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'cap4_dfa_ordini.{ext}')
    plt.close(fig)
    print(f"-> {OUT / 'cap4_dfa_ordini.png'}")


def deriva_composizione(testo, w=2000):
    """Quanto cambia la composizione dei caratteri lungo il documento: scarto
    tipo della media mobile del rango, in unita' del rango medio."""
    s = serie_rango_caratteri(testo)
    if len(s) <= w:
        return np.nan
    mv = np.convolve(s, np.ones(w) / w, mode='valid')
    return float(mv.std() / s.mean())


# ---------------------------------------------------------------------------
def main():
    import tiktoken
    docs, enc = carica()
    print(f"Documenti validi: {len(docs)}")
    umano_txt = testo_umano(docs, enc)

    # --- validazione: la pipeline deve riprodurre i valori gia' salvati -----
    rif = pd.read_csv(RIS / 'baseline_umana.csv', index_col=0).iloc[:, 0]
    a1_um = dfa_alpha(serie_rango_caratteri(umano_txt), 1)
    atteso = float(rif['alpha_DFA'])
    print(f"Baseline umana ricostruita: alpha ordine 1 = {a1_um:.6f}, "
          f"atteso {atteso:.6f}, scarto {abs(a1_um - atteso):.2e}")
    if abs(a1_um - atteso) > 1e-6:
        raise SystemExit('[!] Il testo umano ricostruito non coincide con quello '
                         'del notebook: la figura non sarebbe confrontabile.')

    if CSV.exists() and '--ricalcola' not in sys.argv:
        D = pd.read_csv(CSV)
        print(f"Esponenti letti da {CSV.name} ({len(D)} documenti).")
    else:
        righe, t0 = [], time.time()
        for i, d in enumerate(docs):
            s = serie_rango_caratteri(d['testo'])
            rng = np.random.default_rng(SEED + i)
            righe.append(dict(modello=d['modello'], temperatura=d['temperatura'],
                              sample_id=d['sample_id'],
                              alpha_o1=dfa_alpha(s, 1),
                              alpha_o2=dfa_alpha(s, 2),
                              alpha_o2_mescolato=dfa_alpha_mescolato(s, rng, 2),
                              deriva=deriva_composizione(d['testo'])))
            print(f"  {i+1}/{len(docs)}  ({time.time()-t0:.0f}s)", end='\r')
        print()
        D = pd.DataFrame(righe)
        D.to_csv(CSV, index=False)
        print(f"-> {CSV}")

    # controllo incrociato con le stime gia' presenti nel CSV del notebook
    rifd = pd.read_csv(RIS / 'metriche_per_documento.csv')
    j = D.merge(rifd[['modello', 'temperatura', 'sample_id', 'alpha_DFA',
                      'alpha_DFA_o2']],
                on=['modello', 'temperatura', 'sample_id'])
    print(f"Controllo su {len(j)} documenti: scarto massimo ordine 1 = "
          f"{(j.alpha_o1 - j.alpha_DFA).abs().max():.2e}, ordine 2 = "
          f"{(j.alpha_o2 - j.alpha_DFA_o2).abs().max():.2e}")

    s_um = serie_rango_caratteri(umano_txt)
    UM = dict(alpha_o1=a1_um, alpha_o2=dfa_alpha(s_um, 2))
    print(f"Umano: ordine 1 = {UM['alpha_o1']:.3f}, ordine 2 = {UM['alpha_o2']:.3f}")

    temperature = sorted({d['temperatura'] for d in docs})
    for m in MODELLI:
        esempi = {}
        for d in docs:
            if d['modello'] == m and d['temperatura'] not in esempi:
                esempi[d['temperatura']] = d['testo']
        if esempi:
            figura_modello(m, esempi, umano_txt, D, UM, temperature)
    figura_confronto(D, UM)
    rapporto(D, UM, docs)


def rapporto(D, UM, docs):
    print('\n' + '=' * 72)
    print(f"{'':22} {'ordine 1':>10} {'ordine 2':>10}")
    for eti, f in (('documenti sotto 1/2', lambda c: (D[c] < 0.5).sum()),
                   ('documenti sotto 0.3', lambda c: (D[c] < 0.3).sum()),
                   ('documenti sopra 1.0', lambda c: (D[c] > 1.0).sum()),
                   ('mediana', lambda c: round(D[c].median(), 3)),
                   ('minimo', lambda c: round(D[c].min(), 3)),
                   ('massimo', lambda c: round(D[c].max(), 3))):
        print(f"  {eti:20} {f('alpha_o1'):>10} {f('alpha_o2'):>10}")

    print('\nDispersione fra documenti, per temperatura:')
    t = D.groupby('temperatura')[['alpha_o1', 'alpha_o2']].agg(['mean', 'std']).round(3)
    print(t.to_string())

    try:
        from scipy import stats as sps
    except ImportError:
        return
    print('\nLegame con la non-stazionarieta\' (deriva di composizione):')
    for c in ('alpha_o1', 'alpha_o2'):
        r, p = sps.spearmanr(D.deriva, D[c])
        print(f"  rho(deriva, {c}) = {r:+.3f}   p = {p:.1e}")

    ciclo = RIS / 'ciclo_stallo.csv'
    if ciclo.exists():
        C = pd.read_csv(ciclo)
        K = D.merge(C[['modello', 'temperatura', 'sample_id', 'stallo', 'L_ciclo']],
                    on=['modello', 'temperatura', 'sample_id'])
        st = K[K.stallo]
        print(f"\nDocumenti in stallo (n = {len(st)}):")
        for c in ('alpha_o1', 'alpha_o2'):
            print(f"  {c}: mediana {st[c].median():.3f}, "
                  f"quota sotto 1/2 = {(st[c] < 0.5).mean():.2f}")
        pu = K[~K.stallo]
        print(f"Documenti non in stallo (n = {len(pu)}):")
        for c in ('alpha_o1', 'alpha_o2'):
            print(f"  {c}: mediana {pu[c].median():.3f}, "
                  f"quota sotto 1/2 = {(pu[c] < 0.5).mean():.2f}")


if __name__ == '__main__':
    main()
