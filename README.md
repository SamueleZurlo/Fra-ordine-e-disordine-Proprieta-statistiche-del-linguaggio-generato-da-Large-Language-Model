# Fra ordine e disordine

**Proprietà statistiche del linguaggio generato da Large Language Model**

Tesi di laurea in Fisica, Alma Mater Studiorum – Università di Bologna,
A.A. 2025/2026. Relatore: Prof. Mirko Degli Esposti.

Questo repository contiene la tesi ([`Tesi/main.pdf`](Tesi/main.pdf)), i 242
documenti di testo generato su cui si basa, e il codice che ne produce i
risultati, le tabelle e le figure.

---

## In breve

Il lavoro chiede quali proprietà statistiche del linguaggio naturale un modello
linguistico riproduca davvero, e a quale livello della gerarchia linguistica
smetta di riprodurle. Procede lungo due assi indipendenti.

**La temperatura di campionamento (Capitolo 4).** Cinque modelli Qwen *base*
continuano lo stesso passo di *Guerra e pace* a dodici temperature, da 0.4 a
1.5, per 242 documenti in tutto. Otto criteri tratti da quattro lavori
indipendenti danno la stessa temperatura mediana, $T = 1.0$, per tutti e cinque
i modelli, senza dipendenza dalla dimensione. La finestra è però stretta: fuori
da essa il testo degenera per ripetizione o per incoerenza, e solo due documenti
su 242 non ricadono in alcuna modalità di degenerazione.

**La gerarchia linguistica (Capitolo 5).** Su 39 voci enciclopediche appaiate
per titolo fra Wikipedia e Grokipedia, con *Guerra e pace* come riferimento
letterario, si applica il protocollo di Altmann, Cristadoro e Degli Esposti
(*PNAS* 2012). Il testo generato conserva il meccanismo che produce le
correlazioni a lungo raggio, ma con ampiezza ridotta. Sulle lettere i corpora
sono indistinguibili; il divario si apre al livello delle parole.

---

## Contenuto

```
sweep_qwen*.jsonl                  i cinque dataset di testo generato
Analisi Sweep di temperature/
  analisi_sweep_temperature.ipynb  analisi dello sweep (Capitolo 4)
  Risultati Sweep/                 CSV, tabelle, sintesi e figure del notebook
Wiki/                              corpora enciclopedici (Capitolo 5)
  altmann_wikipedia.ipynb          protocollo di Altmann su Wikipedia
  confronto_grokipedia.ipynb       appaiamento e paternità delle voci Grokipedia
  altmann_tre_corpora.ipynb        Wikipedia, Grokipedia v0.1 e attuale, romanzo
  entropia_intervalli.ipynb        entropia degli intervalli J
  esponente_coda.ipynb             esponente di coda degli intervalli
  risultati_confronto/             titoli appaiati, paternità e revisioni
  risultati_*_v2/                  risultati usati nella tesi
  risultati_*/                     prima versione, conservata per il §5.8
  confronto_N_eff*/                robustezza rispetto alla lunghezza di analisi
analisi_altmann_burstiness.ipynb   protocollo di Altmann sui testi dello sweep
risultati_altmann/                 i suoi risultati (analisi preliminare)
generazione_token_singolo.ipynb    sweep con prompt di un solo token (ablazione)
Tesi/
  main.tex, capitolo*.tex, …       sorgenti LaTeX
  main.pdf                         la tesi compilata, 92 pagine
  figure/                          le figure della tesi
  bozze/                           script che rigenerano figure e numeri
FONTI.md                           fonti esterne, identificatori e impronte
requirements.txt                   pacchetti Python
LICENSE, LICENSE-DATA              licenze del codice e dei dati
```

---

## I dataset generati

Un file per modello, una riga per documento, in formato JSON Lines.

| file | modello | documenti |
|---|---|---|
| `sweep_qwen2.5-1.5b.jsonl` | Qwen2.5-1.5B | 57 |
| `sweep_qwen2.5-3b.jsonl` | Qwen2.5-3B | 53 |
| `sweep_qwen2.5-14b.jsonl` | Qwen2.5-14B | 12 |
| `sweep_qwen3.5-2b-base.jsonl` | Qwen3.5-2B-Base | 60 |
| `sweep_qwen3.5-4b-base.jsonl` | Qwen3.5-4B-Base | 60 |

Ogni documento conta 24 000 token nel tokenizzatore del modello che l'ha
generato. Il campione per temperatura non è uniforme: Qwen2.5-14B ha un
documento per temperatura, e Qwen2.5-3B uno solo a $T = 1.0$.

I campi principali di ogni riga:

| campo | contenuto |
|---|---|
| `generated_text` | il testo generato, senza il prompt |
| `temperature`, `sample_id`, `seed` | condizioni del campionamento |
| `top_p`, `top_k`, `repetition_penalty` | sempre 1, −1 e 1: si campiona con la sola temperatura |
| `prompt_sha256_16`, `prompt_length_chars`, `prompt_length_tokens` | impronta e lunghezza del prompt |
| `n_continuations`, `eos_positions` | riprese dopo un token di fine sequenza, e dove sono cadute |
| `model`, `tokenizer_id`, `run_id`, `generated_at` | provenienza |

Per leggerli:

```python
import pandas as pd
df = pd.read_json("sweep_qwen2.5-3b.jsonl", lines=True)
```

Le impronte SHA-256 dei cinque file sono in [`FONTI.md`](FONTI.md).

---

## Dalla tesi al codice

Le figure in `Tesi/figure/` vengono da tre fonti: i notebook, che le salvano
nelle proprie cartelle di risultati; gli script in `Tesi/bozze/`; e un piccolo
gruppo di illustrazioni del Capitolo 2 il cui codice non è stato conservato.

| figure nella tesi | prodotte da |
|---|---|
| `cap2_invarianza_J` | `Tesi/bozze/figura_invarianza_J.py` |
| le altre sette `cap2_*` | codice non conservato (vedi sotto) |
| `cap3_dimensione_finita` | `Tesi/bozze/figure_cap3.py` |
| `cap4_zipf_*`, `cap4_heaps_curve_*`, `cap4_transizioni_fase`, `cap4_diagramma_fase`, `cap4_temperatura_critica` | `analisi_sweep_temperature.ipynb`, copie di `Risultati Sweep/03_*`–`15_*` |
| `cap4_heaps_R_vs_T` | `Tesi/bozze/figure_cap4_heaps.py` |
| `cap4_dfa_*_o2` | `Tesi/bozze/figure_cap4_dfa_o2.py` |
| `cap4_compressione` | `Tesi/bozze/figure_cap4_compressione.py` |
| `cap4_acf_*` | `Tesi/bozze/figure_cap4_acf.py` |
| `cap4_ciclo_entropia` | `Tesi/bozze/analisi_ciclo_stallo.py` |
| `cap5_wiki_altmann` | `altmann_wikipedia.ipynb` → `risultati_wiki_v2/` |
| `cap5_tre_corpora`, `cap5_altmann_fig3`, `cap5_decomposizione_eq4` | `altmann_tre_corpora.ipynb` → `risultati_tre_corpora_v2/` |
| `cap5_entropia_intervalli` | `entropia_intervalli.ipynb` → `risultati_entropia_v2/` |
| `cap5_altmann_fig2` | `Tesi/bozze/figure_cap5_fig2.py` |
| `cap5_quadranti` | `Tesi/bozze/figure_cap5_quadranti_v2.py` |
| `cap5_esponente_coda` | `Tesi/bozze/rigenera_fig_coda.py` |
| `cap5_verosimiglianza_coda` | `Tesi/bozze/analisi_coda_cap5.py` |
| `cap5_effetto_stoplist` | `Tesi/bozze/figure_cap5_stoplist.py` |

`Tesi/figure/` contiene anche alcune figure di versioni precedenti del testo,
che la versione finale non include.

Gli altri script di `Tesi/bozze/` calcolano numeri citati nel testo o ne
verificano la robustezza: `baseline_quattro_romanzi.py` e
`controllo_letterario.py` ripetono le misure su altri romanzi, `confronto_neff.py`
e `neff_*.py` confrontano due lunghezze di analisi, `verifica_J.py` e
`indaga_appaiati.py` controllano le proprietà dell'entropia degli intervalli.
Ciascuno spiega nella propria intestazione che cosa misura e perché.

### Le cartelle `_v2`

Nel Capitolo 5 il livello delle parole chiave si ottiene escludendo una lista
chiusa di parole funzione. La prima versione della lista, da 133 voci, lasciava
passare connettivi come *like*, *amid*, *though*. In Grokipedia *like* finiva
fra le parole chiave di 29 voci su 39. La lista è stata quindi estesa a 215
voci e l'intera catena rieseguita.

- `Wiki/risultati_*_v2/` sono i risultati con la lista estesa, **quelli riportati
  nella tesi**. Li produce `Tesi/bozze/rigenera_cap5_stoplist.py`, che esegue i
  notebook di `Wiki/` aggiungendo l'estensione della lista senza modificarli.
- `Wiki/risultati_*/` sono i risultati con la lista originale. Restano perché il
  §5.8 misura di quanto la scelta della lista sposti i risultati. Fa eccezione
  `risultati_confronto/`, che non dipende dalla lista e ha una sola versione.

---

## Riprodurre le analisi

### Ambiente

Python 3.11 o successivo e Jupyter.

```bash
pip install -r requirements.txt
```

Il notebook di generazione richiede in più `openai` e `transformers`, e una
chiave API di DeepInfra.

### Capitolo 4: lo sweep di temperatura

Il notebook ha bisogno solo dei cinque file JSONL, inclusi. Al primo avvio
scarica *Guerra e pace* da Project Gutenberg e i vettori GloVe (circa 134 MB).

1. Aprire `Analisi Sweep di temperature/analisi_sweep_temperature.ipynb` dalla
   sua cartella ed eseguire tutte le celle, in circa 5 minuti. I risultati
   finiscono in `Risultati Sweep/`.
2. Rigenerare le figure del capitolo che non escono dal notebook con gli script
   `Tesi/bozze/figure_cap4_*.py` e `Tesi/bozze/analisi_ciclo_stallo.py`.

### Capitolo 5: i corpora enciclopedici

Questa parte richiede i corpora, che il repository non contiene: vedi
[`FONTI.md`](FONTI.md) e la sezione seguente. I notebook si aspettano le copie
locali in `Wiki/cache_wikipedia/`, `Wiki/cache_grokipedia/`,
`Wiki/risultati_confronto/corpus_v01/` e `corpora_cache/`.

1. `altmann_wikipedia.ipynb`, poi `confronto_grokipedia.ipynb`: selezione e
   appaiamento delle voci. Scaricano ciò che non trovano in cache.
2. `altmann_tre_corpora.ipynb`, `entropia_intervalli.ipynb`,
   `esponente_coda.ipynb`, in quest'ordine: producono `Wiki/risultati_*/`.
3. `python Tesi/bozze/rigenera_cap5_stoplist.py`: produce `Wiki/risultati_*_v2/`,
   in circa 10 minuti.

### Gli script di `Tesi/bozze/`

Si lanciano da qualunque cartella, perché ricavano i percorsi dalla propria
posizione. Gli script di figura scrivono in `Tesi/bozze/anteprima/`; con
l'opzione `--applica` sovrascrivono la figura in `Tesi/figure/`.

```bash
python Tesi/bozze/figure_cap3.py
python Tesi/bozze/figure_cap3.py --applica
```

### Verifica della riproducibilità

Il 24 settembre 2026 la catena è stata rieseguita su una copia pulita del
repository, con le copie locali dei corpora, con Python 3.12, numpy 2.2,
pandas 3.0, scipy 1.17 e matplotlib 3.11.

- `analisi_sweep_temperature.ipynb`: ogni CSV di `Risultati Sweep/` coincide
  con quello pubblicato.
- `rigenera_cap5_stoplist.py`: ogni CSV e ogni sintesi delle cartelle `_v2`
  coincide con quella pubblicata.
- `analisi_altmann_burstiness.ipynb`: i risultati coincidono a meno di
  differenze dell'ordine di $10^{-14}$. Fa eccezione la copertura lessicale di
  un singolo documento degenere, che differisce di $3 \cdot 10^{-5}$ senza
  cambiarne la classificazione.
- Figure: tutte le 35 figure della tesi che hanno un codice nel repository,
  cioè tutte tranne le sette del Capitolo 2 elencate più sotto, coincidono
  pixel per pixel con quelle in `Tesi/figure/`.
- `Tesi/main.tex` compila con pdfLaTeX senza errori né riferimenti irrisolti.

Tutte le procedure usano semi fissati, registrati nel codice e nei dataset.

---

## Che cosa non c'è, e perché

**I corpora di terzi.** Sono materiale altrui e non vengono ridistribuiti. Al
loro posto [`FONTI.md`](FONTI.md) riporta per ciascuno la fonte, la data di
acquisizione, gli identificatori per riscaricarlo e un'impronta crittografica
con cui verificare di avere gli stessi byte.

| corpus | ricostruibile? |
|---|---|
| *Guerra e pace* e tre romanzi di controllo | sì, da Project Gutenberg |
| Wikipedia | sì, dall'elenco dei titoli incluso, richiedendo le revisioni in vigore alla data di acquisizione |
| Grokipedia v0.1 | sì, dalle catture dell'Internet Archive |
| Grokipedia, versione attuale | **no** |

Grokipedia non pubblica una cronologia delle revisioni e riscrive le voci in
loco: le catture del 13 agosto 2026 sono l'unico testimone di quel testo. Le
misure che vi si appoggiano possono essere ricontrollate nei valori derivati,
non rifatte a partire dalla fonte.

**Il notebook che ha generato i cinque dataset.** Non è incluso.
`generazione_token_singolo.ipynb` ne riprende il protocollo: stesso endpoint,
stessi parametri, stessa generazione in continuazione e stesso schema dei
record. Cambia solo il prompt, ridotto a un singolo token. È un'ablazione non
usata nella tesi, e il suo output non è nel repository.

**Il codice di sette figure del Capitolo 2.** Le illustrazioni introduttive
(`cap2_zipf_lingue`, `cap2_heaps_confronto`, `cap2_heaps_dimensione_finita`,
`cap2_scomposizione_parole`, `cap2_scomposizione_token`, `cap2_embedding_2d`,
`cap2_burstiness_barcode`) sono state prodotte con codice che non è stato
conservato. Illustrano il quadro teorico, e nessuna misura dei Capitoli 4 e 5
ne dipende.

**Gli articoli di riferimento.** Sono soggetti al copyright degli editori e
vanno reperiti per DOI dalla bibliografia della tesi.

---

## Licenze

Il repository è distribuito sotto due licenze distinte.

- **[MIT](LICENSE)** per il codice: notebook e script.
- **[CC BY 4.0](LICENSE-DATA)** per i dati e i materiali documentali: i dataset
  JSONL, i file di risultato, le figure e il testo della tesi, sorgenti LaTeX
  compresi.

Chi intenda riutilizzare i testi generati verifichi anche le licenze dei
modelli che li hanno prodotti, che non sono uniformi all'interno della famiglia
Qwen.

## Come citare

> Samuele Zurlo, *Fra ordine e disordine: proprietà statistiche del linguaggio
> generato da Large Language Model*, tesi di laurea in Fisica, Università di
> Bologna, A.A. 2025/2026.

```bibtex
@thesis{zurlo2026ordine,
  author      = {Zurlo, Samuele},
  title       = {Fra ordine e disordine: propriet{\`a} statistiche del linguaggio
                 generato da Large Language Model},
  type        = {Tesi di laurea in Fisica},
  institution = {Alma Mater Studiorum -- Universit{\`a} di Bologna},
  year        = {2026}
}
```
