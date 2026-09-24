# Fonti, identificatori e impronte

Questo repository contiene **codice, risultati, figure e i dataset generati**.
Non contiene i corpora di terzi: sono materiale altrui, e ridistribuirlo
porrebbe problemi di licenza che non riguardano il resto del lavoro.

Questo file è ciò che rende comunque verificabile la tesi. Per ogni corpus
riporta la fonte, la data di acquisizione, gli identificatori esatti e
un'impronta crittografica con cui chiunque riscarichi il materiale può
dimostrare di avere gli stessi identici byte.

---

## 1. Come si verifica un'impronta

Per i file singoli, lo SHA-256 del file.

Per le cartelle si usa un'**impronta aggregata**: si elencano i file in ordine
alfabetico con il proprio SHA-256, una riga `nome sha256` ciascuno, e si
calcola lo SHA-256 di quell'elenco. Il comando che la riproduce:

```bash
find CARTELLA -maxdepth 1 -type f -printf '%f\n' | LC_ALL=C sort | while read f; do
  printf '%s %s\n' "$f" "$(sha256sum "CARTELLA/$f" | cut -d' ' -f1)"
done | sha256sum
```

Due copie dello stesso corpus danno la stessa impronta se e solo se contengono
gli stessi file con lo stesso contenuto.

---

## 2. Dataset generati — inclusi nel repository

Testo prodotto dai cinque modelli Qwen lungo lo sweep di temperatura descritto
nel §3.2 della tesi: dodici temperature da 0.4 a 1.5 con passo 0.1, prompt di
circa 2000 token da *Guerra e pace* (dettagli nel §3.4 di questo file),
campionamento a sola temperatura (`top_p = 1`, `top_k = -1`,
`repetition_penalty = 1`), 24 000 token per documento. 242 documenti in tutto.

I nomi sono stati uniformati al momento della pubblicazione; l'impronta è
calcolata sul contenuto e non cambia con il nome del file.

| file | byte | SHA-256 |
|---|---|---|
| `sweep_qwen2.5-1.5b.jsonl` | 6 153 618 | `647fce41b679bcc25e7bd992d3a117675810e28b8897c841de1817661a08ee48` |
| `sweep_qwen2.5-3b.jsonl` | 5 412 606 | `bb8cc1d497492f357b067289ef6a6f6d2aa69055f85901079db109e8242a3b87` |
| `sweep_qwen2.5-14b.jsonl` | 1 232 215 | `914f056f2e86fb6645e23e427f091b1f1b0156d9c1b32d6c27d27554ea24d678` |
| `sweep_qwen3.5-2b-base.jsonl` | 6 006 175 | `5f4b49fff75599da3ce35586fa15bbef6e9de83a571ac9f2b956b5bc215fa2a9` |
| `sweep_qwen3.5-4b-base.jsonl` | 7 248 193 | `da8f42898945adacf7162b81e4aa3d6ebad63acb4328dd2bcba0872ad4bed9a6` |

### Modelli generanti

| modello | tokenizzatore | documenti |
|---|---|---|
| Qwen2.5-1.5B | `Qwen/Qwen2.5-1.5B` | 57 |
| Qwen2.5-3B | `Qwen/Qwen2.5-3B` | 53 |
| Qwen2.5-14B | `Qwen/Qwen2.5-14B` | 12 |
| Qwen3.5-2B-Base | `Qwen/Qwen3.5-2B-Base` | 60 |
| Qwen3.5-4B-Base | `Qwen/Qwen3.5-4B-Base` | 60 |

Generati tramite deployment personali su DeepInfra fra il 28 e il 30 luglio
2026. Ogni riga dei JSONL registra temperatura, seme, parametri di
campionamento, numero di riprese e posizioni dei token di fine sequenza, così
che ciascun documento sia ricostruibile nelle sue condizioni di produzione.

> Chi intenda riutilizzare questi testi verifichi le condizioni delle licenze
> dei singoli modelli: non sono uniformi all'interno della famiglia Qwen.

---

## 3. Corpora non inclusi

### 3.1 Wikipedia — riscaricabile

- **Fonte:** `https://en.wikipedia.org`, testo corrente delle voci
- **Scaricato il:** 13 agosto 2026 (134 voci), 15 agosto 2026 (3 voci)
- **Titoli:** tutti e 137 i candidati sono elencati, con i controlli di qualità,
  in `Wiki/risultati_wiki/dati/articoli_qualita.csv`; 108 superano i controlli
  (`ammesso = True`) ed entrano nell'analisi della sola Wikipedia. I 40
  appaiati con Grokipedia sono la colonna `titolo` di
  `Wiki/risultati_confronto/dati/riepilogo.csv`, e 39 di essi formano le terne
  del Capitolo 5 (*Buddhism* è esclusa, §3.3 della tesi). Entrambi i file sono
  **inclusi nel repository**
- **Attenzione:** i 40 titoli erano stati scelti, il 13 agosto, come i 40 più
  lunghi fra gli ammessi. Le tre voci scaricate il 15 agosto hanno cambiato
  `articoli_qualita.csv`, e con la versione attuale di quel file la stessa
  regola sceglierebbe *Istanbul* al posto di *Rome*. Fa fede quindi
  `riepilogo.csv`, e `confronto_grokipedia.ipynb` legge i titoli da lì.
  Tre file della stessa cartella (`confronto_testo.csv`,
  `metadati_grokipedia.csv`, `sovrapposizione.csv`) vengono da una
  riesecuzione del 15 agosto e contengono *Istanbul* invece di *Rome*. I valori
  citati nella tesi vengono da `riepilogo.csv`: sovrapposizione di 8-grammi con
  mediana 0.0003 e massimo 0.046. Sull'altro insieme di titoli la mediana vale
  0.0004 e il massimo non cambia
- **Estrazione:** rimozione di marcatura, tabelle, note e sezioni di apparato
  (*See also*, *References*, *Notes*, *Bibliography*, *External links* e
  affini), poi troncamento alla lunghezza di analisi
- **Licenza della fonte:** CC BY-SA 4.0
- **Impronta aggregata:** 137 file,
  `4578445982c6488f03157a6e436141d1f999496d7be21a2c8ab5bab2aee19fc4`

> **Avvertenza sulla riproducibilità.** Il download non ha fissato un
> identificatore di revisione: chi riscarica oggi ottiene un testo più recente
> e non otterrà la stessa impronta. Per la riproduzione esatta occorre
> richiedere la revisione corrente alla data di acquisizione, con l'API di
> Wikipedia e `rvstart=2026-08-13T00:00:00Z`, e scaricare poi ciascuna voce
> con `?oldid=`. Gli `oldid` così ricavati andrebbero aggiunti a questo file.

### 3.2 Grokipedia, versione attuale — non riscaricabile

- **Fonte:** `https://grokipedia.com/page/<slug>`, stessi 40 titoli
- **Scaricato il:** 13 agosto 2026 (40 voci), 15 agosto 2026 (1 voce)
- **Impronta aggregata:** 41 file,
  `1570642064d37ab77f87a593175337d60ef7f810c7c2cd94e1bcae3963fd1892`

> **Questo corpus non è ricostruibile.** Come documentato nel §3.3 della tesi,
> la piattaforma non espone alcun endpoint di cronologia — `history`,
> `versions` e `revisions` restituiscono tutti 404 — e le voci vengono
> riscritte in loco. Le catture del 13 agosto 2026 sono l'unico stato
> disponibile di quel testo. Chi disponga di catture proprie può usare
> l'impronta qui sopra per stabilire se coincidono con quelle impiegate; una
> riproduzione indipendente delle misure del Capitolo 5 su questo corpus non è
> però possibile senza i file.

### 3.3 Grokipedia v0.1 — riscaricabile

- **Fonte:** Internet Archive, catture di ottobre 2025 delle stesse 40 voci
- **Recuperato il:** 13 agosto 2026
- **Impronta aggregata (testo estratto):** 40 file,
  `cae1b45cd29990b140ccce0cf70783435ce0f332e96227cec070e6e939597191`
- **Impronta aggregata (catture grezze):** 41 file,
  `f8cb016f429f89e49e1b899edfbed21e9cde747d9aebd82fc74a8fbccd6814bf`

Le catture dell'Internet Archive sono pubbliche e permanenti, quindi questo
corpus è ricostruibile. La versione v0.1 di ogni voce è la **prima** cattura
archiviata, e il suo timestamp è nella colonna `prima` di
`Wiki/risultati_confronto/dati/revisioni_wayback.csv`, incluso nel repository.
L'indirizzo della cattura grezza è

```
https://web.archive.org/web/<prima>id_/https://grokipedia.com/page/<Titolo_con_underscore>
```

e `confronto_grokipedia.ipynb` ne estrae il testo con la stessa procedura usata
per la versione attuale.

### 3.4 *Guerra e pace* — riscaricabile

- **Fonte:** Project Gutenberg, ebook n. 2600 (traduzione Garnett)
- **URL:** `https://www.gutenberg.org/cache/epub/2600/pg2600.txt`
- **Inizio del testo:** prima occorrenza di `Well, Prince, so Genoa and Lucca`
- **Prompt dello sweep:** tutti i prompt partono dallo stesso punto del
  romanzo, i caratteri 8380–16642 del file Gutenberg. Quattro dataset su cinque
  usano gli stessi 8262 caratteri (`prompt_sha256_16 = 494c4747edcbe6a8`):
  sono 2000 token per i tre Qwen2.5 e 2005 per Qwen3.5-4B-Base, che ha
  ricevuto lo stesso testo (`prompt_mode = same_text`). Solo Qwen3.5-2B-Base
  usa 8239 caratteri (`a49d9f8ad18e16b4`), cioè i 2000 token esatti del suo
  tokenizzatore. Il §3.3 della tesi attribuisce la lunghezza di 8239 caratteri
  a due dataset: i file mostrano che è uno solo
- **Stato:** pubblico dominio; si applicano le condizioni d'uso di Project
  Gutenberg al file così come distribuito
- **Impronta aggregata:** 4 file,
  `9602b209a2496b9e02c696d3a01a543ca89bd6165a9275bad6a5f1c9639ebd07`.
  È l'impronta della cartella `corpora_cache/`, che contiene *Guerra e pace*
  (`f6c229e9b581.txt`, nome derivato dall'URL) insieme ai tre romanzi del §3.5

### 3.5 Romanzi di controllo — riscaricabili

Usati dagli script `controllo_letterario.py`, `controllo_funzione*.py` e
`baseline_quattro_romanzi.py` per verificare che i risultati non dipendano dal
solo *Guerra e pace*. Tutti da Project Gutenberg, pubblico dominio.

| romanzo | ebook | file in `corpora_cache/` |
|---|---|---|
| *Moby Dick* | 2701 | `moby_dick.txt` |
| *Middlemarch* | 145 | `middlemarch.txt` |
| *David Copperfield* | 766 | `david_copperfield.txt` |

Gli URL hanno la forma `https://www.gutenberg.org/cache/epub/<n>/pg<n>.txt`.

### 3.6 Figure illustrative del Capitolo 2 — fonti non documentate

Le figure del Capitolo 2 sulla legge di Zipf in cinque lingue (*Moby-Dick*,
*I promessi sposi*, *Don Quijote*, *Vingt mille lieues sous les mers*,
*Faust*) e sulle altre illustrazioni introduttive sono state prodotte con
codice che non è stato conservato. Per questo non se ne conoscono gli
identificatori delle edizioni.

---

## 4. Risorse esterne di analisi

| risorsa | provenienza |
|---|---|
| `cl100k_base` | tokenizzatore di analisi, unico per tutti i corpora, via `tiktoken` |
| `glove-wiki-gigaword-100` | vettori GloVe a 100 dimensioni, dalle release del repository `RaRe-Technologies/gensim-data` su GitHub (circa 134 MB) |

Nessuna delle due è ridistribuita qui: entrambe vengono scaricate dai notebook
al primo utilizzo. Il file GloVe viene scaricato direttamente, senza passare
per il pacchetto `gensim`, e salvato in `Analisi Sweep di temperature/cache/`.

---

## 5. Semi e determinismo

Tutte le procedure sono deterministiche a semi fissati, e i semi sono
registrati nel codice e nei dataset. I notebook salvano ogni risultato
intermedio in file separati e in formato aperto, insieme a una sintesi
testuale dei parametri impiegati, così che ciascuna figura sia ricostruibile
senza ripetere i calcoli.
