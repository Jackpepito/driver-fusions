# Seq Recon: ricostruzione della sequenza da evento di fusione

Questo modulo ricostruisce la proteina di fusione a partire da un evento genomico `gene5-bp5 :: gene3-bp3`, in modo **isoforma-aware**.

File principali:
- `run.py`: orchestrazione su dataset (ChimerSeq), selezione della migliore variante, salvataggio output.
- `seq_reconstruction.py`: logica core di ricostruzione (trascritti, CDS, mapping breakpoint, traduzione).
- `utils.py`: analisi e plotting dei risultati.

## 1) Input dell'evento di fusione

Per ogni riga del dataset servono almeno:
- gene 5': `H_gene`
- gene 3': `T_gene`
- cromosoma e posizione 5': `H_chr`, `H_position`
- cromosoma e posizione 3': `T_chr`, `T_position`
- build genomica: `Genome_Build_Version` (`hg19` o `hg38`)

In `run.py`:
- i cromosomi vengono normalizzati togliendo il prefisso `chr` (`H_chr_clean`, `T_chr_clean`),
- le posizioni sono convertite a numerico,
- righe invalide vengono scartate.

## 2) Inizializzazione delle reference

Per ogni genome build usata nei dati viene creato un `IsoformAwareFusionReconstructor`.

In modalità `local` (quella usata nel run standard):
- carica il FASTA genomico in memoria,
- parse del GTF e costruzione cache `gene -> transcripts`,
- cache su disco (`cache/gtf_<build>_chimerseq/gtf_transcripts.pkl`) per evitare parsing ripetuto.

Ogni `Transcript` contiene:
- `transcript_id`, `chromosome`, `strand`, lista di esoni,
- `cds_start`, `cds_end` genomici (derivati come min/max delle regioni CDS nel GTF).

## 3) Costruzione CDS per una isoforma

Funzione: `build_cds_sequence(tx)`.

Passi:
1. prende gli esoni del trascritto (già ordinati nello verso del trascritto),
2. per ogni esone calcola l'intersezione con l'intervallo CDS `[cds_start, cds_end]`,
3. estrae la sequenza genomica (reverse-complement se strand `-`),
4. concatena i segmenti in una CDS unica 5'->3'.

Output:
- `cds_seq`: CDS concatenata,
- `mapping`: lista di tuple `(g_start, g_end, cds_start, cds_end)` per mappare coordinate genomiche in coordinate CDS.

## 4) Mapping del breakpoint in spazio CDS

Funzione: `map_breakpoint_to_cds(breakpoint, mapping)`.

- Se il breakpoint cade in un segmento mappato, ritorna la posizione 0-based in CDS.
- Se cade fuori, ritorna `None`.

## 5) Estrazione frammenti 5' e 3'

Funzione: `extract_cds_fragment(tx, breakpoint, role, allow_approximation=True)`.

- `role="head"` (gene 5'): prende `CDS[0:cds_pos]`.
- `role="tail"` (gene 3'): prende `CDS[cds_pos:end]`.

Se il breakpoint non è in CDS e `allow_approximation=True`, il codice approssima:
- prima della CDS -> usa inizio CDS,
- dopo la CDS -> usa fine CDS,
- in introne/non mappabile -> sceglie inizio o fine CDS in base alla distanza minima.

Ritorna anche flag:
- `is_in_cds` (breakpoint esatto in CDS),
- `is_approximated`.

## 6) Enumerazione combinazioni isoforma-aware

Funzione: `reconstruct_isoform_fusions(...)`.

Per ogni combinazione `tx5 x tx3`:
1. estrae `cds_head` dal partner 5' e `cds_tail` dal partner 3',
2. scarta se uno dei due frammenti è vuoto,
3. scarta se un frammento è più corto di `min_cds_len` (default 30 nt),
4. costruisce `fusion_cds = cds_head + cds_tail`,
5. calcola `in_frame` con `len(fusion_cds) % 3 == 0`.

Se `allow_out_of_frame=False`, le out-of-frame sono scartate.

## 7) Traduzione in proteina (ORFfinder o fallback)

Se `use_orffinder=True`:
- esegue ORFfinder sulla `fusion_cds` (`-ml 30`, `-n true`, `-s 0`),
- prende il primo ORF restituito,
- salva metadati ORF (`orf_start`, `orf_end`, `orf_frame`),
- se `orf_frame == 0`, marca la fusione come `in_frame=True`.

Se ORFfinder fallisce (o disabilitato):
- traduzione diretta con `translate_seq`,
- stop al primo codone di stop.

Per ogni combinazione valida produce un `FusionIsoformResult` con:
- trascritti scelti (`tx5_id`, `tx3_id`),
- lunghezze (`cds5_len`, `cds3_len`, `fusion_cds_len`),
- proteina `protein_seq`,
- flag frame/approssimazione,
- `quality`.

## 8) Scelta della variante finale per la riga

In `run.py`, dopo aver ottenuto tutte le varianti:
- priorità 1: varianti `in_frame`,
- priorità 2: tra quelle candidate, proteina più lunga.

La sequenza salvata è `best.protein_seq` senza `*`.

Vengono anche salvati:
- `num_variants`, `recon_in_frame`, `recon_quality`,
- `bp5_approx`, `bp3_approx`,
- `best_tx_pair`, `orf_used`, `orf_frame`,
- `frame_consistent` (coerenza tra annotazione dataset e ricostruzione).

## 9) Categorie qualità

Assegnate in `reconstruct_isoform_fusions`:
- `perfect`: entrambi i breakpoint in CDS e in-frame,
- `out_of_frame`: breakpoint in CDS ma out-of-frame,
- `approximate`: almeno un breakpoint approssimato e in-frame,
- `approximate_out_of_frame`: almeno un breakpoint approssimato e out-of-frame,
- `good`: fallback residuale.

## 10) Output

Con `--output <prefix>` il pipeline salva:
- `<prefix>_results.csv`: tabella completa (input + campi ricostruzione),
- `<prefix>_log.txt`: riepilogo run e metriche aggregate,
- file di analisi/plot aggiuntivi se ci sono ricostruzioni riuscite.

## 11) Comando tipico

```bash
python src/seq_recon/run.py \
  --input /work/H2020DeciderFicarra/gcapitani/driver-fusion/data/chimerseq_labeled.csv \
  --genome-build all \
  --output chimerseq_analysis
```

Per disabilitare ORFfinder:

```bash
python src/seq_recon/run.py --no-orffinder --output chimerseq_no_orf
```

## 12) Note pratiche importanti

- Il metodo è isoforma-aware: una stessa fusione gene-gene può produrre più varianti proteiche.
- Le approssimazioni breakpoint aumentano la copertura ma possono ridurre fedeltà biologica.
- La build genomica deve combaciare con la riga (`hg19` vs `hg38`) per evitare mismatch di coordinate.
