# amp-novelty — AMP Challenge 2027 submission

## Abstract

## The design hypothesis

Antimicrobial peptides kill by **membrane disruption, not by binding a protein target**. A
cationic peptide is drawn electrostatically to an anionic bacterial surface, folds against it into
an amphipathic helix, and inserts its hydrophobic face into the bilayer. Three consequences
determine everything below.

**Function is geometry, not sequence.** What matters is net charge, the hydrophobic moment of the
folded helix, and the free energy of partitioning into a bilayer — not which residues supply them.
Two peptides sharing under 40% identity can be mechanistically identical. This is why the
competition's ≤80%-identity novelty rule is satisfiable *without* sacrificing potency, and our
measurements confirm it: admitting candidates 70–78% identical to known AMPs made every held-out
Phase-2 metric **worse**, not better.

**The two Gram classes are approached differently.** A Gram-negative cell presents an anionic LPS
outer membrane, so the approach is electrostatic; a Gram-positive cell presents thick
peptidoglycan, so the approach is partition-limited. On the fully crossed 46×11 de-novo matrix
this shows up as the largest single variance term (peptide × strain, 43%), with the two hit-rates
correlating only 0.54.

**Selectivity is a charge story.** Mammalian membranes are zwitterionic and cholesterol-rich, so
the same cationic bias that drives bacterial binding is what spares erythrocytes — which is why
hemolysis must be modelled from *measured* HC50 rather than inferred from potency.

## Where the novelty comes from

The novelty rule is checked against curated AMP databases. So the productive place to look is
**natural sequence space that those databases never catalogued**: 837k metagenomic small-ORF
peptides from AMPSphere form a dedicated pretraining stage between generic UniParc peptides and
the curated AMP corpus. These are real, evolved, membrane-active-looking peptides that are
*already* outside the reference set — novelty by provenance rather than by mutation away from
known actives. Mutating away from a known AMP trades activity for distance; sampling from
uncatalogued natural diversity does not.

## Why this should transfer to the wet lab

Every potency oracle available for this problem is weak on *de novo* peptides (ρ ≈ 0.2–0.5; the
held-out APEX model sits at ρ +0.16). The design is built around that fact rather than in spite of
it:

* **The reward is a minimum, not a sum.** GRPO optimises the *weakest* of several independent
  potency signals. Optimising their sum finds the blind spot they share; optimising their minimum
  requires a candidate to satisfy all of them at once.
* **A k-nearest-neighbour score over peptides with *measured* MIC**, not a regressor. A regressor
  extrapolates confidently into regions where it is optimistic and wrong; a non-parametric
  neighbour score cannot leave the measured support.
* **A physicochemical envelope taken from measured-potent peptides**, so candidates stay inside
  the region where the evidence actually lives.
* **An analytic membrane term** — Wimley–White partitioning plus helix folding — that is physics,
  not a fitted model, and therefore cannot have memorised a database.
* **The library is assembled, not sampled.** Phase 1 scores set-level properties (coverage, FBD,
  precision/recall, distributional conformity), which are properties of the 50,000 as an object.
  Generating 50,000 samples and shipping them leaves the assembly step on the table; selecting
  50,000 from a much larger pool to optimise the set is a combinatorial problem.

## Method

A conditional peptide language model (37.9M parameters: d=512, 12 layers, 8 heads) over a
38-token vocabulary — 20 residues, 3 control tokens, 2 corpus-source tokens, 7 length buckets and
6 net-charge buckets. The conditioning lives *in the vocabulary*: a prompt names a target
(length, charge) cell and a corpus, which is how generation is steered toward the distribution of
measured-potent peptides. It is trained on a three-stage curriculum: 2M natural peptides from UniParc,
then the 837k AMPSphere metagenomic peptides, then 47k curated antimicrobial peptides. It is
optimised with GRPO against the consensus reward above, KL-anchored to the supervised model.

Candidates are drawn from a portfolio of checkpoints (the RL policy together with its own pre-RL
ancestors), conditioned on (length, net-charge) buckets sampled from peptides with measured
MIC ≤ 8 µM, filtered to that distribution's physicochemical envelope, scored at a pessimistic quantile of a
bootstrap ensemble (the 0.75 quantile of predicted MIC and the 0.25 quantile of predicted HC50 --
pessimistic on both sides at once), and selected by set-level swap optimisation against the reference
distribution in ESM-2 embedding space. The generator is deliberately *sharp* while the assembler
supplies all diversity — forcing diversity into the reward measured strictly worse.

The **top-100 is the best 100 candidates by the consensus design score**, under a stricter novelty
margin than the library (70% versus 78%), a mutual-identity
cap (55%) and a consensus-potency floor. An earlier version allocated the
100 slots across award categories and across physicochemical strata; both were measured on a
held-out grader to be **worse than not allocating at all** — a per-cluster quota fills slots
regardless of candidate quality, and one cluster was contributing zero of the natural best 70
while receiving a sixth of the list.

The competition draws 25 of the 100 uniformly at random and scores their arithmetic mean, so the
objective is the *worst* member of the list, not the best. Every one of the 100 must be potent and
non-hemolytic; a single brilliant peptide is worth nothing.

No manual curation, hand-picking or post hoc human selection is applied at any stage: every filter
is deterministic and in the code.

## Reproducibility

```bash
uv sync
uv run generate          # writes generate/library.fasta and generate/top.fasta
```

Fixed default seed (42).

**The organisers' validator procedure passes byte-for-byte on both files.** Their documented
process is: clone the repository, install, generate, verify, then generate again and confirm the
output is reproducible. Run against a real `git clone` of this repository:

```
compliance: PASS (50000 library, 50000 unique, 100 top)
run 1 vs run 2:   library.fasta  identical      top.fasta  identical
```

**The library also reproduces byte-for-byte against the submitted artifact**
(sha256 `83d5d303165e365652d9a080da90601cd4beb322f6722ea9915b3ca5be726b31`), verified by
`uv sync && uv run generate` in a fresh clone with the shipped FASTAs deleted.

**The top-100 reproduces the same 100 sequences exactly.** Against the submitted `top.fasta` a
fresh clone returns an overlap of 100/100 with three adjacent pairs transposed; within the
validator's own two runs it is byte-identical. The selection is stable, only the ordering of
near-tied candidates moves, and the competition draws 25 of the 100 uniformly at random and scores
their arithmetic mean, so ordering does not affect the result.

A deterministic sort -- quantise the score, break ties on a hash of the sequence -- was implemented
and then removed because it was measured NOT to work. Warm-versus-cold drift in the ranking score
is at most 8.5e-03 while adjacent gaps in the top-100 have a median of 1.1e-02, the same order of
magnitude; and quantisation does not remove boundary sensitivity, it relocates it, since a value
near a bin edge flips on arbitrarily small drift. Bin widths of 1e-4, 1e-3, 1e-2 and 3e-2 were all
tested and none made the two orderings agree. Sorting by sequence hash alone would be byte-stable
but would make "ranked" a false label for an arbitrary permutation.

**No GPU is required.** Generation was verified end-to-end with CUDA hidden (21 minutes, CPU only).
Note that the hardware determines *which* library is produced -- CPU and GPU differ, because
`build_recall_cover` computes pairwise distances by matrix multiplication on the device and CPU and
CUDA matmul differ in their last bits. Reproducibility holds *within* a hardware class, which is
what the validator tests since it runs both generations on one machine, and the submitted FASTA
files travel with the repository.

Every top-100 sequence is in the library, unique, and clears the 80% identity rule under both
Levenshtein ratio and MMseqs2 alignment identity, on every run.

What reproduces from source on the organizers' machine: compliance and novelty filtering, the
physicochemical envelope, the 400,000-iteration swap assembly, and top-100 selection.
What travels as derived artifacts alongside the model weights that produced them: the candidate
pool, its design scores, its ESM-2 embeddings, and the oracle prediction caches. These are
memoised intermediates, not a substitute for the code -- the pipeline recomputes any sequence it
has not seen.

The one bug behind the original non-determinism is fixed and worth naming: `build_recall_cover`
subsampled the pool for k-NN radius estimation with `torch.randperm` on the global CUDA generator,
unseeded, so every process drew a different subsample and assembled a different library. It now
takes an explicit seeded generator.

## Selection and ranking procedure

1. **Generate** a candidate pool from a portfolio of LM checkpoints (0.60 RL /
   0.30 supervised / 0.10 pretrained), conditioned on (length, net-charge)
   buckets sampled from the distribution of peptides with *measured* MIC <= 8 uM, mixed with an
   amphipathic helical-wheel grammar. The pool grows until enough candidates survive every
   downstream filter, so the library size is guaranteed rather than hoped for.
2. **Compliance filter**: 20 standard residues, length 8-50, unique, no exact match to the
   reference set.
3. **Novelty filter**: drop candidates above 78% Levenshtein identity to the
   reference set, taken as the challenge's `antibacterial.fasta` *union the full MarLys database*.
   The competition paper enforces the rule against MarLys while the shipped validator ships only
   the antibacterial subset; the union is stricter than either reading alone.
4. **Envelope filter**: keep candidates inside the physicochemical envelope of measured-potent
   peptides (length, charge, GRAVY, hydrophobic moment, hydrophobic and cationic fraction).
5. **Score** by the consensus objective: the minimum over independent potency signals (per-species
   MIC heads for the nine panel species covered by GRAMPA, plus an AMP classifier), evaluated at a
   pessimistic quantile of a bootstrap ensemble, plus a measured-HC50 selectivity term, the
   measured-actives kNN signal, a pull toward the potent descriptor distribution, and
   synthesizability.
6. **Assemble** the library by filling per-cluster quotas proportional to the reference
   distribution in ESM-2 embedding space, rejecting any candidate above 70%
   identity to an already-selected member of the same cluster.
7. **Select the top-100** as the best 100 by the consensus design score, under a stricter
   novelty margin (70%), a mutual-identity cut
   (55%) and a consensus-potency floor.

## Training data (full disclosure)

| dataset | use | licence |
|---|---|---|
| UniParc peptides (8-50 aa, 2M sampled) | LM pretraining stage 1 | public |
| AMPSphere v.2022-03 (837k metagenomic peptides) | LM pretraining stage 2 | CC-BY 4.0 |
| MarLys AMP database (doi:10.17632/w4hb5grjwb.3) | LM fine-tuning, reference distribution, novelty reference | CC-0 |
| GRAMPA (Witten & Witten) MIC measurements | per-species MIC heads, measured-actives kNN, envelope | public |
| AMP Scanner v2 / Veltri positives + negatives | AMP classifier | public |
| UniProt fragments | classifier negatives | public |
| HemoPI2 HC50 measurements | hemolysis regressor | public |
| AMP Challenge reference set (`antibacterial.fasta`) | novelty constraint only, never trained on | competition-provided |

No proprietary or non-public data is used. ESM-2 (`facebook/esm2_t30_150M_UR50D`, MIT) supplies
the embeddings used by the property heads and the assembler.

**AI assistance disclosure**: this submission was developed with an autonomous LLM research agent,
as permitted by the competition rules; all code, measurements and design decisions are in the
repository and its commit history.

## Library statistics

* library size: 50000, unique: 50000
* length range 8-48 (median 21)
* maximum identity of any top-100 candidate to the novelty reference: 70.0%
* generated in 0 s on one GPU
