"""Deterministic data for amp_design. FIXED FILE — do not edit.

Everything the referee reads comes from here, and every split is fixed by construction so two
runs of evaluate.py see byte-identical data. Build the files once with prepare_data.py.

The eval / val separation is the anti-self-deception device:

    eval side   ref_amp_eval  +  half of the labelled potency data
    val  side   ref_amp_val   +  the other half, disjoint by sequence family

so a library that matches the eval reference because it *memorised* it does not match the val
reference. Sequence families are kept whole on one side (bucketed by prefix + length band),
which is what makes the split meaningful for peptides — near-duplicates are everywhere in AMP
databases and a random split would leak straight through them.
"""

import functools
import hashlib
import gzip
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

AA = "ACDEFGHIKLMNPQRSTVWY"
AASET = set(AA)
MIN_LEN, MAX_LEN = 8, 50
LIBRARY_SIZE = 50_000        # the competition's required library size
TOP_K = 100                  # the competition's required top list size
POTENCY_THRESHOLD_UM = 16.0  # a "success" in the competition's wet-lab scoring
MIC_CEILING_UM = 64.0
HC50_CEILING_UM = 128.0


def _load(name):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing — run: python tasks/amp_design/prepare_data.py")
    if name.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return json.load(f)
    with open(path) as f:
        return json.load(f)


@functools.lru_cache(maxsize=None)
def challenge_reference():
    """The 39,448 known antibacterial peptides the competition checks submissions against.

    Library rule: no sequence may be identical to one of these.
    Top-100 rule: no sequence may exceed 80% identity (Levenshtein ratio) to any of these.
    """
    return tuple(_load("antibacterial.json"))


@functools.lru_cache(maxsize=None)
def novelty_reference():
    """The set the 80%-identity rule is actually checked against.

    The validator ships `antibacterial.fasta` (39,448 peptides), but the competition paper states
    the top-100 rule is enforced against the **MarLys** reference database via MMseqs2. MarLys
    contains 7,963 further valid peptides, and they matter: a top-100 built against the shipped
    subset alone had a candidate at 88% identity to a MarLys-only entry — which the organizers
    would silently replace with "the next valid candidate". So every novelty check here uses the
    union, which is compliant under either reading.

    DRAMP 3.0's antibacterial set was checked against that union as a third opinion: of its 20,227
    valid peptides, 20,142 were already covered and only 85 were new — 43 of them with no neighbour
    above 80% identity to anything we hold. That is a small blind spot but an absolute one, since a
    single top-100 candidate landing near one of those 43 is silently replaced by the organisers.
    Adding the 85 costs nothing measurable in the identity check; adding all 20,227 would have
    grown the hot-loop comparison by 40% for the same 85 sequences of benefit.

    Peptides published as antibacterial after those databases were built are added the same way,
    from known_extra.json: the competition forbids sequences identical to known antibacterials, and
    a database snapshot is not the same thing as the literature. Currently the 36 measured-active
    M3-CAD peptides (doi:10.1002/advs.202515835). Free insurance rather than a fix — nothing in the
    delivered library comes within 69% identity of any of them, so this changes no output today and
    only guards against a future generator drifting towards them.
    """
    extra = []
    for name in ("dramp_extra.json", "known_extra.json"):
        if os.path.exists(os.path.join(DATA, name)):
            extra.extend(_load(name))
    return tuple(sorted(set(challenge_reference()) | {r["seq"] for r in marlys()} | set(extra)))


@functools.lru_cache(maxsize=None)
def reference_amps(split="eval"):
    """Known antibacterial peptides used as the distributional target, split-disjoint."""
    return tuple(_load({"eval": "ref_amp_eval.json", "val": "ref_amp_val.json",
                        "all": "ref_amp_all.json"}[split]))


@functools.lru_cache(maxsize=None)
def marlys():
    """The full MarLys AMP records (sequence + activity labels + published properties)."""
    return _load("marlys.json.gz")


@functools.lru_cache(maxsize=None)
def anchor_potent():
    """Known peptides with measured MIC <= ~8 uM against E. coli (GRAMPA). The 'good' pole."""
    return tuple(_load("anchor_potent.json"))


@functools.lru_cache(maxsize=None)
def anchor_weak():
    """Known peptides with measured MIC >= 100 uM. Real peptides that do not work."""
    return tuple(_load("anchor_weak.json"))


@functools.lru_cache(maxsize=None)
def anchor_uniprot():
    """Generic UniProt fragments — the negative pole (no antimicrobial claim)."""
    return tuple(_load("uniprot_test.json"))


@functools.lru_cache(maxsize=None)
def anchor_decoy(n=4000, seed=7):
    """Synthetic decoys: composition-matched shuffles of known AMPs.

    Same amino-acid composition as real AMPs, destroyed sequence order. A metric that cannot
    separate these from real AMPs is measuring composition, not design.
    """
    rng = random.Random(seed)
    src = list(reference_amps("all"))
    rng.shuffle(src)
    out = []
    for s in src[:n]:
        chars = list(s)
        rng.shuffle(chars)
        out.append("".join(chars))
    return tuple(out)


@functools.lru_cache(maxsize=None)
def baseline_library():
    """The official HydrAMP baseline library (50,000) — the published Phase-1 target to beat."""
    return tuple(_load("baseline_hydramp_library.json.gz"))


@functools.lru_cache(maxsize=None)
def baseline_top():
    """The official HydrAMP baseline top-100."""
    return tuple(_load("baseline_hydramp_top.json"))


# ------------------------------------------------------------------ labelled data for surrogates

def _split_half(seqs, split, seed=11):
    """Family-aware halving: a prefix+length bucket lands wholly in eval or in val.

    The bucket's side is decided by a hash of the bucket key alone, NOT by its position in a
    shuffled list of the keys that happen to be present. That distinction is the whole point: under
    the old rule, adding a single peptide that opened a new bucket changed `sorted(buckets)` and
    therefore reshuffled every other bucket's assignment — so growing the training data silently
    moved sequences across the held-out boundary, and no comparison spanning a data change could be
    trusted. Under this rule a bucket's side is a property of the bucket, fixed for all time, and
    new data lands on whichever side it belongs to without disturbing anything already placed.
    """
    if split == "all":
        return list(seqs)
    out = []
    for s in seqs:
        key = f"{seed}:{s[:3]}:{len(s) // 6}"
        side = "eval" if hashlib.sha256(key.encode()).digest()[0] < 128 else "val"
        if side == split:
            out.append(s)
    return out


@functools.lru_cache(maxsize=None)
def classifier_data(split="eval", large=False):
    """(positives, negatives) for the AMP classifier surrogate.

    `large` (the full 47k curated AMP set against UniProt fragments and shuffles) was tried and
    is OFF by default because it measured WORSE on the population that matters: de-novo AUC fell
    from 0.709 to 0.611 and rho from +0.132 to +0.071. More database AMPs teach the classifier
    "looks like a database AMP" — which is the failure mode, not the objective. The 2k Veltri
    benchmark, curated as a discrimination task rather than scraped for volume, transfers better.
    Data volume is not the lever when the extra data carries the bias you are trying to escape.
    """
    if not large:
        pos = _load("veltri_positive.json")
        neg = _load("veltri_negative.json")
        uni = _load("uniprot_train.json")[:6000]
        return _split_half(pos, split), _split_half(neg + uni, split)

    pos = sorted({r["seq"] for r in marlys()
                  if "antibacterial" in r["act"] or "antimicrobial" in r["act"]})
    neg = _load("veltri_negative.json") + _load("uniprot_train.json")[:30000]
    import random as _r
    rng = _r.Random(4)
    shuffles = []
    for s in pos[:8000]:
        c = list(s)
        rng.shuffle(c)
        shuffles.append("".join(c))
    return _split_half(pos, split), _split_half(sorted(set(neg + shuffles)), split)


@functools.lru_cache(maxsize=None)
def mic_data(split="eval"):
    """(sequences, log10 MIC in uM) from GRAMPA, family-split."""
    rows = _load("grampa_mic.json")
    by_seq = {r["seq"]: r["log10_mic"] for r in rows}
    seqs = _split_half(sorted(by_seq), split)
    return seqs, [by_seq[s] for s in seqs]


@functools.lru_cache(maxsize=None)
def mic_panel_data_pos(split="eval"):
    """(sequences, log10 MIC) against Gram-positive bacteria, median over strains (GRAMPA)."""
    return _panel("mic_gram_pos.json", split)


@functools.lru_cache(maxsize=None)
def mic_panel_data_neg(split="eval"):
    """(sequences, log10 MIC) against Gram-negative bacteria, median over strains (GRAMPA)."""
    return _panel("mic_gram_neg.json", split)


def _panel(name, split):
    rows = _load(name)
    by_seq = {r["seq"]: r["log10_mic"] for r in rows}
    seqs = _split_half(sorted(by_seq), split)
    return seqs, [by_seq[s] for s in seqs]


# The competition's panel by species (Appendix B), restricted to species GRAMPA actually covers.
# Broad-spectrum is the headline category and it is a mean over 20 strains, so "potent" has to
# mean potent against several species at once — a gram+/gram- aggregate cannot express that.
PANEL_SPECIES = ["E. coli", "S. aureus", "P. aeruginosa", "K. pneumoniae", "A. baumannii",
                 "E. faecalis", "E. faecium", "B. subtilis", "S. typhimurium"]


@functools.lru_cache(maxsize=None)
def mic_species_data(species, split="eval"):
    """(sequences, log10 MIC) against one species, median over its strains."""
    rows = grampa_full()
    agg = {}
    for r in rows:
        if r["bacterium"] == species:
            agg.setdefault(r["seq"], []).append(r["log10_mic"])
    by_seq = {s: sorted(v)[len(v) // 2] for s, v in agg.items()}
    seqs = _split_half(sorted(by_seq), split)
    return seqs, [by_seq[s] for s in seqs]


@functools.lru_cache(maxsize=None)
def hemolysis_data(split="eval"):
    """(sequences, log10 HC50 in uM) from HemoPI2 — the selectivity category's ground truth."""
    import math
    rows = _load("hemolysis_hc50.json")
    by_seq = {r["seq"]: math.log10(max(r["hc50_um"], 0.1)) for r in rows}
    seqs = _split_half(sorted(by_seq), split)
    return seqs, [by_seq[s] for s in seqs]


@functools.lru_cache(maxsize=None)
def pretrain_corpus():
    """(amp_sequences, background_sequences) for language-model pretraining."""
    return _load("pretrain_amp.json.gz"), _load("pretrain_background.json.gz")


@functools.lru_cache(maxsize=None)
def ampsphere():
    """836,966 metagenomic small-ORF peptides (AMPSphere v.2022-03, CC-BY).

    Natural, AMP-like, and almost entirely absent from the curated databases the competition's
    novelty rule is checked against — i.e. the region the 80%-identity constraint rewards. Used
    as a pretraining stage between generic UniParc peptides and the curated AMP corpus.
    """
    return _load("pretrain_ampsphere.json.gz")


# GRAMPA arrives here through the HydrAMP starter kit, which kept only peptides flagged
# is_modified == False — 27,116 of the original 51,345 measurements. The 15,989 rows it dropped are
# almost entirely C-TERMINALLY AMIDATED peptides: a mild, extremely common natural modification that
# caps the C-terminus, removes its negative charge and typically raises activity. Including them
# adds 1,675 peptides and 82 organisms; treating them as if they were plain sequences adds a
# systematic label shift of about +1 net charge.
#
# Whether that trade is worth taking is an experiment, not an assumption — this project already
# holds a verdict that growing the AMP classifier's training set from 2k curated examples to 47k
# database entries LOWERED its skill on invented peptides. So the extended file is opt-in and gets
# A/B'd against the default before anything is promoted.
USE_EXTENDED_GRAMPA = os.environ.get("AMP_GRAMPA_EXTENDED", "0") == "1"


@functools.lru_cache(maxsize=None)
def external_cohorts():
    """Independent experimental cohorts, tagged by STUDY, for whole-study holdout.

    The 46-peptide AMP-Diffusion set was the only de-novo ground truth here for a long time, and
    every Phase-2 decision was justified against it. That turned out to be far weaker evidence than
    it looked: the physicochemical envelope scores AUC 0.786 on it and 0.344 — below chance — on the
    first independent cohort added here, while aggregation propensity, measured useless on the
    original, transfers at 0.609.

    So a ranking signal must be evaluated by leaving out an ENTIRE STUDY, never a random subset of
    peptides. Random splits within one study share its generator, its assay and its failure modes,
    and therefore certify a signal that does not generalise.

    Each record carries study, assay and panel so those confounds stay visible rather than being
    averaged away. MIC values are NOT normalised across studies — different assays are different
    measurements, and only within-study ranks and hit definitions are comparable.
    """
    out = []
    for name in sorted(os.listdir(DATA)):
        if name.startswith("cohort_") and name.endswith(".json"):
            out.extend(_load(name))
    return tuple(out)


@functools.lru_cache(maxsize=None)
def grampa_full():
    """GRAMPA measurements: sequence, bacterium, strain, log10 MIC.

    Unmodified peptides only by default; set AMP_GRAMPA_EXTENDED=1 to include the amidated ones.
    """
    if USE_EXTENDED_GRAMPA and os.path.exists(os.path.join(DATA, "grampa_extended.json")):
        return _load("grampa_extended.json")
    return _load("grampa_full.json")


# ------------------------------------------------------------------ experimental reality checks

@functools.lru_cache(maxsize=None)
def experimental_ampdiffusion():
    """506 measured MICs (uM) for AMP-Diffusion peptides, de la Fuente lab, competition strains.

    This is the closest thing to ground truth available offline: generated peptides, measured in
    the same laboratory, on strains from the competition's own panel. Used to test whether a
    *selection rule* actually ranks experimentally-potent peptides first.
    """
    return _load("exp_ampdiffusion_mic.json")


@functools.lru_cache(maxsize=None)
def experimental_hydramp():
    """(MIC rows, HC50 rows) from the HydrAMP wet-lab track record."""
    return _load("exp_hydramp_mic.json"), _load("exp_hydramp_hc50.json")


# ------------------------------------------------------------------ compliance

def is_valid(seq):
    return MIN_LEN <= len(seq) <= MAX_LEN and set(seq) <= AASET


def compliance_filter(seqs):
    """Drop everything the competition's automated compliance check would reject.

    Returns (kept, stats). Order is preserved; the first occurrence of a duplicate is kept.

    The exact-match rule is checked against novelty_reference(), not against the shipped validator
    set alone, for the same reason the 80%-identity rule already is: the competition paper enforces
    against MarLys while the validator ships a 39,448-sequence subset, and a library sequence that
    exactly equals a MarLys-only or DRAMP-only AMP is rejected under the paper's reading and passes
    under the validator's. The union is compliant under either. It costs a set lookup against 47,532
    entries instead of 39,448 and removes at most a handful of candidates from a 200,000 pool.
    """
    ref = set(novelty_reference())
    kept, seen = [], set()
    stats = {"n_in": len(seqs), "bad_alphabet": 0, "bad_length": 0, "duplicate": 0, "known_amp": 0}
    for s in seqs:
        if not set(s) <= AASET:
            stats["bad_alphabet"] += 1
            continue
        if not (MIN_LEN <= len(s) <= MAX_LEN):
            stats["bad_length"] += 1
            continue
        if s in seen:
            stats["duplicate"] += 1
            continue
        if s in ref:
            stats["known_amp"] += 1
            continue
        seen.add(s)
        kept.append(s)
    stats["n_out"] = len(kept)
    stats["compliance_rate"] = len(kept) / max(len(seqs), 1)
    return kept, stats
