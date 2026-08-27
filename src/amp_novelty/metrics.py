"""The local referee's metric panel for amp_design. FIXED FILE — do not edit.

It reconstructs, from public information, the Phase-1 evaluation the organizers describe
(AMP Challenge competition paper, §1.5) and adds a Phase-2 proxy for the wet-lab categories:

  Phase 1 (the 50,000-sequence library)
    surrogate activity     AMP-classifier probability, predicted-MIC hit rate
    sequence-level         uniqueness, internal diversity, novelty vs known AMPs, cluster coverage
    distributional         FBD, MMD, precision, recall in an embedding space vs known AMPs
    property distribution  conformity of charge / hydrophobic moment / length / GRAVY, synthesizability

  Phase 2 (the top-100)
    success rate at MIC <= 16 uM, MIC50, gram+/gram-/MDR breakdown, safety window HC50/MIC50

**The aggregation weights are held out by the organizers.** So this file does not guess one
weight vector: every raw metric is anchored on a known-good/known-bad pair of libraries (potent
measured AMPs vs UniProt fragments — the same discrimination the organizers say the score was
tuned for), and the aggregate is reported both as the mean over random weightings and as the
10th-percentile worst case over them. Climbing that pair means climbing under *any* plausible
weighting, which is the only honest way to optimize against a hidden aggregator.

Anchor values are cached per instrument side in cache/anchors_<side>_v<N>.json.
"""

import glob
import json
import math
import os
import random

import numpy as np
from rapidfuzz import fuzz, process

import benchmark
import oracles
from conformity_exact import conformity_kde

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = oracles.CACHE          # follow the oracle caches onto scratch, off the shared quota
os.makedirs(CACHE, exist_ok=True)

VERSION = 18  # bumped: conform_kde is DIRECT — anchoring a 0.060 separation gave it 16.7x weight
# v17 exists because changing mmd_rbf -> mmd_seqme WITHOUT bumping the version would have been the
# stale-artifact trap in its purest form: the raw values move from ~0.02 to ~20, while the cached
# anchors_*_v16.json still hold good/bad poles on the 0.02 scale. Every scaled MMD would clip to
# the ceiling or the floor, silently corrupting the distributional family, and the results file
# would stamp "16" on rows that are not comparable with each other.
# Sample size for every distribution-level metric. Raised from 1,500 after the referee turned out
# unable to resolve the comparisons it was being used for: the Phase-1 composite's spread across
# subsamples was +/-0.037, and the paired difference between the two leading generators was
# +0.0102 with a standard error of 0.0106 — |t| = 0.96, i.e. no answer at all, on a comparison that
# had already been reported as a win.
#
# Subsample noise falls as 1/sqrt(n), so 1,500 -> 5,000 cuts it by 1.83x. Measured cost, warmed up:
# 2.50s -> 5.72s per scoring pass, against build times of tens of minutes. Three seconds to make the
# referee able to answer the question is not a trade-off, it is an oversight being corrected.
SUBSAMPLE = 5000
NOVELTY_SUBSAMPLE = 1200  # sample used for the expensive identity-vs-reference sweep

# The five descriptors the organisers' ConformityScore is fed, in the order the firewall uses.
# Note they are NOT standardised before the KDE -- that is seqme's choice, and it means length
# (sd ~10) dominates the joint density over gravy (sd ~0.5). Reproducing that is the point.
_KDE_PROPS = ("charge", "hmoment", "length", "gravy", "cationic")
_KDE_COLS = [oracles.DESCRIPTORS.index(k) for k in _KDE_PROPS]
IDENTITY_LIMIT = 80.0     # the competition's 80% identity ceiling (Levenshtein ratio, in %)


# --------------------------------------------------------------------------- helpers

def subsample(seqs, n, seed=0):
    seqs = list(seqs)
    if len(seqs) <= n:
        return seqs
    rng = random.Random(seed)
    return rng.sample(seqs, n)


def max_identity_to(seqs, reference, workers=-1):
    """For each seq, the highest Levenshtein ratio (0-100) against any reference sequence."""
    M = process.cdist(list(seqs), list(reference), scorer=fuzz.ratio, workers=workers,
                      dtype=np.uint8)
    return M.max(axis=1).astype(np.float32)


def mean_pairwise_distance(seqs, workers=-1):
    """Mean normalised Levenshtein distance inside a set (1 - ratio), diagonal excluded."""
    seqs = list(seqs)
    M = process.cdist(seqs, seqs, scorer=fuzz.ratio, workers=workers, dtype=np.uint8).astype(np.float32)
    n = len(seqs)
    np.fill_diagonal(M, 0.0)
    return float(1.0 - (M.sum() / (n * (n - 1))) / 100.0)


def cluster_coverage(seqs, threshold=60.0, workers=-1):
    """Fraction of a sample that survives greedy clustering at `threshold`% identity.

    A library that is 50,000 variations on one motif has low coverage; a library spread over
    many distinct sequence families has high coverage.
    """
    seqs = list(seqs)
    M = process.cdist(seqs, seqs, scorer=fuzz.ratio, workers=workers, dtype=np.uint8)
    n = len(seqs)
    taken = np.zeros(n, dtype=bool)
    clusters = 0
    for i in range(n):
        if taken[i]:
            continue
        clusters += 1
        taken |= (M[i] >= threshold)
        taken[i] = True
    return clusters / max(n, 1)


def frechet_distance(X, Y):
    """Fréchet (Wasserstein-2 between Gaussians) distance — FBD when X, Y are embeddings."""
    import scipy.linalg
    mx, my = X.mean(0), Y.mean(0)
    cx = np.cov(X, rowvar=False)
    cy = np.cov(Y, rowvar=False)
    diff = mx - my
    covmean, _ = scipy.linalg.sqrtm(cx.dot(cy), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(cx) + np.trace(cy) - 2 * np.trace(covmean))


def mmd_rbf(X, Y, gamma=None):
    """Unbiased MMD^2 with an RBF kernel (median heuristic bandwidth)."""
    Z = np.concatenate([X, Y], 0)
    if gamma is None:
        sub = Z[np.random.RandomState(0).choice(len(Z), min(500, len(Z)), replace=False)]
        d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
        med = np.median(d2[d2 > 0]) if (d2 > 0).any() else 1.0
        gamma = 1.0 / max(med, 1e-9)

    def k(A, B):
        d2 = ((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A.dot(B.T))
        return np.exp(-gamma * np.maximum(d2, 0))

    n, m = len(X), len(Y)
    kxx, kyy, kxy = k(X, X), k(Y, Y), k(X, Y)
    np.fill_diagonal(kxx, 0)
    np.fill_diagonal(kyy, 0)
    return float(kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2 * kxy.mean())


def precision_recall(X_gen, X_ref, k=5):
    """Improved precision/recall (Kynkäänniemi et al.): manifold membership via kNN radii."""
    def radii(A):
        d = np.sqrt(np.maximum(((A ** 2).sum(1)[:, None] + (A ** 2).sum(1)[None, :]
                                - 2 * A.dot(A.T)), 0))
        d.sort(axis=1)
        return d[:, k]

    def frac_in(B, A, rA):
        d = np.sqrt(np.maximum(((B ** 2).sum(1)[:, None] + (A ** 2).sum(1)[None, :]
                                - 2 * B.dot(A.T)), 0))
        return float((d <= rA[None, :]).any(axis=1).mean())

    rg, rr = radii(X_gen), radii(X_ref)
    return frac_in(X_gen, X_ref, rr), frac_in(X_ref, X_gen, rg)


def wasserstein1d(a, b):
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    q = np.linspace(0, 1, 256)
    return float(np.abs(np.quantile(a, q) - np.quantile(b, q)).mean())


SYNTH_MAX_RUN = 4          # no more than 4 identical residues in a row
SYNTH_MAX_CYS = 2
SYNTH_MAX_HYDROPHOBIC_RUN = 6


def synthesizable(seq):
    """Empirical solid-phase-synthesis friendliness rules.

    Long homopolymer runs, poly-hydrophobic stretches, several cysteines (uncontrolled
    disulfides), and heavy beta-branch/aromatic loading are the standard causes of failed or
    aggregating peptide syntheses. Matches the spirit of HydrAMP's published filters.
    """
    run, prev = 1, ""
    for a in seq:
        run = run + 1 if a == prev else 1
        if run > SYNTH_MAX_RUN:
            return False
        prev = a
    if seq.count("C") > SYNTH_MAX_CYS:
        return False
    hrun = 0
    for a in seq:
        hrun = hrun + 1 if a in "AVLIMFWY" else 0
        if hrun > SYNTH_MAX_HYDROPHOBIC_RUN:
            return False
    n = len(seq)
    if (seq.count("Q") + seq.count("N")) / n > 0.35:
        return False
    if seq.count("P") / n > 0.25:
        return False
    return True


# --------------------------------------------------------------------------- instruments

class Instruments:
    """One side of the referee: an embedder, a reference set, and a potency oracle.

    eval side   ESM-2 embeddings and the eval half of the reference AMPs for the distributional
                and property families — but the ACTIVITY family is scored with the surrogates
                trained on the *val* half, which solution.py never touches.

                This was a defect until run 7 exposed it: scoring activity with the same models
                the generator optimises is not a measurement, it is a restatement of the
                objective, and it stayed pinned near its ceiling (1.03-1.13) while the external
                oracle's view of the same libraries moved by a factor of 25.
    val  side   a k-mer embedding (no neural net), the disjoint val half of the reference AMPs,
                and APEX-pathogen — an external model from the laboratory that will run the
                assays, which the design loop never gets to see.
    """

    ampredictor = None

    def __init__(self, side="eval"):
        self.side = side
        if side == "eval":
            self.embedder = oracles.ESM2Embedder("150M")
            self.reference = list(benchmark.reference_amps("eval"))
            # held out from the design loop on purpose — see the class docstring
            self.surrogates = oracles.SurrogateBundle("val")
            self.naturalness = oracles.ESM2Naturalness("150M")
            self.apex = None
        else:
            self.embedder = oracles.KmerEmbedder(2)
            self.reference = list(benchmark.reference_amps("val"))
            self.surrogates = oracles.SurrogateBundle("val")
            self.naturalness = oracles.MarkovNaturalness(benchmark.reference_amps("val"))
            self.apex = oracles.ApexOracle()
            # A second external oracle, and one the competition names for its own Phase-1
            # activity family. APEX alone can flatter a library that happens to suit it: on APEX
            # our top-100 scored 5x the measured-potent anchor, on AMPredictor it merely matches
            # the HydrAMP baseline. Two independent externals is the cheapest guard against that.
            try:
                from ampredictor_oracle import AMPredictorOracle
                self.ampredictor = AMPredictorOracle()
            except Exception:
                self.ampredictor = None

    # ---- potency, expressed the same way on both sides -------------------------------------
    def potency(self, seqs):
        """→ dict with per-peptide arrays: p_amp, log10_mic, and (val side) a strain matrix."""
        out = {}
        out["p_amp"] = self.surrogates.amp_probability(seqs)
        out["log10_mic"] = self.surrogates.log10_mic(seqs)
        if self.apex is not None:
            mic = self.apex(seqs)                      # (n, 11) uM
            out["apex_mic"] = mic
            out["apex_log10_mic50"] = np.log10(np.median(mic, axis=1))
        return out


# --------------------------------------------------------------------------- phase 1

def phase1_raw(library, inst, seed=0):
    """Raw (unanchored) Phase-1 metrics for a library."""
    lib = list(library)
    n = len(lib)
    sub = subsample(lib, SUBSAMPLE, seed)
    nov_sub = subsample(lib, NOVELTY_SUBSAMPLE, seed + 1)
    ref_sub = subsample(inst.reference, SUBSAMPLE, seed + 2)

    m = {}
    m["uniqueness"] = len(set(lib)) / max(n, 1)

    pot = inst.potency(sub)
    m["amp_prob"] = float(np.mean(pot["p_amp"]))
    m["mic_hit_rate"] = float(np.mean(pot["log10_mic"] <= math.log10(benchmark.POTENCY_THRESHOLD_UM)))

    m["diversity"] = mean_pairwise_distance(sub)
    m["cluster_coverage"] = cluster_coverage(sub)
    ident = max_identity_to(nov_sub, benchmark.novelty_reference())
    m["novelty_frac"] = float(np.mean(ident <= IDENTITY_LIMIT))
    m["novelty_gap"] = float(np.mean(100.0 - ident) / 100.0)

    Xg = inst.embedder(sub)
    Xr = inst.embedder(ref_sub)
    # FBD is computed on the FULL library, not the subsample. It is the one metric here whose cost
    # does not grow with n in any way that matters — the work is a 640x640 covariance and one matrix
    # square root of the same size — and it is the one with a severe small-sample bias. On identical
    # synthetic distributions it reads 42.9 at n=5,000 and 5.7 at n=50,000, and on real reference
    # sets it falls from 0.540 at n=750 to 0.212 at n=12,000 without having converged.
    #
    # That bias is not a constant offset: its size depends on the distribution's own spread, so two
    # libraries can swap ranks purely by changing n. They did — raising the subsample from 1,500 to
    # 5,000 reversed a generator comparison at |t| = 7.46. Measured cost of using everything:
    # 3.17s against 2.25s. Subsampling this metric was buying a reversal for nine tenths of a second.
    #
    # MMD and precision/recall are genuinely quadratic and stay on the subsample, which is safe:
    # both were measured flat in n over the same range where FBD moved by a factor of 2.5.
    Xg_full = inst.embedder(list(library))
    Xr_full = inst.embedder(list(inst.reference))
    m["fbd"] = frechet_distance(Xg_full, Xr_full)
    m["mmd"] = mmd_seqme(Xg, Xr)   # exact seqme, not our median-bandwidth variant
    p, r = precision_recall(Xg, Xr)
    m["precision"], m["recall"] = p, r

    D = oracles.descriptor_matrix(sub)
    Dr = oracles.descriptor_matrix(ref_sub)
    # conform_kde IS seqme.ConformityScore -- verified equal to 1e-12 over 9 (n_splits,
    # bandwidth) settings by conformity_exact.py --verify. Until v16 this family was five
    # separate 1-D Wasserstein distances, which the referee firewall caught ranking libraries
    # differently from the organisers (Spearman 0.814, top-decile overlap 0.58 over 120
    # libraries, against 0.95+ for every other metric).
    #
    # The two are not a tighter and a looser version of one idea. They have OPPOSITE gradients
    # over most of the range: collapse a library onto the reference mode and the Wasserstein
    # form falls (-0.05 -> -0.64) while the true score RISES (0.52 -> 0.83), because every point
    # is then in the density core and beats nearly every held-out reference point. We spent
    # three experiments' worth of property-family conclusions on a metric pointing the wrong way.
    #
    # The 1-D terms stay recorded as diagnostics -- they are how you SEE which descriptor is
    # off -- but they are out of the scored family.
    m["conform_kde"] = conformity_kde(D[:, _KDE_COLS], Dr[:, _KDE_COLS])
    for prop in ("charge", "hmoment", "length", "gravy", "cationic"):
        j = oracles.DESCRIPTORS.index(prop)
        scale = max(np.std(Dr[:, j]), 1e-6)
        m[f"conform_{prop}"] = float(-wasserstein1d(D[:, j], Dr[:, j]) / scale)

    if getattr(inst, "ampredictor", None) is not None:
        m["ampredictor"] = float(-np.mean(inst.ampredictor(subsample(sub, 400, seed + 9))))
    m["naturalness"] = float(np.mean(inst.naturalness(sub)))
    m["synth_rate"] = float(np.mean([synthesizable(s) for s in sub]))
    return m


# Metrics whose natural scale is already [0,1] and whose good pole is NOT "looks like a known
# AMP": novelty and diversity are zero *by construction* for peptides that are in the reference
# database, so anchoring them on known AMPs would score a library down for obeying the
# competition's own novelty rule. These pass through on their own scale.
DIRECT_METRICS = {"uniqueness", "diversity", "cluster_coverage", "novelty_frac", "novelty_gap",
                  "synth_rate", "top_novelty_ok", "top_identity_margin", "top_diversity",
                  "top_synth_rate",
                  # v18: conform_kde joins its own family-mate rather than being anchored.
                  #
                  # Anchoring divides by the good/bad separation, and conformity's is 0.060 --
                  # measured-potent AMPs score 0.4358 against UniProt fragments at 0.3758, so the
                  # metric barely separates real AMPs from generic protein junk. Dividing by 0.060
                  # multiplies every conformity movement 16.7x, which does not merely inflate a
                  # number: it makes ONE family dominate the Dirichlet-over-random-weightings
                  # aggregate regardless of which weights are sampled, defeating the purpose of
                  # sampling weights at all.
                  #
                  # Measured on cache-free evaluations of three real libraries:
                  #     anchored   core_1 0.8460 > prop_1 0.7538 > base 0.7198
                  #     direct     prop_1 0.7455 > core_1 0.7301 > base 0.7268
                  # The anchored instrument ranks first the arm that makes FBD, MMD and recall
                  # ALL worse; the direct one agrees with those exact metrics. An instrument that
                  # contradicts its own exactly-verified components is the instrument at fault.
                  #
                  # synth_rate was already here for a related reason: its separation is NEGATIVE
                  # (-0.054, potent AMPs are LESS synthesizable than UniProt fragments by our
                  # rule), so anchoring it would have rewarded lower synthesizability. Conformity
                  # was the last property metric still anchored on a near-degenerate pair.
                  "conform_kde"}

P1_FAMILIES = {
    "activity": ["amp_prob", "mic_hit_rate", "ampredictor"],
    "sequence": ["uniqueness", "diversity", "cluster_coverage", "novelty_frac", "novelty_gap"],
    "distributional": ["fbd", "mmd", "precision", "recall", "naturalness"],
    "property": ["conform_kde", "synth_rate"],
}


# --------------------------------------------------------------------------- phase 2

def phase2_raw(top, inst):
    """Raw Phase-2 (wet-lab proxy) metrics for a top-100 list."""
    top = list(top)
    m = {}
    pot = inst.potency(top)
    hemo = oracles.hemolysis_risk(top)
    # pseudo-HC50 in uM: the assay ceiling scaled by predicted risk (see oracles.hemolysis_risk)
    hc50 = benchmark.HC50_CEILING_UM * (1.0 - hemo) + 4.0

    if "apex_mic" in pot:
        mic = pot["apex_mic"]                                   # (n, 11) uM
        thr = benchmark.POTENCY_THRESHOLD_UM
        m["success_rate"] = float(np.mean(mic <= thr))
        m["success_gram_neg"] = float(np.mean(mic[:, oracles.APEX_GRAM_NEG] <= thr))
        m["success_gram_pos"] = float(np.mean(mic[:, oracles.APEX_GRAM_POS] <= thr))
        m["success_mdr"] = float(np.mean(mic[:, oracles.APEX_MDR] <= thr))
        mic50 = np.median(mic, axis=1)
        m["neg_log_mic50"] = float(-np.mean(np.log10(mic50)))
        # Graded companions to the hit-rates. A threshold at 16 uM is a cliff, and predicted MICs
        # for de-novo peptides sit at 30-500 uM, so the hit-rates sat at 0.002-0.03 and the gate
        # was discriminating on noise. The continuous versions give the search a gradient.
        m["neg_log_mic_gram_neg"] = float(-np.mean(np.log10(np.median(mic[:, oracles.APEX_GRAM_NEG], axis=1))))
        m["neg_log_mic_gram_pos"] = float(-np.mean(np.log10(np.median(mic[:, oracles.APEX_GRAM_POS], axis=1))))
        m["neg_log_mic_mdr"] = float(-np.mean(np.log10(np.median(mic[:, oracles.APEX_MDR], axis=1))))
        m["safety_window"] = float(np.mean(np.log10(hc50 / np.maximum(mic50, 0.5))))
        m["min_strain_potency"] = float(np.mean(np.log10(1000.0 / np.maximum(mic.min(axis=1), 0.5))))
    else:
        # design side: the surrogate pair only speaks E. coli MIC + AMP probability
        lm = pot["log10_mic"]
        m["success_rate"] = float(np.mean(lm <= math.log10(benchmark.POTENCY_THRESHOLD_UM)))
        m["success_gram_neg"] = m["success_rate"]
        m["success_gram_pos"] = float(np.mean(pot["p_amp"]))
        m["success_mdr"] = m["success_rate"]
        m["neg_log_mic50"] = float(-np.mean(lm))
        m["neg_log_mic_gram_neg"] = float(-np.mean(lm))
        m["neg_log_mic_gram_pos"] = float(-np.mean(2.0 - 2.0 * pot["p_amp"]))
        m["neg_log_mic_mdr"] = float(-np.mean(lm))
        m["safety_window"] = float(np.mean(np.log10(hc50) - lm))
        m["min_strain_potency"] = float(np.mean(pot["p_amp"] * (2.0 - lm)))

    if getattr(inst, "ampredictor", None) is not None:
        m["top_ampredictor"] = float(-np.mean(inst.ampredictor(top)))

    ident = max_identity_to(top, benchmark.novelty_reference())
    m["top_novelty_ok"] = float(np.mean(ident <= IDENTITY_LIMIT))
    m["top_identity_margin"] = float(np.mean(IDENTITY_LIMIT - ident) / 100.0)
    m["top_diversity"] = mean_pairwise_distance(top)
    m["top_synth_rate"] = float(np.mean([synthesizable(s) for s in top]))
    # The one Phase-2 quantity measured to transfer to de-novo peptides. Every other instrument in
    # this family was checked against the 46 peptides with real MICs and none of them survives:
    # APEX scores rho -0.246 and AUC 0.344 at telling a peptide that kills something from one that
    # kills nothing — below a coin flip, and its own top-25 of those 46 holds 10 duds where a
    # random draw holds 7.1 — while hemolysis_risk reaches AUC 0.552. The trained HC50 head reaches
    # rho +0.370 and AUC 0.797, and it is the val-side head, so it is held out from the design loop
    # exactly like the rest of this side's instruments.
    m["top_hc50_surrogate"] = float(np.mean(inst.surrogates.log10_hc50(top)))
    return m


P2_FAMILIES = {
    "broad": ["success_rate", "neg_log_mic50"],
    # AMPredictor gets a family of its own rather than a seat in "broad". Five of the six Phase-2
    # families were APEX-derived, so a library fitted to APEX's idiosyncrasies could carry the
    # gate. It can: peptides with MEASURED MIC <= 8 uM score APEX succ@16 = 0.030 while our
    # generated top-100 scores 0.206 — a 7x inversion against reality — and on AMPredictor the
    # ordering flips back (measured actives 0.759, our top-100 0.907, lower is better). With one
    # family each, no single external oracle can carry the verdict.
    "second_oracle": ["top_ampredictor"],
    "gram_pos": ["success_gram_pos", "neg_log_mic_gram_pos"],
    "gram_neg": ["success_gram_neg", "neg_log_mic_gram_neg"],
    "mdr": ["success_mdr", "neg_log_mic_mdr"],
    "selectivity": ["safety_window"],
    # A family for the only instrument here with demonstrated de-novo skill. The existing guard —
    # giving AMPredictor its own family so no single external oracle carries the verdict — was the
    # right idea and insufficient: five of seven families remained APEX-derived, and APEX was
    # subsequently measured ANTI-correlated with real MICs on invented peptides. This adds a family
    # scored by an instrument that was measured to work rather than assumed to, which is the same
    # argument applied one step further.
    "measured_transfer": ["top_hc50_surrogate"],
    # Eligibility is a GATE. The competition applies a hard <=80% identity threshold to the
    # top-100 and scores nothing else about it — no identity margin, no mutual diversity. Paying
    # a continuous reward for extra margin invented an objective that pushes candidates away from
    # measured actives for no competition benefit, and it sat at 0.70 in every run, quietly
    # taxing a seventh of Phase 2. Now: full marks when every candidate is compliant and
    # synthesizable, a cliff when they are not.
    "eligibility": ["top_novelty_ok", "top_synth_rate"],
}


def mmd_seqme(Xg, Xr, sigma=10.0, scale=1000.0, estimate="biased"):
    """EXACT seqme.MMD. Biased estimator, fixed sigma=10, scale=1000 -- not our median-bandwidth
    unbiased version, which the referee firewall caught ranking libraries differently (Spearman
    0.952, top-decile overlap 0.67 over 120 libraries: two of twelve top libraries misplaced).

    That is not good enough to separate closely-matched submissions, and after a wrong-signed
    conformity proxy cost a quarter of Phase 1 there is no case for keeping an approximate metric
    when the exact one is thirty lines. Transcribed from seqme/metrics/mmd.py: gaussian_kernels
    with gamma = 1/(2*sigma^2), then k_xx.mean() + k_yy.mean() - 2*k_xy.mean(), times scale.
    """
    Xg = np.ascontiguousarray(Xg, dtype=np.float64)
    Xr = np.ascontiguousarray(Xr, dtype=np.float64)
    gamma = 1.0 / (2.0 * sigma ** 2)
    g_sq = (Xg * Xg).sum(1)
    r_sq = (Xr * Xr).sum(1)
    d_gg = g_sq[:, None] + g_sq[None, :] - 2.0 * (Xg @ Xg.T)
    d_rr = r_sq[:, None] + r_sq[None, :] - 2.0 * (Xr @ Xr.T)
    d_gr = g_sq[:, None] + r_sq[None, :] - 2.0 * (Xg @ Xr.T)
    k_gg, k_rr, k_gr = np.exp(-gamma * d_gg), np.exp(-gamma * d_rr), np.exp(-gamma * d_gr)
    if estimate == "biased":
        a, b = k_gg.mean(), k_rr.mean()
    else:
        m, n = len(Xg), len(Xr)
        a = (k_gg.sum() - np.trace(k_gg)) / (m * (m - 1))
        b = (k_rr.sum() - np.trace(k_rr)) / (n * (n - 1))
    return float(scale * (a + b - 2.0 * k_gr.mean()))


def load_pool(path=None, n_lib=50000, seed=0, params=None, min_n=50000, quiet=False):
    """Load a candidate pool, and never do it silently.

    Three versions of this preceded it. Two were guesses that returned a plausible pool instead of
    an error, and one of them corrupted a result:

      * newest-by-mtime. Any small run writes a fresh pool and becomes "newest". On 2026-08-25 a
        descriptor probe loaded 7,425 sequences instead of 189,421 and concluded a selection "does
        not transfer" -- it had selected every sequence available, so there was nothing to
        transfer. Caught only because the argmax and random-draw arms came out bit-identical.
      * largest-by-size. Also a guess: it returns a 427,172-sequence pool from a different
        generation configuration than the arms under analysis used.
      * by cache key. Correct in principle -- solution._pool_cache_key hashes exactly the
        generation parameters -- but the key mixes in `rng_seed = rng.randrange(1 << 30)`, drawn
        inside build(), so it cannot be reproduced from (params, n_lib, seed) out here.

    So: `path` is explicit and exact and is what analysis code should pass. Without it this falls
    back to the largest pool at or above `min_n`, PRINTS what it chose, and raises rather than
    quietly under-serving a caller that needs 50,000 sequences out of 7,425.
    """
    if path is None:
        cands = []
        for f in glob.glob(os.path.join(CACHE, "pools", "*.json")):
            if "scores_" in f:
                continue
            try:
                cands.append((len(json.load(open(f))["pool"]), f))
            except Exception:
                continue
        if not cands:
            raise FileNotFoundError(f"no readable pool files under {os.path.join(CACHE, 'pools')}")
        cands.sort()
        n, path = cands[-1]
        if n < min_n:
            raise ValueError(f"largest pool has {n} sequences, need >= {min_n}. "
                             f"Generate a pool at that scale before running this analysis.")
        if not quiet:
            print(f"[load_pool] no path given -- using largest of {len(cands)}: "
                  f"{os.path.basename(path)} ({n} sequences). Pass path= to pin it.", flush=True)
    pool = json.load(open(path))["pool"]
    if len(pool) < min_n:
        raise ValueError(f"pool {os.path.basename(path)} has {len(pool)} sequences, "
                         f"need >= {min_n}")
    if not quiet:
        print(f"[load_pool] {os.path.basename(path)}: {len(pool)} sequences", flush=True)
    return pool, path


# --------------------------------------------------------------------------- anchoring

def _anchor_path(side):
    return os.path.join(CACHE, f"anchors_{side}_v{VERSION}.json")


def anchors(inst, force=False):
    """Raw metrics of the good pole (measured-potent AMPs) and the bad pole (UniProt fragments).

    Every raw metric is mapped to 0 at the bad pole and 1 at the good pole, so the composite is
    interpretable without knowing the organizers' weights: 1.0 means "as AMP-like as peptides
    with measured MIC <= 8 uM", 0.0 means "indistinguishable from generic protein fragments".
    """
    path = _anchor_path(inst.side)
    if os.path.exists(path) and not force:
        with open(path) as f:
            return json.load(f)

    good_lib = list(benchmark.anchor_potent())
    bad_lib = list(benchmark.anchor_uniprot())

    def repeat(fn, reps, *args):
        runs = [fn(*args, seed=100 + 17 * r) for r in range(reps)]
        keys = runs[0].keys()
        return ({k: float(np.mean([r[k] for r in runs])) for k in keys},
                {k: float(np.std([r[k] for r in runs])) for k in keys})

    g1, g1s = repeat(phase1_raw, 3, good_lib, inst)
    b1, b1s = repeat(phase1_raw, 3, bad_lib, inst)
    p2g = [phase2_raw(subsample(good_lib, benchmark.TOP_K, seed=3 + r), inst) for r in range(3)]
    p2b = [phase2_raw(subsample(bad_lib, benchmark.TOP_K, seed=13 + r), inst) for r in range(3)]
    g2 = {k: float(np.mean([r[k] for r in p2g])) for k in p2g[0]}
    g2s = {k: float(np.std([r[k] for r in p2g])) for k in p2g[0]}
    b2 = {k: float(np.mean([r[k] for r in p2b])) for k in p2b[0]}
    b2s = {k: float(np.std([r[k] for r in p2b])) for k in p2b[0]}

    a = {"p1_good": g1, "p1_bad": b1, "p1_good_sd": g1s, "p1_bad_sd": b1s,
         "p2_good": g2, "p2_bad": b2, "p2_good_sd": g2s, "p2_bad_sd": b2s}
    with open(path, "w") as f:
        json.dump(a, f, indent=1)
    return a


def _scale(raw, good, bad, good_sd=None, bad_sd=None, noise_k=3.0):
    """Map each metric to 0 = generic peptide fragment, 1 = measured-potent AMP.

    Two guards learned from the first calibration run:
      * DIRECT_METRICS keep their own [0,1] scale — anchoring novelty/diversity on known AMPs
        punishes a library for satisfying the competition's novelty rule.
      * an anchored metric whose good/bad separation is smaller than `noise_k` times its own
        resampling noise is DROPPED. Such a metric contributes pure sampling noise to the
        composite, and a search will happily climb noise.
    """
    out = {}
    for k, v in raw.items():
        if k in DIRECT_METRICS:
            out[k] = float(np.clip(v, -0.5, 3.0))
            continue
        g, b = good.get(k), bad.get(k)
        if g is None or b is None:
            continue
        sep = abs(g - b)
        if sep < 1e-9:
            continue
        if good_sd is not None and bad_sd is not None:
            noise = max(good_sd.get(k, 0.0), bad_sd.get(k, 0.0))
            if sep < noise_k * noise:
                continue
        # ceiling raised from 1.5: run 9's Phase-2 families all sat exactly at the old clip,
        # so the held-out gate had stopped resolving anything above the known-potent anchor —
        # a gate that cannot see improvement is not a gate.
        out[k] = float(np.clip((v - b) / (g - b), -0.5, 3.0))
    return out


def robust_aggregate(scaled, families, n_draws=512, seed=0):
    """Aggregate under a *hidden* weight vector.

    Returns (mean, p10) over Dirichlet-sampled weightings across families and, within each
    family, across its metrics. The mean is the expected ranking signal; the 10th percentile is
    the adversarial one — a library that only wins under a lucky weighting has a low p10.
    """
    fam_vals = {}
    for fam, keys in families.items():
        vals = [scaled[k] for k in keys if k in scaled]
        if vals:
            fam_vals[fam] = np.array(vals, dtype=np.float64)
    if not fam_vals:
        return 0.0, 0.0, {}
    names = sorted(fam_vals)
    rng = np.random.RandomState(seed)
    draws = np.zeros(n_draws)
    for d in range(n_draws):
        w = rng.dirichlet(np.ones(len(names)))
        tot = 0.0
        for wi, fam in zip(w, names):
            v = fam_vals[fam]
            wv = rng.dirichlet(np.ones(len(v)))
            tot += wi * float(wv.dot(v))
        draws[d] = tot
    per_family = {f: float(fam_vals[f].mean()) for f in names}
    return float(draws.mean()), float(np.percentile(draws, 10)), per_family


def evaluate_library(library, top, inst, seed=0):
    """Full anchored report for one (library, top-100) pair on one instrument side."""
    a = anchors(inst)
    p1 = phase1_raw(library, inst, seed=seed)
    p2 = phase2_raw(top, inst)
    s1 = _scale(p1, a["p1_good"], a["p1_bad"], a.get("p1_good_sd"), a.get("p1_bad_sd"))
    s2 = _scale(p2, a["p2_good"], a["p2_bad"], a.get("p2_good_sd"), a.get("p2_bad_sd"))
    m1, q1, f1 = robust_aggregate(s1, P1_FAMILIES, seed=seed)
    m2, q2, f2 = robust_aggregate(s2, P2_FAMILIES, seed=seed + 1)
    phase1 = 0.5 * m1 + 0.5 * q1
    phase2 = 0.5 * m2 + 0.5 * q2
    return {
        "phase1": phase1, "phase2": phase2,
        "phase1_mean": m1, "phase1_p10": q1, "phase2_mean": m2, "phase2_p10": q2,
        "families_p1": f1, "families_p2": f2,
        "raw_p1": p1, "raw_p2": p2, "scaled_p1": s1, "scaled_p2": s2,
    }
