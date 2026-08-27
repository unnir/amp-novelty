"""Optimise the LIBRARY, not the peptides. AGENT-EDITABLE (an assembly lineage).

Every assembly method so far scores candidates independently and then picks a subset — quota
selection, greedy diversity, top-k. But three of Phase 1's four metric families are properties of
the *set*: Fréchet distance to the reference distribution, MMD, manifold precision/recall,
cluster coverage, property conformity. A per-candidate score cannot see them.

So make the set the search object. Start from the quota assembly, then run swap moves —
replace a library member with a pool candidate — accepting a swap when the set-level objective
improves. The objective is the thing the referee actually measures:

    J(S) = -w_fbd * FBD(S, reference) - w_mmd * MMD(S, reference)
           - w_prop * property-conformity distance
           + w_score * mean design score

The trick that makes this cheap: FBD depends on S only through its embedding mean and covariance,
and a swap is a rank-one update of both. So a swap costs O(d^2) instead of O(|S| d^2), and a
50,000-member library can be optimised with tens of thousands of proposals in seconds per sweep.

Cross-domain provenance: this is coreset construction / kernel herding, the same move used to
compress a dataset into a small set that matches its moments — applied here to *build* a library
that matches the distribution of peptides that work.
"""

import os

import numpy as np


class MomentSet:
    """Running mean and covariance of a set of embeddings, with O(d^2) add/remove."""

    def __init__(self, X):
        self.n = len(X)
        self.sum = X.sum(axis=0)
        self.outer = X.T @ X

    def add(self, x):
        self.n += 1
        self.sum += x
        self.outer += np.outer(x, x)

    def remove(self, x):
        self.n -= 1
        self.sum -= x
        self.outer -= np.outer(x, x)

    def mean_cov(self):
        mu = self.sum / max(self.n, 1)
        cov = self.outer / max(self.n - 1, 1) - (self.n / max(self.n - 1, 1)) * np.outer(mu, mu)
        return mu, cov


def frechet(mu1, cov1, mu2, cov2, eps=1e-6):
    import scipy.linalg
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(cov1.dot(cov2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(cov1) + np.trace(cov2) - 2 * np.trace(covmean))


def moment_gap(ms, ref_mu, ref_var):
    """Cheap stand-in for FBD: squared mean gap plus squared gap in per-dimension variance.

    O(d) per proposal against O(d^3) for a matrix square root, which buys ~1000x more swap
    proposals for the same wall-clock. It drops the cross-covariance terms FBD keeps, but under a
    random projection those are small, and the ranking of swaps is what the search consumes.
    """
    mu = ms.sum / max(ms.n, 1)
    var = np.diag(ms.outer) / max(ms.n - 1, 1) - (ms.n / max(ms.n - 1, 1)) * mu * mu
    d = mu - ref_mu
    dv = var - ref_var
    return float(d.dot(d) + dv.dot(dv))


def cluster_entropy(counts):
    """Shannon entropy of the library's cluster occupancy — the coverage term."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


def build_recall_cover(pool_emb, ref_emb, k=5, cap=64, device=None, chunk=2048,
                       radius_sample=20000, seed=0):
    """Which reference points each pool candidate would cover, for an explicit recall term.

    seqme's recall — the metric the competition scores — asks what fraction of REFERENCE points fall
    inside the k-NN ball of some generated point. That is a statement about SUPPORT. The swap
    objective optimises moments, which is a statement about the first two MOMENTS, and the two come
    apart exactly where it matters: a set can reproduce a reference's mean and covariance while
    sitting entirely inside one of its modes and covering none of the others. FBD is a moment
    metric, so moment-matching serves it directly; recall it serves only by accident, and recall is
    our weakest of the four (0.553 against precision 0.760).

    So this precomputes, for every pool candidate i, the reference points lying within i's own k-NN
    radius. The radius is measured against the POOL rather than against the eventual library, which
    is the one approximation here: the true radius depends on what gets selected, and selection is
    what we are deciding. Pool radii are a stable lower-ish bound and do not shift under the swap.

    Returns (indptr, indices) in CSR form, capped at `cap` reference points per candidate so a
    candidate sitting in a dense region cannot dominate the incremental bookkeeping.

    MUST be given the SAME embedding the metric is computed in. The first version of this was passed
    the 64-dimensional random projection the swap optimises moments in, on the reasoning that the
    term should live in the geometry the swap works in. That was the wrong geometry: recall is scored
    on the full embedding, and while a random projection approximately preserves pairwise distances,
    k-NN BALL MEMBERSHIP is far more sensitive than that. The term optimised coverage in a space the
    metric never looks at, and measured recall accordingly did not move (0.619 -> 0.616 with the
    term active at weight 1.5).

    Cost is dominated by the pool self-kNN, which is quadratic in the pool. `radius_sample` estimates
    each candidate's k-NN radius against a random subset rather than the whole pool: a radius is a
    local statistic, so a 20k reference sample is ample, and it turns a 200k x 200k problem into a
    200k x 20k one. Radii so measured are slightly larger than true pool radii and smaller than radii
    in the eventual 50k library, so coverage remains conservatively estimated.
    """
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    P = torch.as_tensor(np.ascontiguousarray(pool_emb), dtype=torch.float32, device=device)
    R = torch.as_tensor(np.ascontiguousarray(ref_emb), dtype=torch.float32, device=device)
    Pn, Rn = (P * P).sum(1), (R * R).sum(1)

    if radius_sample and len(P) > radius_sample:
        # SEEDED. This was the only unseeded RNG call left in the generation path, and it cost a
        # night: torch.randperm on the global CUDA generator drew a different subsample in every
        # process, so the estimated kNN radii differed, so the coverage structure differed, so the
        # greedy initialisation started somewhere else and the whole 50,000-sequence library came
        # out different. Three builds at the identical seed produced three different libraries and
        # coverage of 3898, 3904 and 3906 out of 4000 -- small enough to read as noise, fatal to a
        # competition rule that requires byte-identical reproduction.
        #
        # A local generator, not torch.manual_seed: seeding the global RNG here would silently
        # shift every other torch draw in the process, which is the same class of bug as the dead
        # code that moved results by 0.052 by consuming from a shared stream.
        # Device-local generator, deliberately, after measuring the alternative.
        #
        # torch RNG is device-specific, so this makes the permutation hardware-dependent. Drawing
        # on CPU and moving the indices fixes THAT -- but not the hardware dependence overall,
        # because build_recall_cover also computes pairwise distances by torch matmul on the
        # device, and CPU and CUDA matmul differ in the last bits, so the k-NN radii diverge
        # anyway. Measured: with a CPU-drawn permutation the GPU library moved (83d5d303 ->
        # 945308b7) while the CPU library did not (3a73d937 both times) -- the gap did not close,
        # it just relocated.
        #
        # Full hardware independence would require running the whole coverage build on CPU, which
        # is far too slow for a 190k-member pool. It is also unnecessary: the organisers' validator
        # runs BOTH generations on its own machine, and reproducibility holds within each hardware
        # class (GPU reproduces across runs; CPU produced 3a73d937 in two independent runs). What
        # hardware changes is WHICH library, not whether it reproduces -- and the submitted FASTA
        # files travel with the repository.
        g = torch.Generator(device=device).manual_seed(int(seed) & 0x7FFFFFFF)
        sel = torch.randperm(len(P), device=device, generator=g)[:radius_sample]
        Q, Qn = P[sel], Pn[sel]
    else:
        Q, Qn = P, Pn
    radii = torch.empty(len(P), device=device)
    for a0 in range(0, len(P), chunk):
        b0 = min(a0 + chunk, len(P))
        d = (Pn[a0:b0, None] + Qn[None, :] - 2.0 * P[a0:b0] @ Q.T).clamp_min_(0).sqrt_()
        radii[a0:b0] = d.topk(k + 1, largest=False).values[:, k]
    del d

    rows, cols = [], []
    for a0 in range(0, len(P), chunk):
        b0 = min(a0 + chunk, len(P))
        d = (Pn[a0:b0, None] + Rn[None, :] - 2.0 * P[a0:b0] @ R.T).clamp_min_(0).sqrt_()
        hit = d <= radii[a0:b0, None]
        n_hit = hit.sum(1)
        over = (n_hit > cap).nonzero(as_tuple=True)[0]
        for i in over.tolist():                      # keep the `cap` nearest, drop the rest
            keep = d[i].masked_fill(~hit[i], float("inf")).topk(cap, largest=False).indices
            hit[i] = False
            hit[i, keep] = True
        r, c = hit.nonzero(as_tuple=True)
        rows.append((r + a0).cpu().numpy())
        cols.append(c.cpu().numpy())
    del d, hit
    rows = np.concatenate(rows) if rows else np.zeros(0, np.int64)
    cols = np.concatenate(cols) if cols else np.zeros(0, np.int64)
    order = np.argsort(rows, kind="stable")
    rows, cols = rows[order], cols[order]
    indptr = np.zeros(len(pool_emb) + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=len(pool_emb)), out=indptr[1:])
    return indptr, cols.astype(np.int32), len(ref_emb)


def greedy_cover_init(recall_cover, k, pool_scores, forbid_mask=None, score_weight=1e-3,
                      preselected=None, subsample_p=None):
    """Pick k candidates by lazy-greedy maximum coverage, then top up by score.

    Maximum coverage is submodular, so greedy is within (1 - 1/e) of optimal and, more to the point
    here, it is a fundamentally better STARTING SET than a quota fill. The swap optimiser is local
    search: it inherits whatever basin its initialisation lands in, and a library initialised by
    cluster quotas has no reason to sit in a basin with good reference coverage — quotas equalise
    occupancy, which is not the same thing as covering points.

    Lazy greedy (CELF): keep a heap of upper bounds on each candidate's marginal gain. A candidate's
    true gain only ever falls as others are chosen, so if the top of the heap still beats the second
    entry after recomputation, it is optimal and no other candidate needs touching. In practice this
    evaluates a few percent of the candidates per pick.

    `score_weight` breaks ties by design score, which matters once coverage saturates: at that point
    every remaining candidate has marginal gain 0 and the choice should fall back to quality.

    `subsample_p` is what makes this objective the metric's objective rather than a proxy for it.
    Boolean coverage -- "is this reference point covered by ANY of the 50,000" -- SATURATES, and
    measurably so: on a real run it reached 3,905 of 4,000 reference points, at which point the
    remaining 49,000-odd picks had marginal gain 0 and fell through to the score tie-break. The term
    had no gradient over 99% of the selection, which is exactly why it measured inert.

    But seqme does not score recall on the library. It scores it on a random SUBSAMPLE of the
    library -- metrics.SUBSAMPLE = 5,000 of the 50,000 -- so a reference point only counts if one of
    those 5,000 covers it. With p = 5,000/50,000 the expected credit for a point covered c times is
    1 - (1 - p)^c, and the objective becomes sum over reference points of that. It is still concave
    in c and therefore still submodular, so lazy greedy keeps its (1 - 1/e) guarantee -- but it
    never saturates, because the c-th coverer is still worth p(1-p)^(c-1).

    The arithmetic agrees with what we measure: boolean coverage 0.976 against a measured recall of
    0.663 implies a typical reference point is covered about ten times, and 1 - 0.9^10 = 0.65.
    Redundancy, not coverage, is what recall is actually asking for.
    """
    import heapq
    indptr, idx, n_ref = recall_cover
    n_pool = len(indptr) - 1
    prob = None if not subsample_p else float(np.clip(subsample_p, 1e-6, 1.0))
    covered = np.zeros(n_ref, dtype=bool)
    counts = np.zeros(n_ref, dtype=np.int32)
    chosen, taken = [], np.zeros(n_pool, dtype=bool)
    # Members the caller has already committed to (the protected elites) are not up for selection,
    # but what they cover must count, or greedy spends its budget re-covering them.
    for i in (preselected or []):
        if 0 <= i < n_pool and not taken[i]:
            taken[i] = True
            chosen.append(int(i))
            sl = idx[indptr[i]:indptr[i + 1]]
            if len(sl):
                covered[sl] = True
                np.add.at(counts, sl, 1)
    smax = float(np.max(pool_scores)) - float(np.min(pool_scores)) or 1.0

    def gain(i):
        sl = idx[indptr[i]:indptr[i + 1]]
        if prob is None:
            new = int((~covered[sl]).sum()) if len(sl) else 0
        else:
            # marginal expected credit: p * (1-p)^c summed over the points this candidate reaches
            new = float(prob * np.power(1.0 - prob, counts[sl]).sum()) if len(sl) else 0.0
        return new + score_weight * (float(pool_scores[i]) / smax)

    heap = []
    for i in range(n_pool):
        if forbid_mask is not None and forbid_mask[i]:
            continue
        heapq.heappush(heap, (-gain(i), 0, i))

    it = 0
    while len(chosen) < k and heap:
        neg_g, stamp, i = heapq.heappop(heap)
        if taken[i]:
            continue
        if stamp == it or not heap:
            g = -neg_g
        else:
            g = gain(i)                       # stale bound: recompute and re-insert if beaten
            if heap and g < -heap[0][0]:
                heapq.heappush(heap, (-g, it, i))
                continue
        chosen.append(i)
        taken[i] = True
        sl = idx[indptr[i]:indptr[i + 1]]
        if len(sl):
            covered[sl] = True
            np.add.at(counts, sl, 1)
        it += 1

    if len(chosen) < k:                        # coverage exhausted — fill the rest by score
        rest = [i for i in np.argsort(-np.asarray(pool_scores))
                if not taken[i] and (forbid_mask is None or not forbid_mask[i])]
        chosen.extend(rest[:k - len(chosen)])
    if prob is None:
        return chosen[:k], int(covered.sum())
    # Report the quantity the metric will actually see, not the saturating one.
    expected = float((1.0 - np.power(1.0 - prob, counts)).sum())
    return chosen[:k], expected


def optimise_library(library_idx, pool_emb, pool_scores, ref_mu, ref_cov, rng,
                     iters=20000, w_score=1.0, w_fbd=1.0, batch=64, temperature=0.0,
                     forbid=None, on_accept=None, metric="moment", labels=None, w_cover=0.0,
                     n_clusters=0, recall_cover=None, w_recall=0.0, recall_subsample_p=None,
                     prop=None, ref_prop=None, w_prop=0.0, core_lp=None, w_core=0.0, conf=None, w_conf=0.0,
                     conf_low=0.0, conf_high=1.0):
    """Swap-optimise a library (given as indices into the pool) against the reference moments.

    Returns the improved index set. `forbid(i, j)` may veto a swap (e.g. identity constraints);
    `on_accept(i_out, i_in)` is called after each accepted swap so the caller can maintain its own
    bookkeeping (cluster counts, taken sets).
    """
    cur = list(library_idx)
    in_lib = np.zeros(len(pool_emb), dtype=bool)
    in_lib[cur] = True
    ms = MomentSet(pool_emb[cur])
    ref_var = np.diag(ref_cov)
    if metric == "moment":
        fbd = moment_gap(ms, ref_mu, ref_var)
    else:
        mu, cov = ms.mean_cov()
        fbd = frechet(mu, cov, ref_mu, ref_cov)
    score = float(pool_scores[cur].mean())
    # coverage term: how many distinct behavioural clusters the library occupies, and how evenly.
    # Phase 1 scores clustering-based coverage and internal diversity directly, and neither is
    # visible to a distributional-match objective — a library can match the reference's moments
    # while sitting in half as many clusters.
    use_cover = w_cover > 0 and labels is not None and n_clusters > 0
    if use_cover:
        counts = np.bincount(np.asarray(labels)[cur], minlength=n_clusters).astype(np.float64)
        cover = cluster_entropy(counts)
        cover0 = max(cover, 1e-9)
    else:
        counts = None
        cover = 0.0
        cover0 = 1.0          # not 0.0: the objective divides by it even when w_cover is 0
    # Normalise both terms to their own scale. Raw units differ by three orders of magnitude
    # (moment gaps are ~1e-3, score deltas ~1), so an un-normalised weight silently turns the
    # objective into "maximise mean score" and the distributional term never binds.
    # Explicit recall term: how many REFERENCE points are covered by at least one selected member.
    # Maintained as a per-reference hit counter, so a swap only changes the covered total where a
    # count crosses 0<->1 — the same rank-one trick the moment set uses, and just as cheap.
    # `recall_subsample_p` switches this from "covered at least once" to the quantity seqme actually
    # scores. Recall is measured on a random metrics.SUBSAMPLE of the library, not on the library, so
    # a reference point covered once by a 50,000-member set is only credited with probability p. The
    # boolean version saturates -- 3,905 of 4,000 on a real run -- and a saturated term has no
    # gradient, which is why this measured inert. See greedy_cover_init for the derivation.
    use_recall = w_recall > 0 and recall_cover is not None
    rprob = None if not recall_subsample_p else float(np.clip(recall_subsample_p, 1e-6, 1.0))
    # q^c for every multiplicity that can occur, so the inner loop is a gather rather than a pow.
    # _recall_delta runs once per proposal and there are hundreds of thousands of proposals; np.power
    # on an integer exponent array there is pure overhead when c is small and bounded.
    QPOW = None
    if use_recall and rprob is not None:
        qmax = int(len(library_idx)) + 2
        QPOW = np.power(1.0 - rprob, np.arange(qmax, dtype=np.float64))
    if use_recall:
        cov_indptr, cov_idx, n_ref_pts = recall_cover
        hit_count = np.zeros(n_ref_pts, dtype=np.int32)
        for i in cur:
            np.add.at(hit_count, cov_idx[cov_indptr[i]:cov_indptr[i + 1]], 1)
        covered = (float((1.0 - QPOW[np.minimum(hit_count, len(QPOW) - 1)]).sum())
                   if rprob is not None else int((hit_count > 0).sum()))
        cov_scale = max(covered, 1)
    else:
        cov_indptr = cov_idx = hit_count = None
        covered = 0
        cov_scale = 1

    def _recall_delta(i_out, i_in):
        """Change in the covered-reference objective from swapping i_out for i_in, uncommitted."""
        out_s = cov_idx[cov_indptr[i_out]:cov_indptr[i_out + 1]]
        in_s = cov_idx[cov_indptr[i_in]:cov_indptr[i_in + 1]]
        if rprob is not None:
            def _qsum(a, b):
                t = 0.0
                if len(a):
                    t += QPOW[hit_count[a]].sum()
                if len(b):
                    t += QPOW[hit_count[b]].sum()
                return t
            before = _qsum(out_s, in_s)
            np.subtract.at(hit_count, out_s, 1)
            np.add.at(hit_count, in_s, 1)
            after = _qsum(out_s, in_s)
            np.subtract.at(hit_count, in_s, 1)
            np.add.at(hit_count, out_s, 1)
            # objective is sum(1 - q^c); a fall in sum(q^c) is a rise in the objective
            return float(before - after), out_s, in_s
        np.subtract.at(hit_count, out_s, 1)
        lost = int((hit_count[out_s] == 0).sum()) if len(out_s) else 0
        # a reference point may appear twice in in_s? no — indices are unique per candidate
        gained = int((hit_count[in_s] == 0).sum()) if len(in_s) else 0
        np.add.at(hit_count, out_s, 1)
        return gained - lost, out_s, in_s

    # PROPERTY CONFORMITY. The module docstring above has always described the objective as
    # including "- w_prop * property-conformity distance", and until now that term existed only in
    # the docstring. Phase 1 scores conformity of charge, hydrophobic moment, length, GRAVY and
    # cationic fraction against the reference, as a 1-D Wasserstein distance normalised by the
    # reference's own spread -- and nothing in this objective looked at any of it.
    #
    # The consequence was measurable and backwards: the POOL scores conform_length 0.749 while the
    # LIBRARY assembled from it scores 0.057. Assembly was making the metric worse, because
    # matching embedding moments and covering reference points are both satisfiable while drifting
    # in length, and the design score prefers the middle of the length range.
    #
    # Wasserstein needs a sort per proposal and is unaffordable here. Mean and standard deviation
    # are O(d) incremental with d = 16 descriptors -- free next to the 64x64 covariance -- and they
    # are most of what the gap is: our length mean is 23.08 against the reference's 20.85 and our sd
    # 6.39 against 10.19. Everything is measured in units of the REFERENCE's own sd, which is the
    # same normalisation the metric applies.
    use_prop = w_prop > 0 and prop is not None and ref_prop is not None
    if use_prop:
        ref_pmu, ref_psd = ref_prop
        ref_psd = np.maximum(np.asarray(ref_psd, np.float64), 1e-9)
        P = np.asarray(prop, np.float64)
        psum = P[cur].sum(axis=0)
        psq = (P[cur] ** 2).sum(axis=0)

        def _prop_gap(s1, s2):
            m = s1 / n_cur
            v = np.maximum(s2 / n_cur - m * m, 0.0)
            dm = (m - ref_pmu) / ref_psd
            ds = (np.sqrt(v) - ref_psd) / ref_psd
            return float(dm @ dm + ds @ ds)

        n_cur = len(cur)
        prop_gap = _prop_gap(psum, psq)
        prop0 = max(prop_gap, 1e-9)
    else:
        psum = psq = None
        prop_gap = 0.0
        prop0 = 1.0

    # --- density-core term -------------------------------------------------------------
    # w_prop above matches the library's MEAN and SD to the reference's. The organisers'
    # ConformityScore does not reward that. It fits a Gaussian KDE on the reference and scores
    # each sequence by the fraction of held-out reference points whose log-density it beats --
    # so it rewards sitting in the reference's high-density CORE, and matching the reference's
    # spread actively lowers it by pushing mass into the tails.
    #
    # Measured on the real pool (property_gradient.py): the shipped library scores 0.329, and
    # the 50,000 pool members with the highest reference log-density score 0.622. The v15
    # Wasserstein proxy ranks that same library BELOW the shipped one (-0.764 vs -0.727), which
    # is how the wrong-signed proxy stayed invisible for so long.
    #
    # core_lp[i] is log p_ref(x_i) precomputed once per pool member, so the term is a running
    # mean -- O(1) per proposal, cheaper than every other term here. It is deliberately a
    # SEPARATE knob from w_prop rather than a replacement: the two point opposite ways, and the
    # optimiser has to trade this against FBD and recall, which collapsing onto the mode hurts.
    # --- EXACT conformity, band-penalised ------------------------------------------------
    # conf[i] is candidate i's exact additive contribution to seqme's ConformityScore
    # (conformity_exact.conformity_contributions, asserted equal to the metric to 1e-12). The
    # library's scored conformity is exactly mean(conf[selected]), so this is the statistic
    # itself, not a proxy for it. w_core below optimises log p_ref, which is a proxy -- and
    # reaching for a proxy immediately after a wrong-signed proxy cost us a quarter of Phase 1
    # is the mistake this term exists to stop repeating.
    #
    # BAND, not maximisation. The metric is a rank and rises monotonically as a library collapses
    # onto the reference mode, so maximising it is a Goodhart trap: held-out reference sequences
    # score 0.4883 against their own reference, and anything materially above that has stopped
    # matching the distribution and started concentrating on it. Penalise both sides of a target
    # band and the optimiser fixes a deficit without overshooting into concentration.
    use_conf = w_conf > 0 and conf is not None
    if use_conf:
        conf = np.asarray(conf, dtype=np.float64)
        conf_sum = float(conf[cur].sum())
    else:
        conf_sum = 0.0

    def _conf_pen(csum):
        if not use_conf:
            return 0.0
        c = csum / len(cur)
        return (max(0.0, conf_low - c) ** 2 + max(0.0, c - conf_high) ** 2)

    use_core = w_core > 0 and core_lp is not None
    if use_core:
        core_lp = np.asarray(core_lp, dtype=np.float64)
        core_sum = float(core_lp[cur].sum())
        core_scale = float(np.std(core_lp)) or 1.0
    else:
        core_sum = 0.0
        core_scale = 1.0

    gap0 = max(fbd, 1e-9)
    sscale = float(np.std(pool_scores)) or 1.0
    J = (w_score * (score / sscale) - w_fbd * (fbd / gap0) + w_cover * (cover / cover0)
         + w_recall * (covered / cov_scale) - w_prop * (prop_gap / prop0)
         + w_core * (core_sum / (len(cur) * core_scale))
         - w_conf * _conf_pen(conf_sum))

    outside = np.flatnonzero(~in_lib)
    if len(outside) == 0:
        return cur, {"accepted": 0, "fbd": fbd, "score": score}

    accepted = 0
    n = len(cur)
    for it in range(iters):
        pos = rng.randrange(n)
        i_out = cur[pos]
        i_in = int(outside[rng.randrange(len(outside))])
        if in_lib[i_in]:
            continue
        if forbid is not None and forbid(i_out, i_in):
            continue

        ms.remove(pool_emb[i_out])
        ms.add(pool_emb[i_in])
        if metric == "moment":
            new_fbd = moment_gap(ms, ref_mu, ref_var)
        else:
            mu, cov = ms.mean_cov()
            new_fbd = frechet(mu, cov, ref_mu, ref_cov)
        new_score = score + (pool_scores[i_in] - pool_scores[i_out]) / n
        if use_cover:
            counts[labels[i_out]] -= 1
            counts[labels[i_in]] += 1
            new_cover = cluster_entropy(counts)
        else:
            new_cover = cover
        if use_recall:
            d_cov, out_s, in_s = _recall_delta(i_out, i_in)
            new_covered = covered + d_cov
        else:
            d_cov, out_s, in_s, new_covered = 0, None, None, covered
        if use_prop:
            new_psum = psum - P[i_out] + P[i_in]
            new_psq = psq - P[i_out] ** 2 + P[i_in] ** 2
            new_prop = _prop_gap(new_psum, new_psq)
        else:
            new_psum = new_psq = None
            new_prop = prop_gap
        new_core = core_sum + (core_lp[i_in] - core_lp[i_out]) if use_core else core_sum
        new_conf = conf_sum + (conf[i_in] - conf[i_out]) if use_conf else conf_sum
        new_J = (w_score * (new_score / sscale) - w_fbd * (new_fbd / gap0)
                 + w_cover * (new_cover / cover0) + w_recall * (new_covered / cov_scale)
                 - w_prop * (new_prop / prop0)
                 + w_core * (new_core / (len(cur) * core_scale))
                 - w_conf * _conf_pen(new_conf))

        take = new_J > J
        if not take and temperature > 0:
            take = rng.random() < np.exp((new_J - J) / temperature)
        if take:
            cur[pos] = i_in
            in_lib[i_out] = False
            in_lib[i_in] = True
            J, fbd, score, cover = new_J, new_fbd, new_score, new_cover
            if use_prop:
                psum, psq, prop_gap = new_psum, new_psq, new_prop
            if use_core:
                core_sum = new_core
            if use_conf:
                conf_sum = new_conf
            if use_recall:                     # commit the counter only on an accepted swap
                np.subtract.at(hit_count, out_s, 1)
                np.add.at(hit_count, in_s, 1)
                covered = new_covered
            accepted += 1
            if on_accept is not None:
                on_accept(i_out, i_in)
        else:
            ms.remove(pool_emb[i_in])
            ms.add(pool_emb[i_out])
            if use_cover:
                counts[labels[i_in]] -= 1
                counts[labels[i_out]] += 1

    return cur, {"accepted": accepted, "fbd": fbd, "score": score, "cover": cover,
                 "covered_refs": covered, "recall_frac": covered / max(cov_scale, 1)}


def reference_moments(ref_emb):
    mu = ref_emb.mean(axis=0)
    cov = np.cov(ref_emb, rowvar=False)
    return mu, cov


def project(emb, k=64, seed=0):
    """Random projection to k dimensions — FBD on a 640-dim embedding costs a 640x640 sqrtm per
    proposal, which is the whole budget. The Fréchet distance between Gaussians is preserved in
    expectation under a random projection, and the ranking of swaps is what matters here."""
    rng = np.random.RandomState(seed)
    P = rng.normal(size=(emb.shape[1], k)) / np.sqrt(k)
    return emb @ P, P


def parallel_chains(library_idx, pool_emb, pool_scores, ref_mu, ref_cov, seeds,
                    iters=200000, w_score=1.0, w_fbd=1.0, metric="moment", workers=None):
    """Run independent swap chains on every core and keep the best library.

    A single chain is a greedy walk with one basin; the harness doctrine is that a method with
    one instance becomes a portfolio of diversified copies. Chains differ only by seed, which is
    enough — the acceptance path is sequential and history-dependent.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    args = [(list(library_idx), pool_emb, pool_scores, ref_mu, ref_cov, s, iters,
             w_score, w_fbd, metric) for s in seeds]
    workers = workers or min(len(seeds), (os.cpu_count() or 8))
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as ex:
        out = list(ex.map(_chain_worker, args))
    sscale = float(np.std(pool_scores)) or 1.0
    gap0 = max(out[0][1]["fbd"], 1e-9)
    best = max(out, key=lambda r: w_score * (r[1]["score"] / sscale) - w_fbd * (r[1]["fbd"] / gap0))
    return best


def _chain_worker(a):
    import random as _random
    (lib, emb, sc, mu, cov, seed, iters, w_score, w_fbd, metric) = a
    return optimise_library(lib, emb, sc, mu, cov, _random.Random(seed), iters=iters,
                            w_score=w_score, w_fbd=w_fbd, metric=metric)


def _self_test():
    """Check the incremental recall bookkeeping against a full recomputation.

    optimise_library maintains hit_count by applying and un-applying deltas around every proposal
    and committing only on acceptance. That is the shape of code that breaks silently: a delta that
    is subtly wrong still returns a library, still reports a plausible objective, and only shows up
    as an A/B arm that mysteriously fails to move. So verify the objective the optimiser THINKS it
    reached against the one recomputed from scratch on the returned library.

        python tasks/amp_design/gen/setopt.py --self-test
    """
    import random
    rs = np.random.RandomState(3)
    n_pool, n_ref, d, p = 1200, 300, 16, 0.1
    rows = [np.unique(rs.randint(0, n_ref, rs.randint(1, 10))) for _ in range(n_pool)]
    indptr = np.zeros(n_pool + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(r) for r in rows])
    idx = np.concatenate(rows)
    cover = (indptr, idx, n_ref)
    emb = rs.normal(size=(n_pool, d))
    scores = rs.normal(size=n_pool)
    ref = rs.normal(size=(500, d))

    def exact(sel):
        c = np.zeros(n_ref, dtype=np.int64)
        for i in sel:
            np.add.at(c, idx[indptr[i]:indptr[i + 1]], 1)
        return float((1.0 - (1.0 - p) ** c).sum())

    lib = list(range(400))
    new, info = optimise_library(list(lib), emb, scores, ref.mean(0), np.cov(ref, rowvar=False),
                                 random.Random(0), iters=4000, w_fbd=1.0, recall_cover=cover,
                                 w_recall=2.0, recall_subsample_p=p)
    ok = True
    if len(new) != len(lib) or len(set(new)) != len(lib):
        print(f"FAIL library size/uniqueness: {len(new)} / {len(set(new))} against {len(lib)}")
        ok = False
    before, after = exact(lib), exact(new)
    if after <= before:
        print(f"FAIL expected coverage did not improve: {before:.4f} -> {after:.4f}")
        ok = False

    # The boolean objective must still saturate -- that is the defect this replaced, and a
    # regression to it would look like "the term stopped helping" rather than like a bug.
    sel_b, val_b = greedy_cover_init(cover, 400, scores)
    sel_p, val_p = greedy_cover_init(cover, 400, scores, subsample_p=p)
    if not exact(sel_p) > exact(sel_b):
        print(f"FAIL greedy: probabilistic {exact(sel_p):.2f} did not beat boolean {exact(sel_b):.2f}")
        ok = False

    # The property term must actually close a conformity gap, and must do it by choosing
    # differently rather than by accepting less.
    dp = 5
    pr = rs.normal(loc=[2.0, 0, 3.0, 0, 0], scale=[1.0, 1, 0.6, 1, 1], size=(n_pool, dp))
    rf = rs.normal(loc=np.zeros(dp), scale=np.ones(dp), size=(3000, dp))
    ref_prop = (rf.mean(0), rf.std(0))

    def pgap(sel):
        P = pr[sel]
        dm = (P.mean(0) - ref_prop[0]) / ref_prop[1]
        ds = (P.std(0) - ref_prop[1]) / ref_prop[1]
        return float(dm @ dm + ds @ ds)

    gaps = {}
    for w in (0.0, 2.0):
        sel, inf = optimise_library(list(lib), emb, scores, ref.mean(0), np.cov(ref, rowvar=False),
                                    random.Random(0), iters=20000, w_fbd=1.0,
                                    prop=pr, ref_prop=ref_prop, w_prop=w)
        gaps[w] = (pgap(sel), inf["accepted"])
    if not gaps[2.0][0] < gaps[0.0][0] * 0.95:
        print(f"FAIL property term did not close the gap: {gaps[0.0][0]:.3f} -> {gaps[2.0][0]:.3f}")
        ok = False

    # The density-core term must raise the mean reference log-density -- and it must do so by
    # NARROWING the library, which is the one place it provably disagrees with w_prop.
    #
    # The first version of this check asserted the combined mean+sd gap got worse under w_core,
    # and it failed: on a pool shifted away from the reference, moving to the density core also
    # moves the mean toward the reference, so the combined gap CLOSES. The two terms agree on
    # location and differ only on spread. Asserting the wrong half is how a self-test launders a
    # wrong mental model into a green check, so the assertion is on the spread term alone.
    def sdgap(sel):
        """Mean library spread, in units of the reference's own sd. Not a GAP -- the direct
        quantity. Asserting on a distance to the reference sd is ambiguous about direction, and
        the whole point of this term is its direction."""
        return float(np.mean(pr[sel].std(0) / ref_prop[1]))

    from sklearn.neighbors import KernelDensity
    kde_t = KernelDensity(kernel="gaussian", bandwidth="silverman").fit(rf)
    lp_pool = kde_t.score_samples(pr)
    cores = {}
    for w in (0.0, 3.0):
        sel, inf = optimise_library(list(lib), emb, scores, ref.mean(0), np.cov(ref, rowvar=False),
                                    random.Random(0), iters=20000, w_fbd=1.0,
                                    core_lp=lp_pool, w_core=w)
        cores[w] = (float(lp_pool[sel].mean()), pgap(sel), sdgap(sel), inf["accepted"])
    if not cores[3.0][0] > cores[0.0][0]:
        print(f"FAIL core term did not raise reference log-density: "
              f"{cores[0.0][0]:.3f} -> {cores[3.0][0]:.3f}")
        ok = False
    if not cores[3.0][2] < cores[0.0][2] * 0.98:
        print(f"FAIL core term did not NARROW the library -- it is not distinguishable from "
              f"w_prop: mean spread {cores[0.0][2]:.3f} -> {cores[3.0][2]:.3f}")
        ok = False

    print(f"swap: expected coverage {before:.2f} -> {after:.2f} over {info['accepted']} accepted")
    print(f"core log-density {cores[0.0][0]:.3f} at w_core=0 -> {cores[3.0][0]:.3f} at w_core=3; "
          f"mean spread {cores[0.0][2]:.3f} -> {cores[3.0][2]:.3f} (must FALL: core narrows), "
          f"mean+sd gap {cores[0.0][1]:.2f} -> {cores[3.0][1]:.2f}")
    print(f"property gap {pgap(lib):.2f} start, {gaps[0.0][0]:.2f} at w_prop=0 "
          f"({gaps[0.0][1]} accepted), {gaps[2.0][0]:.2f} at w_prop=2 ({gaps[2.0][1]} accepted)")
    print(f"greedy: boolean picks score {exact(sel_b):.2f}, probabilistic picks score "
          f"{exact(sel_p):.2f}, both at boolean coverage "
          f"{int(val_b)}/{n_ref}")
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys as _sys
    if "--self-test" in _sys.argv:
        raise SystemExit(_self_test())
    print(__doc__)
