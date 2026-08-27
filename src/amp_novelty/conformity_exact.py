"""Exact seqme-compatible conformity score, and proof that it IS exact.

The firewall (referee_firewall.py) found our property family is the one place where we
rank libraries differently from the organisers: Spearman 0.814, top-decile overlap 0.58
over a 120-library zoo. Every other metric is 0.95+.

The cause is not a bug, it is a different statistic:

    ours    five independent 1-D Wasserstein distances between marginals, negated.
            Penalises ANY marginal mismatch, in either direction.

    seqme   ONE joint Gaussian KDE fitted on 4/5 of the reference, and the score is the
            fraction of the held-out reference fifth whose log-density the generated
            point beats.  Averaged over 5 folds.

Those disagree in a specific, exploitable way.  A library squeezed onto the reference
MODE has narrower marginals than the reference -- ours punishes that -- but every one of
its points sits in the density core, so it beats nearly every held-out reference point
and seqme scores it near 1.0.  Ours rewards matching the spread; theirs rewards being
typical.  On a metric that is 1/4 of Phase 1 that is not a detail.

Two further facts fall out of reading their code, both of which matter more than the
formula:

  * KernelDensity is fitted on the RAW descriptor matrix with a single scalar bandwidth
    ('silverman').  The descriptors are not standardised.  Length has a standard
    deviation of ~10 and gravy ~0.5, so the joint density is very nearly a density in
    length alone -- the other four descriptors move the score only through their
    correlation with it.
  * The score is a rank, so it saturates.  Once a library is inside the core, pushing it
    further in buys nothing, and the metric cannot distinguish good from very good.

Usage:  python conformity_exact.py --verify     (agreement against seqme, plus timings)
"""
import argparse
import time

import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KernelDensity


def conformity_kde(gen, ref, n_splits=5, bandwidth="silverman", seed=0):
    """seqme.ConformityScore on already-computed descriptor matrices.

    gen: (n_gen, d) descriptors of the library.  ref: (n_ref, d) of the reference.
    Returns the mean over folds -- the same number seqme reports as `MetricResult.value`.
    """
    gen = np.asarray(gen, dtype=float)
    ref = np.asarray(ref, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for train_idx, val_idx in kf.split(ref):
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(ref[train_idx])
        lp_val = kde.score_samples(ref[val_idx])
        lp_gen = kde.score_samples(gen)
        counts = (lp_gen[:, None] >= lp_val[None, :]).sum(axis=1)
        scores.append((counts / (len(lp_val) + 1)).mean())
    return float(np.mean(scores))


def _verify():
    import seqme

    rng = np.random.default_rng(0)
    # Descriptor-like scales on purpose: a wide one (length) and narrow ones.
    ref_d = np.column_stack([rng.normal(24, 9, 600), rng.normal(4.5, 2.0, 600),
                             rng.normal(-0.3, 0.5, 600)])
    gen_d = np.column_stack([rng.normal(26, 6, 400), rng.normal(5.0, 1.6, 400),
                             rng.normal(-0.2, 0.4, 400)])

    # seqme takes sequences and predictor callables; feed it indices and look the
    # descriptors up, so both sides see byte-identical numbers.
    ref_seqs = [f"R{i}" for i in range(len(ref_d))]
    gen_seqs = [f"G{i}" for i in range(len(gen_d))]
    table = {s: ref_d[i] for i, s in enumerate(ref_seqs)}
    table.update({s: gen_d[i] for i, s in enumerate(gen_seqs)})

    def pred(j):
        return lambda seqs: np.array([table[s][j] for s in seqs])

    for n_splits in (2, 5, 8):
        for bw in ("silverman", "scott", 2.0):
            theirs = seqme.metrics.ConformityScore(
                ref_seqs, [pred(0), pred(1), pred(2)], n_splits=n_splits,
                kde_bandwidth=bw, seed=0)(gen_seqs).value
            ours = conformity_kde(gen_d, ref_d, n_splits=n_splits, bandwidth=bw, seed=0)
            flag = "OK " if abs(theirs - ours) < 1e-12 else "MISMATCH"
            print(f"  {flag} n_splits={n_splits} bw={str(bw):9s} "
                  f"seqme={theirs:.12f} ours={ours:.12f}")

    print("\ncost at competition scale (5 descriptors, 3000 reference):")
    ref_big = rng.normal(0, 1, (3000, 5)) * np.array([9, 2, 0.5, 1.5, 3])
    for n_gen in (3000, 10000, 50000):
        gen_big = rng.normal(0.2, 0.9, (n_gen, 5)) * np.array([9, 2, 0.5, 1.5, 3])
        t = time.time()
        v = conformity_kde(gen_big, ref_big)
        print(f"  n_gen={n_gen:6d}   {time.time() - t:6.2f}s   score={v:.4f}")

    print("\nthe disagreement, made concrete -- shrink a library onto the reference mode:")
    ref_s = rng.normal(0, 1, (1500, 5)) * np.array([9, 2, 0.5, 1.5, 3])
    for shrink in (1.0, 0.8, 0.6, 0.4, 0.2, 0.05):
        g = rng.normal(0, 1, (1500, 5)) * np.array([9, 2, 0.5, 1.5, 3]) * shrink
        kde_v = conformity_kde(g, ref_s)
        w = -np.mean([_w1(g[:, j], ref_s[:, j]) / max(ref_s[:, j].std(), 1e-6)
                      for j in range(5)])
        print(f"  spread x{shrink:<5} seqme-conformity {kde_v:.3f}   ours {w:+.3f}")
    print("  -> theirs rises as the library collapses; ours falls. Opposite gradients.")


def conformity_contributions(gen, ref, n_splits=5, bandwidth="silverman", seed=0):
    """Per-candidate EXACT contribution to seqme's ConformityScore.

    The metric is additive over the generated set, which the swap optimiser can exploit exactly
    instead of through a proxy. Per fold f, seqme scores the library as the mean over generated
    points of #{r in V_f : logp_f(x) >= logp_f(r)} / (|V_f| + 1), then averages the folds. Both
    are means, so they commute:

        C(L) = mean_x [ mean_f count_f(x)/(|V_f|+1) ] = mean_x c(x)

    So c(x) is a fixed number per candidate for a fixed reference and fold split, and a swap moves
    the scored metric by exactly (c(in) - c(out)) / |L|. No proxy, no gradient-sign risk -- this IS
    the statistic being scored.

    _self_test asserts mean(c) == conformity_kde() to 1e-12; if that ever fails the decomposition
    is wrong and the term must not be used.
    """
    gen = np.asarray(gen, dtype=float)
    ref = np.asarray(ref, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    acc = np.zeros(len(gen), dtype=np.float64)
    for train_idx, val_idx in kf.split(ref):
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(ref[train_idx])
        lp_val = np.sort(kde.score_samples(ref[val_idx]))
        lp_gen = kde.score_samples(gen)
        # ">=" in seqme == number of val points whose log-density is <= this candidate's
        acc += np.searchsorted(lp_val, lp_gen, side="right") / (len(lp_val) + 1)
    return acc / kf.get_n_splits()


def _self_test():
    """mean(contributions) must equal the metric itself, or the decomposition is wrong."""
    rng = np.random.default_rng(1)
    ok = True
    for n_ref, n_gen, d in ((400, 300, 3), (900, 1500, 5), (600, 50, 2)):
        ref = rng.normal(0, 1, (n_ref, d)) * np.array([9, 2, 0.5, 1.5, 3][:d])
        gen = rng.normal(0.3, 0.8, (n_gen, d)) * np.array([9, 2, 0.5, 1.5, 3][:d])
        for splits in (2, 5):
            whole = conformity_kde(gen, ref, n_splits=splits)
            parts = conformity_contributions(gen, ref, n_splits=splits).mean()
            good = abs(whole - parts) < 1e-12
            ok &= good
            print(f"  {'OK ' if good else 'FAIL'} n_ref={n_ref} n_gen={n_gen} d={d} "
                  f"splits={splits}  metric={whole:.12f} mean(c)={parts:.12f}")
    # and the swap-update identity the optimiser relies on
    ref = rng.normal(0, 1, (500, 4)) * np.array([9, 2, 0.5, 1.5])
    pool = rng.normal(0.2, 0.9, (400, 4)) * np.array([9, 2, 0.5, 1.5])
    c = conformity_contributions(pool, ref)
    lib = list(range(200))
    before = conformity_kde(pool[lib], ref)
    i_out, i_in = lib[7], 350
    lib2 = [i_in if x == i_out else x for x in lib]
    after_direct = conformity_kde(pool[lib2], ref)
    after_incr = before + (c[i_in] - c[i_out]) / len(lib)
    good = abs(after_direct - after_incr) < 1e-12
    ok &= good
    print(f"  {'OK ' if good else 'FAIL'} O(1) swap update: direct={after_direct:.12f} "
          f"incremental={after_incr:.12f}")
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


def _w1(a, b):
    from scipy.stats import wasserstein_distance
    return wasserstein_distance(a, b)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(_self_test())
    if a.verify:
        _verify()
