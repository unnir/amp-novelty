"""GRPO fine-tuning of the peptide LM against a consensus reward. AGENT-EDITABLE.

Why GRPO and not DPO: the reward here is *computable per sample* (predicted MIC, AMP
probability, HC50, novelty, synthesizability), so there is no need to manufacture preference
pairs. Group-relative advantages give a baseline for free, and — the part DPO cannot express —
the group is also where the **diversity penalty** lives: a sample is rewarded relative to its own
group, and penalised for resembling its group-mates. Mode collapse is the default failure of RL
on sequence generators, and Phase 1 scores diversity and coverage directly.

Three anti-reward-hacking devices, because optimising hard against a learned oracle is exactly
the failure mode this competition exists to expose:

  consensus, not sum   potency is the *minimum* of independent normalised signals (Gram-negative
                       MIC, Gram-positive MIC, AMP probability). A sequence that fools one head
                       gains nothing.
  KL anchor            a penalty against the SFT reference model keeps samples inside the region
                       where the oracles were trained and are therefore meaningful.
  held-out gate        APEX is never in the reward. It only grades the result afterwards.

    python tasks/amp_design/gen/rl.py --steps 400
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from rapidfuzz import fuzz, process

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import benchmark
import lm as lmod
import metrics
import oracles

CKPT_DIR = os.path.join(os.path.dirname(HERE), "checkpoint")

# Reward normalisation anchors, in the units each oracle speaks. Fixed constants rather than
# batch statistics: batch-relative normalisation lets the policy inflate its own reward by
# collapsing the batch's variance.
MIC_GOOD, MIC_BAD = 0.6, 2.0          # log10 uM  (4 uM is good, 100 uM is useless)
HC50_GOOD, HC50_BAD = 2.1, 1.0        # log10 uM  (>=128 uM is safe, 10 uM is toxic)
# ESM-2 mean log-likelihood is a BAND, not a direction. Run 2 anchored it as "higher is better"
# and the policy walked straight into the low-entropy attractor: repetitive KWKKLL-style peptides
# that ESM-2 predicts almost perfectly (mean LL -0.47 vs -2.2 for measured-potent AMPs), which
# collapsed manifold recall to 0.21 and left FBD at 5.7. Real AMPs are LESS predictable than
# generic protein, not more. So the reward is now proximity to the reference band.
NAT_BAND_K = 2.0                      # tolerated deviation, in robust sigmas of the reference
# Novelty SATURATES: below 70% identity to the reference set there is nothing further to gain.
# Run 1 learned this the hard way — an unbounded novelty term is a gradient off the data
# manifold, and the potency heads extrapolate optimistically out there (APEX top-100 success
# fell 143x while the design-side reward rose 66%).
NOVELTY_SAT, NOVELTY_LIMIT = 70.0, 80.0


class Reward:
    """The design-side reward. Never sees APEX, never sees the val reference split."""

    def __init__(self, weights=None, ref_sample=None, seed=0, envelope_band=0.0):
        self.s = oracles.SurrogateBundle("eval").load()
        rng = np.random.RandomState(seed)
        ref = list(benchmark.novelty_reference())
        # The novelty term is checked against the FULL reference. It used to use a fixed random
        # 8,000-sequence subsample "to keep the term cheap", and that was a premature optimisation
        # that opened a reward-hacking surface: the policy can drive identity below the saturation
        # point against the 17% it sees — the term then reads +1.000 and supplies no gradient —
        # while drifting to within 80% of the 83% it never sees. It did exactly that. The strongest
        # generator trained this way put 6.28% of its samples over the competition's hard identity
        # limit, against 0.22% for the same architecture trained before the reward dominated.
        #
        # The full check costs 52 ms per step against 20 ms, i.e. 13 seconds added to a 23-minute
        # run. A constraint worth enforcing is worth enforcing against the whole constraint.
        self.ref = ([ref[i] for i in rng.choice(len(ref), min(ref_sample, len(ref)), replace=False)]
                    if ref_sample else ref)
        # 0.0 reproduces the historical point-attractor form exactly; a positive band makes the
        # term flat inside that radius of the potent centre.
        self.envelope_band = envelope_band
        self.nat = oracles.ESM2Naturalness("150M")
        self.knn = oracles.ActivesKNN("eval")
        import solution as _sol
        self._envelope_distance = _sol.envelope_distance
        ref_ll = self.nat(list(benchmark.anchor_potent())[:768])
        self.nat_mid = float(np.median(ref_ll))
        self.nat_scale = float(1.4826 * np.median(np.abs(ref_ll - self.nat_mid))) or 0.2
        # naturalness defaults to OFF: measured net-negative on the held-out gate (see rl notes)
        # Diversity has to scale with the rest of the reward. At knn 1.0 + envelope 1.2 against
        # diversity 0.4 the policy collapsed (batch diversity 0.45 -> 0.12, uniqueness -> 0.74):
        # the cheapest way to satisfy two "be like the actives" terms is to emit one peptide.
        # Weighted by what each component is measurably worth on the 46 de-novo peptides with real
        # MICs, not by what it sounds like it should be worth. Audited in reward_audit.py:
        #
        #   component     rho vs hit-rate   AUC at telling a dud from a non-dud
        #   envelope           +0.371              0.786
        #   selectivity        +0.365              0.793
        #   potency            +0.024              0.340
        #   knn                -0.121              0.291
        #
        # The two components with real skill had carried 0.8 and 0.5; the two that are actively
        # anti-predictive had carried 1.0 and 0.7, and the total reward came out at AUC 0.401 —
        # below chance at avoiding peptides that kill nothing. Its favourite 25 of the 46 held 9
        # duds where a random draw holds 7.1. The generator has been trained against that.
        #
        # Two DIFFERENT edits live here and they carry different risk, which is why --reward-mode
        # can separate them:
        #
        #   subtractive   potency 1.0 -> 0.3 and knn 0.7 -> 0.0. These remove weight from signals
        #                 that are anti-predictive off-manifold. Low risk: it stops the generator
        #                 being pushed toward a wrong target, and adds no pressure of its own.
        #   additive      envelope 0.8 -> 1.6 and selectivity 0.5 -> 1.0. MEASURED HARMFUL — the
        #                 default is 'subtractive' for that reason. Raising selectivity's weight
        #                 saturated it at its clip ceiling within 150 steps while the potency
        #                 component collapsed from 0.55 to 0.03. The cause is that the selectivity
        #                 term is population-dependent: across REAL AMPs it scores measured-WEAK
        #                 ones higher than measured-potent ones (0.823 vs 0.713, with 50% of the
        #                 weak set pinned at the ceiling against 26% of the potent set), even
        #                 though on the de-novo cohort higher predicted HC50 genuinely tracks
        #                 activity (rho +0.407). A signal whose direction depends on where the
        #                 distribution sits is safe to SELECT with, because selection is evaluated
        #                 at a fixed distribution, and unsafe to OPTIMISE, because optimisation
        #                 moves the distribution across the flip.
        #
        # potency drops to 0.3 rather than to 0 deliberately: the measured cohort is entirely
        # generator output and therefore already peptide-like, so it cannot test whether potency is
        # doing useful work further out, and the KL anchor is not a substitute for that.
        # diversity, novelty and synth are left alone — they serve mode collapse, the 80%-identity
        # rule and synthesis feasibility respectively, so their negative correlation with activity
        # is expected rather than a defect.
        self.w = dict(potency=0.3, selectivity=1.0, novelty=0.4, synth=0.3, diversity=1.2,
                      naturalness=0.0, knn=0.0, envelope=1.6)
        if weights:
            self.w.update(weights)

    def __call__(self, seqs, group_size):
        n = len(seqs)
        valid = np.array([benchmark.is_valid(s) for s in seqs])
        safe = [s if benchmark.is_valid(s) else "AAAAAAAA" for s in seqs]

        p_amp = self.s.amp_probability(safe, cached=False)
        mic_neg = self.s.log10_mic_gram_neg(safe, cached=False)
        mic_pos = self.s.log10_mic_gram_pos(safe, cached=False)
        hc50 = self.s.log10_hc50(safe, cached=False)

        def norm(x, good, bad):
            return np.clip((x - bad) / (good - bad), 0.0, 1.0)

        potency = np.minimum.reduce([norm(mic_neg, MIC_GOOD, MIC_BAD),
                                     norm(mic_pos, MIC_GOOD, MIC_BAD),
                                     p_amp])                      # consensus = the weakest signal
        selectivity = norm(hc50, HC50_GOOD, HC50_BAD)

        ident = process.cdist(safe, self.ref, scorer=fuzz.ratio, workers=-1,
                              dtype=np.uint8).max(axis=1).astype(np.float32)
        novelty = np.clip((NOVELTY_LIMIT - ident) / (NOVELTY_LIMIT - NOVELTY_SAT), 0.0, 1.0)
        novelty[ident > NOVELTY_LIMIT] = -1.0                     # void under the competition rule

        ll = self.nat.score_nocache(safe)
        naturalness = np.clip(1.0 - np.abs(ll - self.nat_mid) / (NAT_BAND_K * self.nat_scale),
                              0.0, 1.0)

        # The two signals that closed the Phase-2 gap at selection time, moved into the reward so
        # the policy stops spending its yield on candidates the selector will throw away:
        #   knn      proximity to peptides with a MEASURED MIC (cannot extrapolate)
        #   envelope Mahalanobis distance to the measured-potent descriptor distribution
        knn_raw = self.knn(safe, cached=False)
        knn = np.clip((knn_raw - (-0.003)) / (0.014 - (-0.003)), 0.0, 1.0)
        # Envelope as a PLATEAU, not a point attractor. The distance form rewards being ever closer
        # to the measured-potent centre, so once the anti-predictive potency and kNN weights were
        # removed and this term dominated relatively more, the policy concentrated on that centre
        # and the generated GRAVY spread collapsed from 0.765 to 0.489 — a 36% loss of exactly the
        # range the Gram-preference axis selects along, which deletes the category specialists the
        # top-100 allocation depends on before selection can reach them.
        #
        # Membership is what the evidence supports, not proximity: the envelope's AUC of 0.781 at
        # telling a dud from a non-dud is about being INSIDE the region occupied by peptides that
        # work, and a peptide at its edge is as valid as one at its middle. Full reward inside the
        # band, falling off outside it — the same shape dg_band already uses in design_score.
        ed = np.asarray(self._envelope_distance(safe), float)
        band = float(self.envelope_band)
        env = np.clip(1.0 - np.maximum(ed - band, 0.0) / 6.0, 0.0, 1.0)

        synth = np.array([1.0 if metrics.synthesizable(s) else 0.0 for s in safe])

        # group-internal diversity: penalise resembling your own group-mates
        div = np.zeros(n, dtype=np.float32)
        for g in range(0, n, group_size):
            grp = safe[g:g + group_size]
            M = process.cdist(grp, grp, scorer=fuzz.ratio, workers=-1, dtype=np.uint8).astype(np.float32)
            np.fill_diagonal(M, 0.0)
            div[g:g + group_size] = 1.0 - M.max(axis=1) / 100.0

        r = (self.w["potency"] * potency + self.w["selectivity"] * selectivity
             + self.w["novelty"] * novelty + self.w["synth"] * synth + self.w["diversity"] * div
             + self.w["naturalness"] * naturalness + self.w["knn"] * knn
             + self.w["envelope"] * env)
        r = np.where(valid, r, -2.0)
        parts = {"potency": potency.mean(), "selectivity": selectivity.mean(),
                 "novelty": novelty.mean(), "synth": synth.mean(), "diversity": div.mean(),
                 "naturalness": naturalness.mean(), "knn": knn.mean(), "envelope": env.mean(),
                 "loglik": ll.mean(), "identity": ident.mean(), "valid": valid.mean(),
                 "mic_neg": mic_neg.mean(), "hc50": hc50.mean()}
        return r.astype(np.float32), parts


def _save_verified(payload, path):
    """Write a checkpoint and read it back before trusting it.

    A disk-quota failure once truncated a checkpoint at exactly 128 MiB; nothing complained until
    an A/B run days later died on a corrupt zip archive and the training had to be repeated. Write
    to a temporary file, reload it, and only then move it into place.
    """
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    try:
        torch.load(tmp, map_location="cpu", weights_only=False)
    except Exception as e:
        os.remove(tmp)
        raise RuntimeError(f"checkpoint failed verification, not saved: {e}") from e
    os.replace(tmp, path)


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = lmod.PeptideLM(**ck["config"]).to(device)
    m.load_state_dict(ck["model"])
    return m, ck["config"]


def grpo(steps=400, groups=64, group_size=8, lr=1e-5, beta=0.15, temperature=1.0, top_p=0.95,
         seed=0, device=None, log_every=10, tag=None, weights=None, init=None, envelope_band=0.0):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sft = os.path.join(CKPT_DIR, init or "lm_sft.pt")
    if not os.path.exists(sft):
        raise SystemExit("run gen/train.py --stage sft first")
    policy, cfg = load_model(sft, device)
    ref, _ = load_model(sft, device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    reward_fn = Reward(seed=seed, weights=weights, envelope_band=envelope_band)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    g = torch.Generator(device=device).manual_seed(seed)
    n = groups * group_size
    hist = []

    for step in range(steps):
        t0 = time.time()
        policy.eval()
        prompts = lmod.make_prompts(groups, device=device)
        prompts = prompts.repeat_interleave(group_size, dim=0)          # same prompt per group
        with torch.no_grad():
            seqs, idx = policy.sample(prompts, temperature=temperature, top_p=top_p, generator=g)

        r, parts = reward_fn(seqs, group_size)
        R = torch.tensor(r, device=device).view(groups, group_size)
        # GRPO's whole gradient lives in WITHIN-group reward variance: a group whose members all
        # score alike contributes nothing but the noise that the +1e-4 lets through. Groups are
        # sampled uniformly regardless, so if the variance is heavy-tailed most of the compute is
        # spent on groups that cannot teach anything. Measured here rather than assumed, because it
        # decides whether adaptive allocation (drop degenerate groups, branch more where the
        # variance is) is worth building for this task or is a solution to somebody else's problem.
        gstd = R.std(dim=1)
        _gs = gstd.detach().cpu().numpy()
        _order = np.sort(_gs)[::-1]
        _tot = float(_order.sum()) + 1e-12
        var_stats = {
            "median_std": float(np.median(_gs)),
            "frac_degenerate": float((_gs < 0.05 * (np.median(_gs) + 1e-12)).mean()),
            "share_top_quartile": float(_order[:max(1, len(_order) // 4)].sum() / _tot),
        }
        adv = (R - R.mean(dim=1, keepdim=True)) / (R.std(dim=1, keepdim=True) + 1e-4)
        adv = adv.view(-1)

        policy.train()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            lp, mask = policy.sequence_logprob(idx)
            with torch.no_grad():
                lp_ref, _ = ref.sequence_logprob(idx)
        ntok = mask.sum(1).clamp(min=1)
        kl = (lp - lp_ref) / ntok
        loss = -(adv * (lp / ntok)).mean() + beta * (kl ** 2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        hist.append(float(r.mean()))
        if step % log_every == 0 or step == steps - 1:
            uniq = len(set(seqs)) / max(len(seqs), 1)
            print(f"  step {step:4d}/{steps}  reward {np.mean(hist[-log_every:]):+.3f}  "
                  f"pot {parts['potency']:.3f}  sel {parts['selectivity']:.3f}  "
                  f"nov {parts['novelty']:+.3f}  nat {parts['naturalness']:.3f} "
                  f"(LL {parts['loglik']:+.2f}/{reward_fn.nat_mid:+.2f})  "
                  f"knn {parts['knn']:.3f}  env {parts['envelope']:.3f}  "
                  f"div {parts['diversity']:.3f}  "
                  f"uniq {uniq:.2f}  KL {float(kl.abs().mean()):.4f}  {time.time() - t0:.1f}s",
                  flush=True)
            print(f"        group-variance: median std {var_stats['median_std']:.4f}  "
                  f"degenerate {var_stats['frac_degenerate']:.0%}  "
                  f"top-quartile carries {var_stats['share_top_quartile']:.0%} of it",
                  flush=True)

    payload = {"model": policy.state_dict(), "config": cfg,
               "reward_weights": reward_fn.w, "reward_history": hist,
               "hyper": {"steps": steps, "beta": beta, "lr": lr, "temperature": temperature,
                         "seed": seed}}
    # every RL run keeps its own checkpoint: a discarded lineage must stay reproducible, and
    # run 3 was thrown away only to discover its predecessor's weights had been overwritten
    out = os.path.join(CKPT_DIR, f"lm_grpo_{tag}.pt" if tag else "lm_grpo.pt")
    _save_verified(payload, out)
    if not tag:                       # a tagged run is an experiment; it must not become the
        _save_verified(payload,       # champion by side effect, only by an explicit promotion
                       os.path.join(CKPT_DIR, "lm_grpo.pt"))   # what the generator loads
    print(f"saved {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--groups", type=int, default=64)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.15)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None, help="name this checkpoint (kept alongside the champion)")
    ap.add_argument("--init", default=None, help="SFT checkpoint to start from (default lm_sft.pt)")
    ap.add_argument("--envelope-band", type=float, default=0.0,
                    help="radius inside which the envelope term is flat; 0 = the historical "
                         "point-attractor form")
    ap.add_argument("--reward-mode", choices=["full", "subtractive", "legacy"], default="subtractive",
                    help="full: both edits. subtractive: drop the anti-predictive weights only, "
                         "leaving envelope and selectivity where they were. legacy: the weights "
                         "used before the reward audit.")
    ap.add_argument("--w-naturalness", type=float, default=0.0,
                    help="0 by default: the likelihood term measured net-negative on the held-out "
                         "gate (run 3, val 0.691 -> 0.471)")
    ap.add_argument("--w-novelty", type=float, default=0.4)
    ap.add_argument("--w-knn", type=float, default=0.0)
    ap.add_argument("--w-envelope", type=float, default=1.6)
    ap.add_argument("--w-diversity", type=float, default=0.4,
                    help="the pool is assembled from a portfolio of policies and quota-selected, "
                         "so batch diversity in the reward is largely redundant with the "
                         "assembler and costs precision (measured: div 1.2 -> val 0.82 vs "
                         "div 0.4 -> val 1.31)")
    args = ap.parse_args()
    # 'full' is retained only so the failure can be reproduced; 'subtractive' is the shipped mode.
    W = {"full": {},
         "subtractive": {"envelope": 0.8, "selectivity": 0.5},
         "legacy": {"potency": 1.0, "knn": 0.7, "envelope": 1.2, "selectivity": 0.5}}[args.reward_mode]
    grpo(steps=args.steps, groups=args.groups, group_size=args.group_size, lr=args.lr,
         beta=args.beta, temperature=args.temperature, seed=args.seed, tag=args.tag,
         init=args.init, envelope_band=args.envelope_band,
         weights=dict({"naturalness": args.w_naturalness, "novelty": args.w_novelty,
                       "knn": args.w_knn, "envelope": args.w_envelope,
                       "diversity": args.w_diversity}, **W))
    return 0


if __name__ == "__main__":
    sys.exit(main())
