"""Curriculum pretraining + SFT for the peptide LM. AGENT-EDITABLE (part of the generator).

    stage 1  background   2M natural peptides from UniParc — the grammar of real peptide
                          sequences (which residues follow which, how length and composition
                          co-vary). Cheap and broad.
    stage 2  AMP corpus   47k curated antimicrobial peptides — the dialect we want, entered as
                          a *fine-tune* rather than a mixture so the model keeps the broad prior
                          and can still be pulled back toward it by the KL term during RL.

Both stages are token-budgeted, bf16, and single-GPU. The whole curriculum is minutes on an
L40S: the point of a 7M-parameter model is that pretraining is not where the compute goes.

    python tasks/amp_design/gen/train.py --stage pretrain
    python tasks/amp_design/gen/train.py --stage sft
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import benchmark
import lm as lmod
import oracles

CKPT_DIR = os.path.join(os.path.dirname(HERE), "checkpoint")
os.makedirs(CKPT_DIR, exist_ok=True)

# The tokenised corpus is ~900 MB and entirely regenerable, so it lives on scratch rather than in
# the workspace. A quota failure there truncated it mid-write and killed a run with a torch
# serialization error that read like a bug in the model code — and the same class of failure has
# twice left a SOURCE file zero bytes, so writes into the workspace are worth minimising.
TOKEN_CACHE = os.environ.get("AMP_TOKEN_CACHE", "/tmp/amp_design_tokens")
os.makedirs(TOKEN_CACHE, exist_ok=True)


def encode_corpus(seqs, src_token, device="cpu", chunk=200_000):
    """Tokenise a corpus into a padded int16 tensor. Charge is computed once, in parallel."""
    out = np.full((len(seqs), lmod.CONTEXT), lmod.PAD, dtype=np.int16)
    charges = np.zeros(len(seqs), dtype=np.float32)
    for i in range(0, len(seqs), chunk):
        part = seqs[i:i + chunk]
        D = oracles.descriptor_matrix(part)
        charges[i:i + len(part)] = D[:, oracles.DESCRIPTORS.index("charge")]
    for i, s in enumerate(seqs):
        toks = lmod.encode(s, src_token, float(charges[i]))
        out[i, :len(toks)] = toks
    return torch.from_numpy(out.astype(np.int64))


def heldout_loss(model, data, device, n=20000, batch_size=512, seed=123):
    """Loss on a held-out slice — the number that says whether more capacity is worth buying.

    Training loss always falls with parameters; held-out loss falls only while the corpus can
    support them. At ~2.9M sequences (roughly 90M tokens) a 40M-parameter model is already far past
    the compute-optimal token-per-parameter ratio, so this is what decides whether scaling further
    buys generalisation or memorisation.
    """
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(data), generator=g)[:n]
    model.eval()
    tot, seen = 0.0, 0
    with torch.no_grad():
        for b in range(0, len(idx), batch_size):
            batch = data[idx[b:b + batch_size]].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                loss = model.loss(batch)
            tot += float(loss) * len(batch)
            seen += len(batch)
    model.train()
    return tot / max(seen, 1)


def train(model, data, steps, batch_size, lr, device, warmup=200, log_every=100, seed=0,
          holdout=None):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warmup, 1.0) * 0.5 * (1 + math.cos(math.pi * s / max(steps, 1))))
    g = torch.Generator().manual_seed(seed)
    n = len(data)
    t0, run = time.time(), None
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,), generator=g)
        batch = data[idx].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            loss = model.loss(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if step % log_every == 0 or step == steps - 1:
            tok_s = (step + 1) * batch_size * lmod.CONTEXT / (time.time() - t0)
            extra = ""
            if holdout is not None and (step == steps - 1 or step % (log_every * 10) == 0):
                extra = f"  heldout {heldout_loss(model, holdout, device):.4f}"
            print(f"  step {step:5d}/{steps}  loss {run:.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"{tok_s / 1e6:.2f}M tok/s{extra}", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pretrain", "ampsphere", "sft"], required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", action="store_true", help="name the checkpoint by architecture")
    ap.add_argument("--helix-pe", action="store_true",
                    help="add helical-wheel phase to the positional encoding")
    ap.add_argument("--init", default=None,
                    help="checkpoint to initialise from, overriding the default search. Give this "
                         "whenever you train a second lineage: without it every size writes to the "
                         "same lm_pretrain.pt / lm_sft.pt and a scaling experiment silently "
                         "overwrites the champion the generator is still sampling from.")
    ap.add_argument("--out", default=None, help="checkpoint filename to write, overriding the "
                                                "stage default")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    amp_seqs, background = benchmark.pretrain_corpus()

    if args.stage == "ampsphere":
        # Stage 1.5: metagenomic small-ORF peptides. They are natural and AMP-like but sit
        # outside the curated databases, which is exactly the novel-but-plausible region the
        # competition's 80%-identity rule rewards. Entered as a continuation of the generic
        # pretrain so the model keeps the broad peptide grammar underneath it.
        sphere = benchmark.ampsphere()
        print(f"stage 1.5: {len(sphere)} AMPSphere peptides")
        cache = os.path.join(TOKEN_CACHE, "tokens_ampsphere.pt")
        if os.path.exists(cache):
            data = torch.load(cache)
        else:
            data = encode_corpus(sphere, lmod.SRC_AMP)
            torch.save(data, cache)
        model = lmod.PeptideLM(args.d, args.layers, args.heads, helix_pe=args.helix_pe).to(device)
        pre = os.path.join(CKPT_DIR, args.init or "lm_pretrain.pt")
        if os.path.exists(pre):
            model.load_state_dict(torch.load(pre, map_location=device, weights_only=False)["model"])
            print("  continued from the UniParc pretrain")
        steps = args.steps or 2500
        train(model, data, steps, args.batch_size, args.lr or 2e-4, device, seed=args.seed)
        path = os.path.join(CKPT_DIR, "lm_ampsphere.pt")
    elif args.stage == "pretrain":
        print(f"stage 1: {len(background)} background peptides + {len(amp_seqs)} AMPs")
        cache = os.path.join(TOKEN_CACHE, "tokens_background.pt")
        if os.path.exists(cache):
            data = torch.load(cache)
        else:
            t0 = time.time()
            data = torch.cat([encode_corpus(background, lmod.SRC_BACKGROUND),
                              encode_corpus(amp_seqs, lmod.SRC_AMP)])
            torch.save(data, cache)
            print(f"  tokenised in {time.time() - t0:.0f}s -> {tuple(data.shape)}")
        model = lmod.PeptideLM(args.d, args.layers, args.heads, helix_pe=args.helix_pe).to(device)
        print(f"  model: {lmod.count_parameters(model) / 1e6:.2f}M parameters")
        steps = args.steps or 4000
        cut = max(len(data) - 100_000, int(0.9 * len(data)))    # hold out the tail
        train(model, data[:cut], steps, args.batch_size, args.lr or 3e-4, device, seed=args.seed,
              holdout=data[cut:])
        path = os.path.join(CKPT_DIR, f"lm_pretrain_d{args.d}L{args.layers}.pt"
                            if args.tag else "lm_pretrain.pt")
    else:
        print(f"stage 2: SFT on {len(amp_seqs)} curated AMPs")
        data = encode_corpus(amp_seqs, lmod.SRC_AMP)
        model = lmod.PeptideLM(args.d, args.layers, args.heads, helix_pe=args.helix_pe).to(device)
        for cand in ((args.init,) if args.init else ("lm_ampsphere.pt", "lm_pretrain.pt")):
            pre = os.path.join(CKPT_DIR, cand)
            if os.path.exists(pre):
                model.load_state_dict(torch.load(pre, map_location=device,
                                                 weights_only=False)["model"])
                print(f"  initialised from {cand}")
                break
        steps = args.steps or 1500
        train(model, data, steps, min(args.batch_size, 512), args.lr or 1e-4, device, seed=args.seed)
        path = os.path.join(CKPT_DIR, "lm_sft.pt")

    if args.out:
        path = os.path.join(CKPT_DIR, args.out)
    payload = {"model": model.state_dict(),
               "config": {"d": args.d, "layers": args.layers, "heads": args.heads,
                          "helix_pe": args.helix_pe}}
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    torch.load(tmp, map_location="cpu", weights_only=False)   # verify before trusting the file
    os.replace(tmp, path)
    print(f"saved {path}")

    model.eval()
    g = torch.Generator(device=device).manual_seed(1234)
    prompts = lmod.make_prompts(12, device=device, rng=None)
    seqs, _ = model.sample(prompts, temperature=1.0, top_p=0.95, generator=g)
    print("samples:")
    for s in seqs[:12]:
        d = oracles.descriptors(s) if s else None
        print(f"  {s:<52} len={len(s):2d}"
              + (f" charge={d[1]:+.1f} gravy={d[3]:+.2f}" if d is not None else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
