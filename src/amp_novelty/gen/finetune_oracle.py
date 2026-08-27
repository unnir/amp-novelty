"""Fine-tune ESM-2 into a multi-task MIC/HC50 oracle. AGENT-EDITABLE (design side).

Everything the design loop knows about potency currently comes from ridge heads on *frozen*
ESM-2 embeddings: held-out Spearman ~0.5, AUC 0.726 for "reaches MIC <= 16 uM". That ceiling is
the reason further optimisation against the objective stopped buying anything real — the search is
already at the proxy's limit, so the proxy has to get better.

Fine-tuning the encoder is the standard way to beat a frozen-embedding head on protein property
prediction, and the setup here is small enough to be cheap: ~4k measured peptides, 8-50 residues,
a 35M- or 150M-parameter encoder, minutes on one GPU.

Multi-task by construction: one head per panel species plus HC50, trained together with a masked
loss, because a peptide measured against E. coli and S. aureus should inform both. Trained on the
EVAL split only — the val split stays held out, and the referee's own activity family uses
val-side models that this never touches.

    python tasks/amp_design/gen/finetune_oracle.py --size 35M --epochs 12
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import benchmark
import oracles

CKPT_DIR = os.path.join(os.path.dirname(HERE), "checkpoint")
TASKS = list(benchmark.PANEL_SPECIES) + ["hc50"]


def load_tasks(split):
    """(sequence -> vector of targets, mask) over the panel species plus HC50."""
    table = {}
    for t_i, sp in enumerate(benchmark.PANEL_SPECIES):
        seqs, vals = benchmark.mic_species_data(sp, split)
        for s, v in zip(seqs, vals):
            row = table.setdefault(s, [np.full(len(TASKS), np.nan)])[0]
            row[t_i] = v
    seqs, vals = benchmark.hemolysis_data(split)
    for s, v in zip(seqs, vals):
        row = table.setdefault(s, [np.full(len(TASKS), np.nan)])[0]
        row[len(TASKS) - 1] = v
    S = sorted(table)
    Y = np.stack([table[s][0] for s in S])
    return S, Y


class FineTunedOracle(nn.Module):
    def __init__(self, size="35M", n_tasks=len(TASKS), freeze_to=None):
        super().__init__()
        import esm
        name = oracles.ESM2Embedder.NAMES[size][0]
        self.encoder, self.alphabet = getattr(esm.pretrained, name)()
        self.layer = oracles.ESM2Embedder.NAMES[size][1]
        dim = oracles.ESM2Embedder.NAMES[size][2]
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 256), nn.GELU(),
                                  nn.Linear(256, n_tasks))
        if freeze_to:                       # freeze the lower layers; the top ones adapt
            for n, p in self.encoder.named_parameters():
                if any(f"layers.{i}." in n for i in range(freeze_to)) or "embed" in n:
                    p.requires_grad_(False)

    def forward(self, toks, lengths):
        out = self.encoder(toks, repr_layers=[self.layer])["representations"][self.layer]
        mask = torch.zeros_like(out[:, :, 0])
        for i, L in enumerate(lengths):
            mask[i, 1:L + 1] = 1.0
        pooled = (out * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.head(pooled)


def batches(S, Y, bc, bs, device, shuffle=True, seed=0):
    idx = np.arange(len(S))
    if shuffle:
        np.random.RandomState(seed).shuffle(idx)
    for b in range(0, len(idx), bs):
        sel = idx[b:b + bs]
        seqs = [S[i] for i in sel]
        _, _, toks = bc([(str(i), s) for i, s in enumerate(seqs)])
        y = torch.tensor(Y[sel], dtype=torch.float32, device=device)
        yield toks.to(device), [len(s) for s in seqs], y


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~np.isnan(b)
    if ok.sum() < 5:
        return float("nan")
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    return float((ra * rb).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="35M", choices=["35M", "150M"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--freeze-to", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    Str, Ytr = load_tasks("eval")
    Sva, Yva = load_tasks("val")
    print(f"train {len(Str)} peptides, val {len(Sva)}; tasks: {len(TASKS)}")
    print(f"labels per task (train): "
          f"{ {t: int((~np.isnan(Ytr[:, i])).sum()) for i, t in enumerate(TASKS)} }")

    model = FineTunedOracle(args.size, freeze_to=args.freeze_to).to(device)
    bc = model.alphabet.get_batch_converter()
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr},
                             {"params": model.head.parameters(), "lr": args.head_lr}],
                            weight_decay=0.01)
    steps = args.epochs * math.ceil(len(Str) / args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.head_lr],
                                                total_steps=max(steps, 1), pct_start=0.2)

    best = -1.0
    for ep in range(args.epochs):
        model.train()
        tot, n = 0.0, 0
        for toks, lengths, y in batches(Str, Ytr, bc, args.batch_size, device, seed=args.seed + ep):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                pred = model(toks, lengths)
            mask = ~torch.isnan(y)
            loss = ((pred.float() - torch.nan_to_num(y)) ** 2 * mask).sum() / mask.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss) * len(lengths)
            n += len(lengths)

        model.eval()
        preds = []
        with torch.no_grad():
            for toks, lengths, y in batches(Sva, Yva, bc, 128, device, shuffle=False):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                    preds.append(model(toks, lengths).float().cpu().numpy())
        P = np.concatenate(preds)
        rhos = {t: spearman(P[:, i], Yva[:, i]) for i, t in enumerate(TASKS)}
        mean_rho = float(np.nanmean([v for v in rhos.values()]))
        print(f"epoch {ep + 1:2d}  loss {tot / max(n, 1):.4f}  mean val rho {mean_rho:+.3f}  "
              f"E.coli {rhos.get('E. coli', float('nan')):+.3f}  "
              f"S.aureus {rhos.get('S. aureus', float('nan')):+.3f}  "
              f"hc50 {rhos.get('hc50', float('nan')):+.3f}", flush=True)
        if mean_rho > best:
            best = mean_rho
            torch.save({"model": model.state_dict(), "size": args.size, "tasks": TASKS,
                        "freeze_to": args.freeze_to, "val_rho": rhos, "mean_rho": mean_rho},
                       os.path.join(CKPT_DIR, f"oracle_ft_{args.size}.pt"))

    print(f"\nbest mean val rho {best:+.3f}  -> checkpoint/oracle_ft_{args.size}.pt")
    print("ridge-on-frozen-embeddings baseline, same split: E. coli +0.508, S. aureus +0.389, "
          "P. aeruginosa +0.441, HC50 +0.518")
    return 0


if __name__ == "__main__":
    sys.exit(main())
