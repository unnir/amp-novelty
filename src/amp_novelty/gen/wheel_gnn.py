"""A GNN on the helical-wheel graph. AGENT-EDITABLE (an oracle lineage).

There is already one GNN in this loop — AMPredictor, a graph conv net over ESM-1b node features
and a predicted contact map — and per cohort it behaves like every other database-trained model:
rho +0.71 on analogues of known AMPs, +0.06 on de-novo designs. Architecture did not fix transfer,
because the problem is what the model learns to recognise, not how it aggregates.

So this tests a sharper hypothesis: does GEOMETRIC INDUCTIVE BIAS transfer where a learned
representation does not? The graph is the amphipathic helix itself —

    nodes   one per residue, features PHYSICOCHEMICAL only (hydrophobicity, Eisenberg scale,
            volume, aromaticity, charge, helix propensity) — deliberately no protein-LM embedding,
            because an embedding is where the similarity-to-known-AMPs shortcut lives
    edges   i-i+1 (backbone) plus i-i+3 and i-i+4 (the residues that sit on the same face of an
            alpha helix)

Message passing on that graph computes, structurally, "what is on my face of the helix" — the
quantity a hydrophobic moment compresses into one number. If geometry is what transfers, this
should beat the fine-tuned transformer on de-novo peptides while losing to it on analogues.

    python tasks/amp_design/gen/wheel_gnn.py --epochs 40
"""

import argparse
import os
import sys

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

VOL = dict(zip("AVLIPFMWGSTCYNQDEKRH",
               [88, 140, 166, 166, 112, 190, 163, 227, 60, 89,
                116, 108, 193, 114, 143, 111, 138, 168, 173, 153]))
PS_HELIX = dict(zip("AELMKRQIWSYFHVNTGDCP",
                    [0.0, 0.21, 0.24, 0.24, 0.26, 0.21, 0.39, 0.41, 0.49, 0.5,
                     0.53, 0.54, 0.61, 0.61, 0.65, 0.66, 1.0, 0.69, 0.68, 3.16]))


def node_features(seq):
    out = np.zeros((len(seq), 7), dtype=np.float32)
    for i, a in enumerate(seq):
        out[i] = [oracles.KD.get(a, 0.0) / 4.5,
                  oracles.EISENBERG.get(a, 0.0),
                  VOL.get(a, 120) / 227.0,
                  1.0 if a in "FWY" else 0.0,
                  1.0 if a in "KR" else 0.0,
                  -1.0 if a in "DE" else 0.0,
                  1.0 - PS_HELIX.get(a, 0.6) / 3.16]
    return out


def wheel_edges(n):
    src, dst = [], []
    for i in range(n):
        for d in (1, 3, 4):
            if i + d < n:
                src += [i, i + d]
                dst += [i + d, i]
    return np.array([src, dst], dtype=np.int64) if src else np.zeros((2, 0), dtype=np.int64)


class WheelGNN(nn.Module):
    def __init__(self, hidden=128, layers=3, n_tasks=len(TASKS)):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.inp = nn.Linear(7, hidden)
        self.convs = nn.ModuleList([GCNConv(hidden, hidden) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(2 * hidden, 128), nn.GELU(), nn.Linear(128, n_tasks))

    def forward(self, data):
        from torch_geometric.nn import global_max_pool, global_mean_pool
        x = torch.relu(self.inp(data.x))
        for conv, norm in zip(self.convs, self.norms):
            x = torch.relu(norm(conv(x, data.edge_index))) + x
        pooled = torch.cat([global_mean_pool(x, data.batch), global_max_pool(x, data.batch)], -1)
        return self.head(pooled)


def build(seqs, Y=None):
    from torch_geometric.data import Data
    out = []
    for i, s in enumerate(seqs):
        d = Data(x=torch.tensor(node_features(s)),
                 edge_index=torch.tensor(wheel_edges(len(s))))
        d.y = torch.tensor(Y[i] if Y is not None else np.zeros(len(TASKS)),
                           dtype=torch.float32).unsqueeze(0)
        out.append(d)
    return out


def load_tasks(split):
    table = {}
    for t_i, sp in enumerate(benchmark.PANEL_SPECIES):
        seqs, vals = benchmark.mic_species_data(sp, split)
        for s, v in zip(seqs, vals):
            table.setdefault(s, np.full(len(TASKS), np.nan))[t_i] = v
    seqs, vals = benchmark.hemolysis_data(split)
    for s, v in zip(seqs, vals):
        table.setdefault(s, np.full(len(TASKS), np.nan))[len(TASKS) - 1] = v
    S = sorted(table)
    return S, np.stack([table[s] for s in S])


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
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()

    from torch_geometric.loader import DataLoader
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Str, Ytr = load_tasks("eval")
    Sva, Yva = load_tasks("val")
    print(f"train {len(Str)}, val {len(Sva)} peptides; physicochemical nodes, wheel edges",
          flush=True)

    model = WheelGNN(args.hidden).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e3:.0f}k", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    tr = DataLoader(build(Str, Ytr), batch_size=args.batch_size, shuffle=True)
    va = DataLoader(build(Sva, Yva), batch_size=256, shuffle=False)

    best, best_state = -1.0, None
    for ep in range(args.epochs):
        model.train()
        for batch in tr:
            batch = batch.to(device)
            pred = model(batch)
            y = batch.y.view(pred.shape)
            mask = ~torch.isnan(y)
            loss = ((pred - torch.nan_to_num(y)) ** 2 * mask).sum() / mask.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in va:
                preds.append(model(batch.to(device)).cpu().numpy())
        P = np.concatenate(preds)
        rho = float(np.nanmean([spearman(P[:, i], Yva[:, i]) for i in range(len(TASKS))]))
        if rho > best:
            best, best_state = rho, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  mean val rho {rho:+.3f}", flush=True)

    model.load_state_dict(best_state)
    torch.save({"model": best_state, "hidden": args.hidden, "tasks": TASKS, "val_rho": best},
               os.path.join(CKPT_DIR, "wheel_gnn.pt"))
    print(f"\nbest mean val rho {best:+.3f}  (fine-tuned transformer +0.650, ridge ~+0.47)",
          flush=True)

    sys.path.insert(0, os.path.dirname(HERE))
    import reality_check
    for name, dmap in (("AMP-Diffusion (de novo)", reality_check._exp_ampdiff()),
                       ("HydrAMP (analogues)", reality_check._exp_hydramp()[0])):
        seqs = [s for s in sorted(dmap) if benchmark.is_valid(s)]
        meas = np.log10(np.array([dmap[s] for s in seqs]))
        loader = DataLoader(build(seqs), batch_size=256, shuffle=False)
        preds = []
        with torch.no_grad():
            for batch in loader:
                preds.append(model(batch.to(device)).cpu().numpy())
        P = np.concatenate(preds)
        v = -np.median(P[:, :-1], axis=1)
        active = (meas <= np.log10(benchmark.POTENCY_THRESHOLD_UM)).astype(int)
        order = np.argsort(-v)
        print(f"  {name:26s} rho {spearman(v, -meas):+.3f}   p@10 {active[order[:10]].mean():.0%}"
              f"   (n={len(seqs)})", flush=True)
    print("\nreference — fine-tuned transformer on the same cohorts: de novo rho +0.034, "
          "analogues +0.689")
    return 0


if __name__ == "__main__":
    sys.exit(main())
