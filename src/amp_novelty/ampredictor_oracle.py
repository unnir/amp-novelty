"""AMPredictor as a referee instrument. FIXED FILE — a measurement device, never a design tool.

AMPredictor (Dong et al., *eLife* 2025, doi:10.7554/eLife.97330) is one of the three surrogate
activity models the competition names for its Phase-1 scoring, alongside MBC-Attention and
DeepAMP. It predicts MIC from a graph over the peptide: ESM-1b per-residue embeddings as node
features, the ESM-1b predicted contact map (thresholded at 0.5) as edges, and a per-residue
Morgan-fingerprint vector concatenated at the head.

This module re-implements the upstream inference path in memory. The original writes one .pt file
per sequence to disk and rebuilds a PyG dataset from the directory, which is fine for a paper's
test set and hopeless for scoring a 50,000-member library.

Held out from the design loop by the same rule as APEX: `solution.py` must not import this.
"""

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
AMPREDICTOR_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "explorations", "amp", "AMPredictor"))
MAX_LEN = 65          # upstream pads every peptide to 65 residues
FP_BITS = 2048


class AMPredictorOracle:
    """Predicted MIC (the paper's target is log-scaled) for a list of peptides."""

    def __init__(self, device=None, batch_size=64):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self._model = None
        self.cache_path = os.path.join(CACHE, "ampredictor.pkl")
        self.cache = {}
        if os.path.exists(self.cache_path):
            try:
                import pickle
                with open(self.cache_path, "rb") as f:
                    self.cache = pickle.load(f)
            except Exception:
                self.cache = {}

    # ---------------------------------------------------------------- setup
    def _load(self):
        if self._model is not None:
            return
        sys.path.insert(0, AMPREDICTOR_DIR)
        from AMPredictor import GNNPredictor

        state = torch.load(os.path.join(AMPREDICTOR_DIR, "models", "model_GNNPredictor_.model"),
                           map_location="cpu", weights_only=False)
        model = GNNPredictor()
        model.load_state_dict(state)
        self._model = model.to(self.device).eval()

        import esm
        esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self._esm = esm_model.to(self.device).eval()
        self._bc = alphabet.get_batch_converter()

        with open(os.path.join(AMPREDICTOR_DIR, "data", "smiles_file.json")) as f:
            smiles = json.load(f)
        from rdkit import Chem
        from rdkit.Chem import AllChem
        self._fp = {}
        for aa, smi in smiles.items():
            mol = Chem.MolFromSmiles(smi)
            # upstream calls Chem.GetMorganFingerprintAsBitVect, which moved to AllChem in
            # current RDKit; same fingerprint, same radius and bit count
            self._fp[aa] = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=3,
                                                                         nBits=FP_BITS))

    # ---------------------------------------------------------------- features
    def _fingerprint(self, seq):
        rows = np.zeros((MAX_LEN, FP_BITS), dtype=np.float32)
        for i, aa in enumerate(seq[:MAX_LEN]):
            v = self._fp.get(aa)
            if v is not None:
                rows[i] = v
        return rows.mean(axis=1)          # -> (65,), matching the upstream head

    def _graphs(self, seqs):
        from torch_geometric.data import Data
        out = []
        for b in range(0, len(seqs), self.batch_size):
            chunk = seqs[b:b + self.batch_size]
            _, _, toks = self._bc([(str(i), s) for i, s in enumerate(chunk)])
            toks = toks.to(self.device)
            with torch.no_grad():
                res = self._esm(toks, repr_layers=[33], return_contacts=True)
            reps = res["representations"][33]
            contacts = res["contacts"]
            for i, s in enumerate(chunk):
                L = len(s)
                x = torch.zeros(MAX_LEN, 1280, device=self.device)
                x[:L] = reps[i, 1:L + 1]
                cm = torch.zeros(MAX_LEN, MAX_LEN, device=self.device)
                cm[:L, :L] = contacts[i, :L, :L]
                cm = cm + torch.eye(MAX_LEN, device=self.device)
                edge = (cm >= 0.5).nonzero(as_tuple=False).t().contiguous()
                out.append(Data(x=x.cpu(), edge_index=edge.cpu()))
        return out

    # ---------------------------------------------------------------- inference
    def __call__(self, seqs):
        """→ array of predicted activity (upstream scale; lower = more potent MIC)."""
        seqs = [s for s in seqs]
        missing = [s for s in dict.fromkeys(seqs) if s not in self.cache]
        if missing:
            self._load()
            from torch_geometric.loader import DataLoader
            graphs = self._graphs(missing)
            fps = torch.tensor(np.stack([self._fingerprint(s) for s in missing]),
                               dtype=torch.float32)
            loader = DataLoader(graphs, batch_size=self.batch_size, shuffle=False)
            preds, seen = [], 0
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(self.device)
                    n = int(batch.num_graphs)
                    fp = fps[seen:seen + n].to(self.device)
                    preds.append(self._model(batch, fp).cpu().numpy().flatten())
                    seen += n
            preds = np.concatenate(preds)
            for s, p in zip(missing, preds):
                self.cache[s] = float(p)
            import pickle
            tmp = self.cache_path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(self.cache, f, protocol=4)
            os.replace(tmp, self.cache_path)
        return np.array([self.cache[s] for s in seqs], dtype=np.float32)


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    import benchmark

    o = AMPredictorOracle()
    sets = {
        "measured potent": list(benchmark.anchor_potent())[:150],
        "measured weak": list(benchmark.anchor_weak())[:150],
        "uniprot": list(benchmark.anchor_uniprot())[:150],
        "HydrAMP top-100": list(benchmark.baseline_top()),
    }
    top = os.path.join(HERE, "best_top.fasta")
    if os.path.exists(top):
        sets["our top-100"] = [l.strip() for l in open(top) if not l.startswith(">")]
    for name, s in sets.items():
        s = [x for x in s if benchmark.is_valid(x)]
        p = o(s)
        print(f"{name:20s} mean {p.mean():+.3f}  median {np.median(p):+.3f}  n={len(s)}")
