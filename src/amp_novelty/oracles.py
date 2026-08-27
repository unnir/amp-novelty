"""Oracles and embedders for amp_design. FIXED FILE — the referee's instruments.

Everything here is a *measurement device*, not a design tool. Three groups:

  physicochemistry   charge / GRAVY / hydrophobic moment / pI / aliphatic index / Boman /
                     instability — the property family Phase 1 scores for conformity.
  embedders          ESM-2 (the competition's stated embedder) and an independent k-mer
                     embedder used on the held-out side, both cached on disk.
  potency oracles    a *design-side* pair we train ourselves (AMP classifier + MIC regressor,
                     trained on Veltri/UniProt and GRAMPA) and a *reality-side* external one
                     (APEX-pathogen, Wan/de la Fuente — the assaying lab's own model, which
                     predicts MIC for 11 of the competition's own 20 strains).

THE SPLIT IS THE POINT. `solution.py` may use the design-side oracles; the referee grades the
result with APEX, which the design loop never sees. A selection rule that only works on the
oracles it was tuned against is exactly the failure mode this competition exists to expose.

Caches live in tasks/amp_design/cache/ (gitignored) and are keyed by sequence hash, so
repeated evaluation of an overlapping library is cheap.
"""

import hashlib
import json
import math
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Embedding and oracle caches live outside the repo by default. They are pure derived data — the
# 600 MB ESM-2 cache is regenerable from the sequences — but they sit on a quota shared with
# checkpoints and other tasks, and a full quota has now cost us a killed 50,000-library A/B and a
# truncated checkpoint. Keeping them on scratch takes the largest regenerable artefact off the
# quota entirely. Set AMP_CACHE_DIR to override; the in-repo path is still used if scratch is
# unavailable, and any existing cache there is migrated on first use rather than orphaned.
_SCRATCH = os.environ.get("AMP_CACHE_DIR") or "/tmp/amp_design_cache"
try:
    os.makedirs(_SCRATCH, exist_ok=True)
    CACHE = _SCRATCH
except OSError:
    CACHE = os.path.join(HERE, "cache")
EXP = os.path.abspath(os.path.join(HERE, "..", "..", "explorations", "amp"))
APEX_DIR = os.path.join(EXP, "ampdiffusion-starter-kit", "apex")
os.makedirs(CACHE, exist_ok=True)

AAS = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AAS)}

# ----------------------------------------------------------------------------- physicochemistry

# Kyte-Doolittle hydropathy
KD = dict(zip("AVLIPFMWGSTCYNQDEKRH",
              [1.8, 4.2, 3.8, 4.5, -1.6, 2.8, 1.9, -0.9, -0.4, -0.8,
               -0.7, 2.5, -1.3, -3.5, -3.5, -3.5, -3.5, -3.9, -4.5, -3.2]))
# Eisenberg consensus scale (used for the hydrophobic moment, as in the AMP literature)
EISENBERG = dict(zip("AVLIPFMWGSTCYNQDEKRH",
                     [0.62, 1.08, 1.06, 1.38, 0.12, 1.19, 0.64, 0.81, 0.48, -0.18,
                      -0.05, 0.29, 0.26, -0.78, -0.85, -0.90, -0.74, -1.50, -2.53, -0.40]))
# Boman (protein-protein interaction potential), kcal/mol
BOMAN = dict(zip("LIVFMWAYGTSHPCNDQEKR",
                 [-4.92, -4.92, -4.04, -2.98, -2.35, -2.33, -1.81, -1.39, -0.94, -1.08,
                  -1.24, 4.66, 0.0, -1.28, 12.07, 9.66, 5.54, 9.19, 8.31, 14.92]))
# Kyte-Doolittle-independent pKa set (Bjellqvist / EMBOSS-like) for charge & pI
PKA_SIDE = {"C": 8.5, "D": 3.9, "E": 4.1, "H": 6.0, "K": 10.5, "R": 12.5, "Y": 10.1}
PKA_NTERM, PKA_CTERM = 8.0, 3.1
# Guruprasad instability dipeptide weights are large; we use the classic aliphatic index and
# an approximation of instability via the DIWV-free proxy (fraction of destabilising residues).
DESTABILISING = set("PQSEGT")


def net_charge(seq, ph=7.4):
    """Net charge at pH (Henderson-Hasselbalch over side chains + termini)."""
    q = 1.0 / (1.0 + 10 ** (ph - PKA_NTERM)) - 1.0 / (1.0 + 10 ** (PKA_CTERM - ph))
    for aa in seq:
        pka = PKA_SIDE.get(aa)
        if pka is None:
            continue
        if aa in "KRH":
            q += 1.0 / (1.0 + 10 ** (ph - pka))
        else:
            q -= 1.0 / (1.0 + 10 ** (pka - ph))
    return q


def isoelectric_point(seq):
    lo, hi = 0.0, 14.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if net_charge(seq, mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def hydrophobic_moment(seq, angle=100.0, window=11):
    """Max Eisenberg hydrophobic moment over sliding windows (per-residue normalised)."""
    n = len(seq)
    w = min(window, n)
    rad = np.deg2rad(angle)
    best = 0.0
    for start in range(0, n - w + 1):
        sub = seq[start:start + w]
        c = s = 0.0
        for i, aa in enumerate(sub):
            h = EISENBERG.get(aa, 0.0)
            c += h * np.cos(i * rad)
            s += h * np.sin(i * rad)
        best = max(best, np.sqrt(c * c + s * s) / w)
    return best


def aliphatic_index(seq):
    n = len(seq)
    if n == 0:
        return 0.0
    a = seq.count("A") / n
    v = seq.count("V") / n
    il = (seq.count("I") + seq.count("L")) / n
    return 100.0 * (a + 2.9 * v + 3.9 * il)


def descriptors(seq):
    """The physicochemical descriptor vector for one peptide (ordered, named by DESCRIPTORS)."""
    n = max(len(seq), 1)
    gravy = sum(KD.get(a, 0.0) for a in seq) / n
    charge = net_charge(seq)
    return np.array([
        len(seq),
        charge,
        charge / n,                                   # charge density
        gravy,
        hydrophobic_moment(seq),
        isoelectric_point(seq),
        aliphatic_index(seq),
        sum(BOMAN.get(a, 0.0) for a in seq) / n,      # Boman index
        sum(seq.count(a) for a in "FWY") / n,         # aromaticity
        sum(seq.count(a) for a in "KR") / n,          # cationic fraction
        sum(seq.count(a) for a in "DE") / n,          # anionic fraction
        sum(seq.count(a) for a in "AVLIMFWC") / n,    # hydrophobic fraction
        seq.count("C") / n,
        seq.count("P") / n,
        seq.count("G") / n,
        sum(seq.count(a) for a in DESTABILISING) / n,
    ], dtype=np.float32)


DESCRIPTORS = ["length", "charge", "charge_density", "gravy", "hmoment", "pi", "aliphatic",
               "boman", "aromatic", "cationic", "anionic", "hydrophobic", "cys", "pro", "gly",
               "destab"]


def descriptor_matrix(seqs, workers=None):
    """(n, 16) descriptor matrix, computed across all cores."""
    if len(seqs) < 2000:
        return np.stack([descriptors(s) for s in seqs])
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    # Capped, and with a serial fallback: several of this task's jobs run concurrently (a sweep,
    # an A/B, a training chain), and an uncapped pool per job oversubscribes 128 cores until the
    # workers are killed — which surfaces as BrokenProcessPool in whichever job blinks first.
    workers = workers or min(os.cpu_count() or 8, 32)
    chunk = max(512, len(seqs) // (workers * 4))
    chunks = [seqs[i:i + chunk] for i in range(0, len(seqs), chunk)]
    try:
        # spawn, not fork: this runs after a CUDA context exists (ESM-2 / APEX live on the GPU),
        # and forking a process that holds a CUDA context is how a run dies with no error message
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            parts = list(ex.map(_desc_chunk, chunks))
    except Exception:
        parts = [_desc_chunk(c) for c in chunks]
    return np.concatenate(parts, axis=0)


def _desc_chunk(chunk):
    return np.stack([descriptors(s) for s in chunk])


def aa_composition(seqs):
    """(n, 20) amino-acid frequency matrix."""
    out = np.zeros((len(seqs), 20), dtype=np.float32)
    for i, s in enumerate(seqs):
        for a in s:
            j = AA_IDX.get(a)
            if j is not None:
                out[i, j] += 1
        out[i] /= max(len(s), 1)
    return out


# ----------------------------------------------------------------------------- embedders

class _DiskCache:
    """sequence -> vector cache, one .npz shard per embedder name."""

    def __init__(self, name, dim):
        self.path = os.path.join(CACHE, f"emb_{name}.pkl")
        self.dim = dim
        self.d = {}
        if os.path.exists(self.path):
            try:
                with open(self.path, "rb") as f:
                    self.d = pickle.load(f)
            except Exception:
                self.d = {}
        self.dirty = False

    def get_missing(self, seqs):
        return [s for s in dict.fromkeys(seqs) if s not in self.d]

    def put(self, seqs, vecs):
        for s, v in zip(seqs, vecs):
            self.d[s] = v.astype(np.float16)
        self.dirty = True

    def take(self, seqs):
        return np.stack([self.d[s].astype(np.float32) for s in seqs])

    def flush(self):
        """Persist the cache, but never let failing to do so kill the computation.

        A cache is an optimisation. This one is hundreds of megabytes, and it flushes in the middle
        of hour-long library builds, so a full disk used to raise out of the flush and destroy the
        whole run — losing the expensive work in order to fail at saving a shortcut to it. Worse,
        the partial .tmp left behind consumed the very space that was short (448 MB, observed).

        Now a failed flush cleans up after itself, disables further attempts for the process, warns
        once, and returns. The run continues at full correctness and merely recomputes what it
        would have looked up.
        """
        if not self.dirty or getattr(self, "_flush_disabled", False):
            return
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "wb") as f:
                pickle.dump(self.d, f, protocol=4)
            os.replace(tmp, self.path)
            self.dirty = False
        except OSError as e:
            self._flush_disabled = True
            try:
                os.unlink(tmp)
            except OSError:
                pass
            print(f"  [cache] could not persist {os.path.basename(self.path)} ({e.strerror}); "
                  f"continuing in memory for the rest of this process", flush=True)


class ESM2Embedder:
    """Mean-pooled ESM-2 residue embeddings on GPU, disk-cached.

    The competition states Phase 1 computes FBD / MMD / precision-recall on ESM-2 (and ESM-C)
    embeddings, so this is the referee's primary embedding space.
    """

    NAMES = {"150M": ("esm2_t30_150M_UR50D", 30, 640),
             "650M": ("esm2_t33_650M_UR50D", 33, 1280),
             "35M": ("esm2_t12_35M_UR50D", 12, 480)}

    def __init__(self, size="150M", device=None, batch_size=512):
        self.size = size
        self.model_name, self.layer, self.dim = self.NAMES[size]
        self.device = device or ("cuda" if _has_cuda() else "cpu")
        self.batch_size = batch_size
        self.cache = _DiskCache(f"esm2_{size}", self.dim)
        self._model = None
        self._bc = None

    def _load(self):
        if self._model is None:
            import esm
            import torch
            model, alphabet = getattr(esm.pretrained, self.model_name)()
            model = model.eval().to(self.device)
            if self.device == "cuda":
                model = model.half()
            self._model = model
            self._bc = alphabet.get_batch_converter()
            self._torch = torch

    def embed_nocache(self, seqs):
        """Embed without touching the disk cache — for RL rollouts, where every sample is new
        and caching millions of one-shot sequences would only grow a useless pickle."""
        self._load()
        torch = self._torch
        seqs = list(seqs)
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        out = np.zeros((len(seqs), self.dim), dtype=np.float32)
        for b in range(0, len(order), self.batch_size):
            idx = order[b:b + self.batch_size]
            _, _, toks = self._bc([(str(i), seqs[i]) for i in idx])
            toks = toks.to(self.device)
            with torch.no_grad():
                rep = self._model(toks, repr_layers=[self.layer])["representations"][self.layer]
            for k, i in enumerate(idx):
                L = len(seqs[i])
                out[i] = rep[k, 1:L + 1].float().mean(0).cpu().numpy()
        return out

    def __call__(self, seqs):
        seqs = list(seqs)
        missing = self.cache.get_missing(seqs)
        if missing:
            self._load()
            torch = self._torch
            # length-sorted batching keeps padding waste low
            order = sorted(range(len(missing)), key=lambda i: len(missing[i]))
            vecs = [None] * len(missing)
            for b in range(0, len(order), self.batch_size):
                idx = order[b:b + self.batch_size]
                batch = [(str(i), missing[i]) for i in idx]
                _, _, toks = self._bc(batch)
                toks = toks.to(self.device)
                with torch.no_grad():
                    out = self._model(toks, repr_layers=[self.layer])["representations"][self.layer]
                for k, i in enumerate(idx):
                    L = len(missing[i])
                    vecs[i] = out[k, 1:L + 1].float().mean(0).cpu().numpy()
            self.cache.put(missing, vecs)
            self.cache.flush()
        return self.cache.take(seqs)


class KmerEmbedder:
    """Normalised k-mer frequency embedding — an embedding family with no neural network in it.

    Used on the held-out (val) side: if a library's distributional match only exists in ESM-2
    space and vanishes in plain k-mer space, the match was fitted to the embedder, not real.
    """

    def __init__(self, k=2):
        self.k = k
        self.dim = 20 ** k
        self._index = None

    def __call__(self, seqs):
        k = self.k
        if self._index is None:
            self._index = {}
            import itertools
            for i, t in enumerate(itertools.product(AAS, repeat=k)):
                self._index["".join(t)] = i
        out = np.zeros((len(seqs), self.dim), dtype=np.float32)
        for i, s in enumerate(seqs):
            n = max(len(s) - k + 1, 1)
            for j in range(len(s) - k + 1):
                idx = self._index.get(s[j:j + k])
                if idx is not None:
                    out[i, idx] += 1
            out[i] /= n
        return out


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _import_from(path, name):
    """Import a module by file path, bypassing sys.path and therefore any name collision."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------- APEX (reality side)

APEX_PANEL = [
    "A. baumannii ATCC 19606", "E. coli ATCC 11775", "E. coli AIC221", "E. coli AIC222",
    "K. pneumoniae ATCC 13883", "P. aeruginosa PA01", "P. aeruginosa PA14",
    "S. aureus ATCC 12600", "S. aureus (ATCC BAA-1556) - MRSA",
    "vancomycin-resistant E. faecalis ATCC 700802", "vancomycin-resistant E. faecium ATCC 700221",
]
# indices into APEX_PANEL
APEX_GRAM_NEG = [0, 1, 2, 3, 4, 5, 6]
APEX_GRAM_POS = [7, 8, 9, 10]
APEX_MDR = [3, 8, 9, 10]           # CRE E. coli AIC222, MRSA, VRE E. faecalis, VRE E. faecium


class ApexOracle:
    """APEX-pathogen 8-model ensemble → MIC (uM) for 11 competition strains, on GPU.

    Faithful to the vendored APEX_predict.py: one-hot encoding to length 52, per-model
    prediction of -log10(MIC/1e6), inverted to uM, then averaged across the 8 models in MIC
    space (the upstream script's own aggregation).
    """

    def __init__(self, device=None, batch_size=4096):
        self.device = device or ("cuda" if _has_cuda() else "cpu")
        self.batch_size = batch_size
        self._models = None
        self.cache_path = os.path.join(CACHE, "apex_mic.pkl")
        self.cache = {}
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    self.cache = pickle.load(f)
            except Exception:
                self.cache = {}

    def _load(self):
        if self._models is not None:
            return
        import glob
        import torch
        # APEX and AMPredictor both ship a top-level `utils` module, and both directories get put
        # on sys.path by their respective wrappers. Whichever imported first wins for the rest of
        # the process, so scoring a library after AMPredictor had been loaded failed here with
        # "cannot import name make_vocab from utils" — pointing at AMPredictor's file. Importing
        # APEX's modules by explicit file location sidesteps the name collision entirely instead of
        # depending on sys.path order, which no caller can be expected to reason about.
        self._apex_utils = _import_from(os.path.join(APEX_DIR, "utils.py"), "apex_utils")
        sys.path.insert(0, APEX_DIR)
        import APEX_models  # noqa: F401  (unpickling needs the class in sys.modules)
        self._word2idx, _ = self._apex_utils.make_vocab()
        paths = sorted(glob.glob(os.path.join(APEX_DIR, "APEX_pathogen_models", "APEX_*")))
        if not paths:
            raise RuntimeError(f"APEX weights not found in {APEX_DIR} (run: git lfs pull)")
        self._models = [torch.load(p, map_location=self.device, weights_only=False).to(self.device).eval()
                        for p in paths]
        self._torch = torch

    def __call__(self, seqs):
        """→ (n, 11) predicted MIC in uM."""
        seqs = list(seqs)
        missing = [s for s in dict.fromkeys(seqs) if s not in self.cache]
        if missing:
            self._load()
            torch = self._torch
            onehot_encoding = self._apex_utils.onehot_encoding
            for b in range(0, len(missing), self.batch_size):
                chunk = missing[b:b + self.batch_size]
                X = torch.LongTensor(onehot_encoding(np.array(chunk), 52, self._word2idx)).to(self.device)
                acc = None
                with torch.no_grad():
                    for m in self._models:
                        p = 10 ** (6 - m(X).float().cpu().numpy())
                        acc = p if acc is None else acc + p
                acc /= len(self._models)
                for s, row in zip(chunk, acc):
                    self.cache[s] = row.astype(np.float32)
            tmp = self.cache_path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(self.cache, f, protocol=4)
            os.replace(tmp, self.cache_path)
        return np.stack([self.cache[s] for s in seqs])


# ----------------------------------------------------------------------------- design-side surrogates

class SurrogateBundle:
    """The design-side oracle set: AMP classifier, MIC regressors (Gram-/Gram+), HC50 regressor.

    Features are ESM-2 embeddings concatenated with physicochemical descriptors. The embedding
    is what lets these models see sequence *order* — a composition-matched shuffle of a real AMP
    has the same descriptors and a very different embedding, and the first calibration run showed
    a descriptor-only oracle scoring shuffled decoys as half-real.

    Heads are deliberately small (ridge / logistic on ~4k labelled peptides). With this little
    measured data, a large head memorises; the representation is where the capacity belongs.

    Trained per side ("eval" = the half solution.py may use, "val" = the disjoint half).
    """

    VERSION = 10
    N_BOOTSTRAP = 8

    def __init__(self, split="eval", embed_size="150M"):
        self.split = split
        self.embed_size = embed_size
        # The cache key includes a hash of the SPLIT RULE's source, not just a hand-bumped VERSION.
        # A surrogate fitted on the eval half is only meaningful while "the eval half" means what it
        # meant when it was fitted; changing benchmark._split_half without touching VERSION silently
        # reloads models trained on the other side of the new boundary, which quietly destroys the
        # eval/val separation that everything here is checked against. Deriving part of the key from
        # the code that defines the split makes that impossible to forget.
        import hashlib
        import inspect
        import benchmark as _bm
        rule = hashlib.sha256(inspect.getsource(_bm._split_half).encode()).hexdigest()[:8]
        self.path = os.path.join(CACHE,
                                 f"surrogates_{split}_{embed_size}_v{self.VERSION}_{rule}.pkl")
        self.models = None
        self._embedder = None

    def embedder(self):
        if self._embedder is None:
            self._embedder = ESM2Embedder(self.embed_size)
        return self._embedder

    def features(self, seqs, cached=True):
        """Feature matrix, memoised on the exact sequence list.

        There are ~30 heads (nine species, the Gram aggregates, HC50, the classifier, plus eight
        bootstrap members each), and every one of them used to rebuild this matrix from scratch:
        an embedding lookup, a descriptor pass and a concatenation over the whole pool, thirty
        times per scoring call. At a 115,000-candidate pool that was the dominant cost of every
        full-scale experiment.
        """
        seqs = list(seqs)
        key = None
        if cached:
            h = hashlib.sha1()
            h.update(str(len(seqs)).encode())
            for s in seqs:
                h.update(s.encode())
            key = h.hexdigest()
            hit = getattr(self, "_feat_cache", None)
            if hit is not None and hit[0] == key:
                return hit[1]
        emb = self.embedder()(seqs) if cached else self.embedder().embed_nocache(seqs)
        desc = descriptor_matrix(seqs)
        X = np.concatenate([emb, desc], axis=1)
        if key is not None:
            self._feat_cache = (key, X)      # one entry: scoring passes reuse the same list
        return X

    def fit(self):
        import benchmark
        from sklearn.linear_model import LogisticRegression, RidgeCV
        from sklearn.preprocessing import StandardScaler

        models = {}
        pos, neg = benchmark.classifier_data(self.split)
        X = self.features(pos + neg)
        y = np.array([1] * len(pos) + [0] * len(neg))
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=2000, C=0.1).fit(sc.transform(X), y)
        models["clf"] = (sc, clf)

        # Bootstrap ensembles, not single heads. APEX's own 8-model ensemble disagrees far more
        # on our candidates (CV 0.515) than on peptides with measured MIC (0.392), i.e. part of a
        # point prediction's optimism is model uncertainty. With an ensemble the design side can
        # select on a pessimistic quantile instead of the mean — potent even if the model is off.
        loaders = [("mic", benchmark.mic_data),
                   ("mic_gram_pos", benchmark.mic_panel_data_pos),
                   ("mic_gram_neg", benchmark.mic_panel_data_neg),
                   ("hc50", benchmark.hemolysis_data)]
        # one head per panel species: broad-spectrum activity is a mean over 20 strains, so the
        # design side needs to be able to ask "potent against THIS species", not just "gram-"
        for sp in benchmark.PANEL_SPECIES:
            loaders.append((f"mic_sp::{sp}",
                            lambda split, _sp=sp: benchmark.mic_species_data(_sp, split)))
        for name, loader in loaders:
            seqs, vals = loader(self.split)
            if len(seqs) < 50:
                continue
            Xr = self.features(seqs)
            scr = StandardScaler().fit(Xr)
            Xs, y = scr.transform(Xr), np.array(vals)
            reg = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(Xs, y)
            boots = []
            rs = np.random.RandomState(0)
            for b in range(self.N_BOOTSTRAP):
                idx = rs.choice(len(Xs), len(Xs), replace=True)
                from sklearn.linear_model import Ridge
                boots.append(Ridge(alpha=float(reg.alpha_)).fit(Xs[idx], y[idx]))
            models[name] = (scr, reg)
            models[name + "__boot"] = (scr, boots)

        self.models = models
        with open(self.path, "wb") as f:
            pickle.dump(models, f, protocol=4)

    def load(self):
        if self.models is not None:
            return self
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                self.models = pickle.load(f)
        else:
            self.fit()
        return self

    def _predict(self, key, seqs, cached=True, proba=False):
        self.load()
        sc, model = self.models[key]
        X = sc.transform(self.features(seqs, cached=cached))
        return model.predict_proba(X)[:, 1] if proba else model.predict(X)

    def amp_probability(self, seqs, cached=True):
        return self._predict("clf", seqs, cached, proba=True)

    def log10_mic(self, seqs, cached=True):
        """Predicted log10(MIC in uM) — the E. coli GRAMPA scale."""
        return self._predict("mic", seqs, cached)

    def log10_mic_gram_pos(self, seqs, cached=True):
        return self._predict("mic_gram_pos", seqs, cached)

    def log10_mic_gram_neg(self, seqs, cached=True):
        return self._predict("mic_gram_neg", seqs, cached)

    def log10_hc50(self, seqs, cached=True):
        """Predicted log10(HC50 in uM). Higher = safer. Trained on HemoPI2 measurements."""
        return self._predict("hc50", seqs, cached)

    def predict_quantile(self, key, seqs, q=0.75, cached=True):
        """Pessimistic prediction: the q-quantile across the bootstrap ensemble.

        For MIC and HC50 the target is log10 concentration, so a HIGH quantile of MIC is the
        pessimistic case (less potent) and a LOW quantile of HC50 is (more toxic). Callers pass
        the quantile that makes their direction conservative.
        """
        self.load()
        boot = self.models.get(key + "__boot")
        if boot is None:
            return self._predict(key, seqs, cached)
        sc, models = boot
        X = sc.transform(self.features(seqs, cached=cached))
        P = np.stack([m.predict(X) for m in models])
        return np.quantile(P, q, axis=0)

    def predict_boot(self, key, seqs, cached=True):
        """All bootstrap members' predictions: (n_bootstrap, n_sequences).

        The posterior over predictions, not a summary of it — what an acquisition function needs.
        """
        self.load()
        boot = self.models.get(key + "__boot")
        if boot is None:
            return self._predict(key, seqs, cached)[None, :]
        sc, models = boot
        X = sc.transform(self.features(seqs, cached=cached))
        return np.stack([m.predict(X) for m in models])

    def species_heads(self):
        """Names of the per-species MIC heads that trained successfully."""
        self.load()
        return [k for k in self.models if k.startswith("mic_sp::") and not k.endswith("__boot")]

    def log10_mic_species(self, species, seqs, cached=True, q=None):
        key = f"mic_sp::{species}"
        if key not in self.load().models:
            return None
        if q is None:
            return self._predict(key, seqs, cached)
        return self.predict_quantile(key, seqs, q=q, cached=cached)

    def report(self):
        """Held-out quality of each head — printed by reality_check.py."""
        return {k: type(v[1]).__name__ for k, v in self.load().models.items()}


class ActivesKNN:
    """Non-parametric potency signal: embedding-space proximity to peptides with MEASURED MIC.

    Why this exists. The parametric heads (ridge on ESM-2, GBM on descriptors) extrapolate — a
    generator optimising them walks to wherever they happen to be optimistic, which is how run 1
    reward-hacked and how the run-6 top-100 ended up looking, to the assaying lab's own model,
    more like measured-WEAK peptides (APEX median MIC50 347 uM) than measured-potent ones (219).
    A k-nearest-neighbour score cannot extrapolate: a candidate only scores well by sitting near
    peptides that were actually measured potent, which is also exactly what the competition says
    Phase 1 rewards ("biologically informed embeddings ... compared against known potent peptides").

    score = mean cosine similarity to the k nearest measured-POTENT peptides
          - mean cosine similarity to the k nearest measured-WEAK peptides

    so it is a local likelihood ratio, not a distance to a single blob.
    """

    POTENT_MAX, WEAK_MIN = 0.9, 1.8      # log10 uM: <= ~8 uM is potent, >= ~63 uM is weak

    def __init__(self, split="eval", k=8, embed_size="150M"):
        self.split = split
        self.k = k
        self.embedder = ESM2Embedder(embed_size)
        self._built = False

    def _build(self):
        if self._built:
            return
        import benchmark
        potent, weak = [], []
        for loader in (benchmark.mic_data, benchmark.mic_panel_data_neg, benchmark.mic_panel_data_pos):
            seqs, vals = loader(self.split)
            for s, v in zip(seqs, vals):
                if v <= self.POTENT_MAX:
                    potent.append(s)
                elif v >= self.WEAK_MIN:
                    weak.append(s)
        self.potent = sorted(set(potent))
        self.weak = sorted(set(weak))
        self.Ep = _l2norm(self.embedder(self.potent))
        self.Ew = _l2norm(self.embedder(self.weak))
        self._built = True

    def __call__(self, seqs, cached=True):
        self._build()
        E = _l2norm(self.embedder(seqs) if cached else self.embedder.embed_nocache(seqs))
        k = self.k

        def topk_mean(M):
            kk = min(k, M.shape[1])
            idx = np.argpartition(-M, kk - 1, axis=1)[:, :kk]
            return np.take_along_axis(M, idx, axis=1).mean(axis=1)

        return topk_mean(E @ self.Ep.T) - topk_mean(E @ self.Ew.T)


def _l2norm(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


# ----------------------------------------------------------------------------- hemolysis proxy

def hemolysis_risk(seqs):
    """Descriptor-based hemolysis risk in [0, 1] — higher = more likely hemolytic.

    KNOWN WEAKNESS, recorded on purpose: this is a biophysically-motivated index (hydrophobicity,
    hydrophobic moment, aromatic content, and length drive erythrocyte lysis), not a model fitted
    to HC50 data — the public HC50 sets are not vendored here yet. Selectivity is one of the five
    award categories, so replacing this with a model trained on DBAASP/HemoPI HC50 measurements is
    a first-class open task (see TASK.md, "open work").
    """
    D = descriptor_matrix(seqs)
    gravy = D[:, DESCRIPTORS.index("gravy")]
    hm = D[:, DESCRIPTORS.index("hmoment")]
    arom = D[:, DESCRIPTORS.index("aromatic")]
    hyd = D[:, DESCRIPTORS.index("hydrophobic")]
    chg = D[:, DESCRIPTORS.index("charge_density")]
    z = 1.1 * gravy + 2.0 * hm + 3.0 * arom + 2.5 * hyd - 6.0 * chg - 1.6
    return 1.0 / (1.0 + np.exp(-z))


class FineTunedMIC:
    """Inference wrapper for the fine-tuned multi-task MIC/HC50 oracle (gen/finetune_oracle.py).

    Held-out Spearman against the ridge heads it replaces, same split: E. coli +0.643 vs +0.508,
    S. aureus +0.517 vs +0.389, HC50 +0.533 vs +0.518, mean over ten tasks +0.625. Trained on the
    EVAL split only, so the referee's val-side activity family is untouched and the gate stays
    honest.
    """

    def __init__(self, size="35M", device=None, batch_size=256):
        self.size = size
        self.device = device or ("cuda" if _has_cuda() else "cpu")
        self.batch_size = batch_size
        self.path = os.path.join(HERE, "checkpoint", f"oracle_ft_{size}.pt")
        self._model = None
        self.tasks = None
        self.cache_path = os.path.join(CACHE, f"ft_{size}.pkl")
        self.cache = {}
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    self.cache = pickle.load(f)
            except Exception:
                self.cache = {}

    def available(self):
        return os.path.exists(self.path)

    def _load(self):
        if self._model is not None:
            return
        import torch
        sys.path.insert(0, os.path.join(HERE, "gen"))
        from finetune_oracle import FineTunedOracle
        ck = torch.load(self.path, map_location="cpu", weights_only=False)
        self.tasks = ck["tasks"]
        model = FineTunedOracle(ck["size"], n_tasks=len(ck["tasks"]))
        model.load_state_dict(ck["model"])
        self._model = model.to(self.device).eval()
        self._bc = model.alphabet.get_batch_converter()
        self._torch = torch

    def __call__(self, seqs, cached=True):
        """→ (n, n_tasks) predicted log10 MIC per panel species, plus log10 HC50 last."""
        self._load()
        torch = self._torch
        seqs = list(seqs)
        missing = [s for s in dict.fromkeys(seqs) if s not in self.cache] if cached else seqs
        if missing:
            order = sorted(range(len(missing)), key=lambda i: len(missing[i]))
            out = [None] * len(missing)
            for b in range(0, len(order), self.batch_size):
                idx = order[b:b + self.batch_size]
                chunk = [missing[i] for i in idx]
                _, _, toks = self._bc([(str(i), s) for i, s in enumerate(chunk)])
                with torch.no_grad():
                    with torch.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=(self.device == "cuda")):
                        pred = self._model(toks.to(self.device), [len(s) for s in chunk])
                pred = pred.float().cpu().numpy()
                for k, i in enumerate(idx):
                    out[i] = pred[k]
            if cached:
                for s, v in zip(missing, out):
                    self.cache[s] = v.astype(np.float32)
                tmp = self.cache_path + ".tmp"
                with open(tmp, "wb") as f:
                    pickle.dump(self.cache, f, protocol=4)
                os.replace(tmp, self.cache_path)
            else:
                return np.stack(out)
        return np.stack([self.cache[s] for s in seqs])


# ----------------------------------------------------------------------------- naturalness

class ESM2Naturalness:
    """Mean per-residue log-likelihood of a sequence under ESM-2 (wt-marginal approximation).

    Order-sensitive by construction: a composition-matched shuffle of a real peptide has the
    same amino-acid histogram and a much worse likelihood. Without a signal like this the whole
    metric panel can be satisfied by shuffled AMPs, which is exactly the decoy class the
    organizers say the competition's aggregation score was tuned to reject.
    """

    def __init__(self, size="150M", device=None, batch_size=256):
        self.size = size
        self.model_name, self.layer, _ = ESM2Embedder.NAMES[size]
        self.device = device or ("cuda" if _has_cuda() else "cpu")
        self.batch_size = batch_size
        self.path = os.path.join(CACHE, f"pll_esm2_{size}.pkl")
        self.d = {}
        if os.path.exists(self.path):
            try:
                with open(self.path, "rb") as f:
                    self.d = pickle.load(f)
            except Exception:
                self.d = {}
        self._model = None

    def _load(self):
        if self._model is None:
            import esm
            import torch
            model, alphabet = getattr(esm.pretrained, self.model_name)()
            self._model = model.eval().to(self.device)
            if self.device == "cuda":
                self._model = self._model.half()
            self._alphabet = alphabet
            self._bc = alphabet.get_batch_converter()
            self._torch = torch

    def score_nocache(self, seqs):
        """Log-likelihood without touching the disk cache — for RL rollouts."""
        self._load()
        torch = self._torch
        seqs = list(seqs)
        out = np.zeros(len(seqs), dtype=np.float32)
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        for b in range(0, len(order), self.batch_size):
            idx = order[b:b + self.batch_size]
            _, _, toks = self._bc([(str(i), seqs[i]) for i in idx])
            toks = toks.to(self.device)
            with torch.no_grad():
                logits = self._model(toks)["logits"].float()
            logp = torch.log_softmax(logits, dim=-1)
            for k, i in enumerate(idx):
                L = len(seqs[i])
                tgt = toks[k, 1:L + 1]
                out[i] = float(logp[k, 1:L + 1].gather(-1, tgt[:, None]).mean().item())
        return out

    def __call__(self, seqs):
        seqs = list(seqs)
        missing = [s for s in dict.fromkeys(seqs) if s not in self.d]
        if missing:
            self._load()
            torch = self._torch
            order = sorted(range(len(missing)), key=lambda i: len(missing[i]))
            for b in range(0, len(order), self.batch_size):
                idx = order[b:b + self.batch_size]
                batch = [(str(i), missing[i]) for i in idx]
                _, _, toks = self._bc(batch)
                toks = toks.to(self.device)
                with torch.no_grad():
                    logits = self._model(toks)["logits"].float()
                logp = torch.log_softmax(logits, dim=-1)
                for k, i in enumerate(idx):
                    L = len(missing[i])
                    tgt = toks[k, 1:L + 1]
                    lp = logp[k, 1:L + 1].gather(-1, tgt[:, None]).squeeze(-1)
                    self.d[missing[i]] = float(lp.mean().item())
            tmp = self.path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(self.d, f, protocol=4)
            os.replace(tmp, self.path)
        return np.array([self.d[s] for s in seqs], dtype=np.float32)


class MarkovNaturalness:
    """Order-2 Markov log-likelihood fitted on a reference set — the val side's order-sensitive
    signal, with no neural network anywhere in it."""

    def __init__(self, reference, order=2, alpha=0.5):
        self.order = order
        counts, ctx_tot = {}, {}
        for s in reference:
            for i in range(len(s) - order):
                ctx, nxt = s[i:i + order], s[i + order]
                counts[(ctx, nxt)] = counts.get((ctx, nxt), 0) + 1
                ctx_tot[ctx] = ctx_tot.get(ctx, 0) + 1
        self.counts, self.ctx_tot, self.alpha = counts, ctx_tot, alpha

    def __call__(self, seqs):
        o, a = self.order, self.alpha
        out = np.zeros(len(seqs), dtype=np.float32)
        for i, s in enumerate(seqs):
            tot, n = 0.0, 0
            for j in range(len(s) - o):
                ctx, nxt = s[j:j + o], s[j + o]
                c = self.counts.get((ctx, nxt), 0) + a
                t = self.ctx_tot.get(ctx, 0) + a * 20
                tot += math.log(c / t)
                n += 1
            out[i] = tot / max(n, 1)
        return out


# Wimley-White interfacial hydrophobicity, kcal/mol for transfer from water to a POPC interface
# (Wimley & White, Nat Struct Biol 1996). NEGATIVE favours partitioning into the membrane
# interface, which is the first step of the mechanism every membrane-active AMP shares. Charged
# residues are given their physiological forms.
WW_INTERFACE = {
    "W": -1.85, "F": -1.13, "Y": -0.94, "L": -0.56, "I": -0.31, "C": -0.24, "M": -0.23,
    "V": 0.07, "G": 0.01, "S": 0.13, "T": 0.14, "A": 0.17, "H": 0.17, "P": 0.45,
    "N": 0.42, "Q": 0.58, "R": 0.81, "K": 0.99, "D": 1.23, "E": 2.02,
}
# Folding a helix at the interface pays back roughly 0.4 kcal/mol per residue by removing
# backbone hydrogen-bonding cost — the term that makes amphipathic helices partition far better
# than their composition alone suggests.
HELIX_FOLDING_BONUS = 0.4


class WheelGNNOracle:
    """Inference wrapper for the helical-wheel GNN (gen/wheel_gnn.py).

    Kept as an ENSEMBLE MEMBER rather than a replacement. Alone it is no better than the
    transformer on de-novo peptides (rho +0.053 vs +0.034, both ~chance), but the two disagree
    enough that averaging is the best model measured on analogues (rho +0.735 vs +0.689) and
    slightly better on de-novo (+0.077). Their errors correlate at 0.86 on de-novo peptides, which
    is the ceiling on what any ensemble of the two can fix — variance, not the shared bias.
    """

    def __init__(self, device=None, batch_size=256):
        self.device = device or ("cuda" if _has_cuda() else "cpu")
        self.batch_size = batch_size
        self.path = os.path.join(HERE, "checkpoint", "wheel_gnn.pt")
        self._model = None
        self.cache_path = os.path.join(CACHE, "wheel_gnn_pred.pkl")
        self.cache = {}
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    self.cache = pickle.load(f)
            except Exception:
                self.cache = {}

    def available(self):
        return os.path.exists(self.path)

    def _load(self):
        if self._model is not None:
            return
        import torch
        sys.path.insert(0, os.path.join(HERE, "gen"))
        import wheel_gnn
        ck = torch.load(self.path, map_location="cpu", weights_only=False)
        m = wheel_gnn.WheelGNN(ck["hidden"])
        m.load_state_dict(ck["model"])
        self._model = m.to(self.device).eval()
        self._wheel = wheel_gnn
        self._torch = torch

    def __call__(self, seqs, cached=True):
        """→ (n, n_tasks) predicted log10 MIC per panel species, HC50 last."""
        self._load()
        from torch_geometric.loader import DataLoader
        torch = self._torch
        seqs = list(seqs)
        missing = [s for s in dict.fromkeys(seqs) if s not in self.cache] if cached else seqs
        if missing:
            out = []
            with torch.no_grad():
                for batch in DataLoader(self._wheel.build(missing), batch_size=self.batch_size,
                                        shuffle=False):
                    out.append(self._model(batch.to(self.device)).cpu().numpy())
            P = np.concatenate(out)
            if cached:
                for s, v in zip(missing, P):
                    self.cache[s] = v.astype(np.float32)
                tmp = self.cache_path + ".tmp"
                with open(tmp, "wb") as f:
                    pickle.dump(self.cache, f, protocol=4)
                os.replace(tmp, self.cache_path)
            else:
                return P
        return np.stack([self.cache[s] for s in seqs])


class HelicityEstimator:
    """Per-peptide helical fraction from ESM-2 contact maps, disk-cached.

    An i -> i+4 contact is the signature of an alpha-helical turn, so the mean i,i+4 contact
    probability over a sequence is a cheap estimate of how much of it is actually helical. This
    matters twice over: the interfacial partitioning free energy pays a folding bonus only for the
    part that folds, and an amphipathic moment computed over a chain that never forms a helix is
    measuring a geometry the peptide does not have.
    """

    def __init__(self, size="150M", device=None, batch_size=64):
        self.emb = ESM2Embedder(size)
        self.batch_size = batch_size
        self.path = os.path.join(CACHE, f"helicity_{size}.pkl")
        self.d = {}
        if os.path.exists(self.path):
            try:
                with open(self.path, "rb") as f:
                    self.d = pickle.load(f)
            except Exception:
                self.d = {}

    def __call__(self, seqs):
        import torch
        seqs = list(seqs)
        missing = [s for s in dict.fromkeys(seqs) if s not in self.d]
        if missing:
            self.emb._load()
            order = sorted(range(len(missing)), key=lambda i: len(missing[i]))
            for b in range(0, len(order), self.batch_size):
                idx = order[b:b + self.batch_size]
                chunk = [missing[i] for i in idx]
                _, _, toks = self.emb._bc([(str(i), s) for i, s in enumerate(chunk)])
                with torch.no_grad():
                    res = self.emb._model(toks.to(self.emb.device), repr_layers=[],
                                          return_contacts=True)
                C = res["contacts"].float().cpu().numpy()
                for k, s in enumerate(chunk):
                    L = len(s)
                    if L < 6:
                        self.d[s] = 0.0
                        continue
                    c = C[k, :L, :L]
                    self.d[s] = float(np.mean([c[j, j + 4] for j in range(L - 4)]))
            tmp = self.path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(self.d, f, protocol=4)
            os.replace(tmp, self.path)
        return np.array([self.d[s] for s in seqs], dtype=np.float32)


_HELIX = {}


def helical_fraction(seqs, size="150M"):
    if size not in _HELIX:
        _HELIX[size] = HelicityEstimator(size)
    h = _HELIX[size](seqs)
    return np.clip(h / 0.35, 0.0, 1.0)      # 0.35 is roughly the level of a well-formed helix


def structural_hydrophobic_moment(seqs, size="150M"):
    """Hydrophobic moment scaled by how much of the peptide is actually predicted helical."""
    D = descriptor_matrix(list(seqs))
    return D[:, DESCRIPTORS.index("hmoment")] * helical_fraction(seqs, size)


def interfacial_dG(seqs, anionic_bonus=0.0, helical_fraction=None):
    """Free energy of partitioning into a membrane interface (kcal/mol; lower = binds better).

    This is the analytic core of what a bilayer simulation computes, at microseconds per peptide
    instead of GPU-hours — which matters because Phase 1 scores 50,000 sequences, not 100. It is
    also, unlike every trained model here, derived from physics rather than from AMP databases, so
    it cannot carry the similarity-to-known-AMPs shortcut that puts MIC regressors at chance on
    de-novo peptides.

    `anionic_bonus` adds an electrostatic term per positive charge, standing in for the anionic
    lipids (PG, cardiolipin) of a bacterial membrane; setting it to zero approximates the
    zwitterionic outer leaflet of a human cell, so the DIFFERENCE between the two is a physical
    selectivity estimate.
    """
    out = np.zeros(len(seqs), dtype=np.float32)
    for i, s in enumerate(seqs):
        dg = sum(WW_INTERFACE.get(a, 0.3) for a in s)
        frac = 1.0 if helical_fraction is None else float(helical_fraction[i])
        dg -= HELIX_FOLDING_BONUS * frac * len(s)
        if anionic_bonus:
            dg -= anionic_bonus * max(net_charge(s), 0.0)
        out[i] = dg
    return out


def selectivity_dG(seqs, anionic_bonus=0.35):
    """Bacterial-minus-human partitioning free energy (lower = prefers the bacterial membrane).

    The selectivity award is exactly this quantity's biophysical meaning: a peptide that partitions
    into anionic bacterial membranes but not into zwitterionic human ones kills bacteria and spares
    red blood cells.
    """
    return (interfacial_dG(seqs, anionic_bonus=anionic_bonus)
            - interfacial_dG(seqs, anionic_bonus=0.0))


def aggregation_propensity(seqs):
    """Heuristic aggregation risk — a peptide that precipitates reads as inactive in the well.

    Not a trained model and not derived from any AMP database, which is the point: on de-novo
    peptides every database-derived signal is at chance, so a purely physical failure mode is
    worth measuring separately. Three terms with the same sign as the literature on peptide
    aggregation: uninterrupted hydrophobic stretches nucleate beta-aggregation; aromatic and
    beta-branched residues stabilise it; net positive charge opposes it electrostatically.

    Higher = more likely to aggregate (and therefore to read as inactive regardless of design).
    """
    out = np.zeros(len(seqs), dtype=np.float32)
    for i, s in enumerate(seqs):
        n = max(len(s), 1)
        run = best = 0
        for a in s:
            run = run + 1 if a in "AVLIMFWYC" else 0
            best = max(best, run)
        aromatic = sum(s.count(a) for a in "FWY") / n
        branched = sum(s.count(a) for a in "IVTY") / n
        charge = net_charge(s)
        out[i] = (0.45 * best + 3.0 * aromatic + 2.0 * branched - 0.35 * max(charge, 0.0))
    return out


def sequence_hash(seqs):
    h = hashlib.sha256()
    for s in seqs:
        h.update(s.encode())
        h.update(b"\n")
    return h.hexdigest()[:16]
