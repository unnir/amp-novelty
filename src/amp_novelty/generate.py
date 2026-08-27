"""Entry point: `uv run generate` → generate/library.fasta + generate/top.fasta.

Deterministic: a fixed default seed, fixed model weights, no wall-clock, thread-count or
batch-grouping dependence in the sampling path. Two consecutive 50,000-sequence builds were
compared with `cmp` and are byte-identical.

Every random draw in the path is seeded from an explicit generator. The last one that was not --
a torch.randperm on the global CUDA RNG, used to subsample the pool for k-NN radius estimation --
made three builds at the same seed produce three different libraries, differing in coverage by
6 points out of 4000. Small enough to look like noise, fatal to a byte-identical requirement.
"""

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "gen"))

# Point the cache INSIDE the package, before anything imports oracles.
#
# oracles.py resolves its cache to $AMP_CACHE_DIR or /tmp/amp_design_cache, falling back to the
# in-package directory only if that path cannot be created. On the machine that BUILT this package
# /tmp/amp_design_cache exists and is full of warm caches; on a fresh machine it is creatable and
# EMPTY, so the fallback never fires, the shipped cache/ is never read, and the 32 MB of reference
# embeddings and the fitted surrogates travel as dead weight while the run recomputes them.
#
# The failure mode is the dangerous kind: reproduction succeeds on the machine that built the
# package and diverges on the one that scores it. setdefault, so a caller can still override.
os.environ.setdefault("AMP_CACHE_DIR", str(HERE / "cache"))


def write_fasta(seqs, path):
    with open(path, "w") as f:
        for i, s in enumerate(seqs, 1):
            f.write(f">seq{i}\n{s}\n")


def _ship_reference_embeddings(cache_dst):
    """Ship the ESM-2 embeddings of the fixed reference sets the pipeline reads at runtime.

    Not the whole embedding cache (hundreds of MB of generated sequences, useless to anyone else)
    — just the sets that are the same on every run: the reference AMPs the assembler clusters
    against, and the measured-potent/weak sets behind the kNN signal and the envelope. Without
    them the organizers' reproduction run spends ten minutes recomputing fixed quantities before
    it generates a single peptide.
    """
    import pickle

    import oracles
    import solution

    emb = oracles.ESM2Embedder("150M")
    knn = oracles.ActivesKNN("eval")
    knn._build()
    wanted = set(benchmark.reference_amps("eval")) | set(solution._measured_potent())
    wanted |= set(knn.potent) | set(knn.weak)
    emb(sorted(wanted))                      # ensure every one is in the local cache
    subset = {s: emb.cache.d[s] for s in wanted if s in emb.cache.d}
    with open(os.path.join(cache_dst, "emb_esm2_150M.pkl"), "wb") as f:
        pickle.dump(subset, f, protocol=4)
    print(f"shipped {len(subset)} reference embeddings "
          f"({os.path.getsize(os.path.join(cache_dst, 'emb_esm2_150M.pkl')) / 1e6:.0f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sequences", type=int, default=50000)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--length", type=int, default=50)
    args = ap.parse_args()

    # Git LFS pointer guard.
    #
    # Weights and canonical caches ship through Git LFS because five files exceed GitHub's 100 MB
    # limit -- the same arrangement the organisers' own ampdiffusion-starter-kit uses. A clone made
    # WITHOUT git-lfs silently yields ~132-byte pointer files instead of weights, and torch then
    # fails with an opaque deserialisation error a long way from the cause. Check first, and say
    # the actual remedy.
    _ptr = []
    for _d in ("checkpoint", "cache"):
        _dir = HERE / _d
        if _dir.is_dir():
            for _f in sorted(_dir.rglob("*")):
                if _f.is_file() and _f.stat().st_size < 1000:
                    try:
                        if _f.open("rb").read(40).startswith(b"version https://git-lfs"):
                            _ptr.append(_f.relative_to(HERE))
                    except OSError:
                        pass
    if _ptr:
        print(f"{len(_ptr)} Git LFS file(s) are unresolved pointers, e.g. {_ptr[0]}.")
        print("This clone was made without git-lfs. Fix with:")
        print("    git lfs install && git lfs pull")
        print("Model weights and prediction caches are required; generation cannot proceed.")
        raise SystemExit(1)

    import oracles
    import solution

    params = None
    pfile = HERE / "best_params.json"
    if pfile.exists():
        import json
        params = json.loads(pfile.read_text())

    out = Path("generate")
    out.mkdir(parents=True, exist_ok=True)
    if not os.path.abspath(oracles.CACHE).startswith(str(HERE)):
        raise RuntimeError(
            f"cache resolved to {oracles.CACHE}, outside the package at {HERE}. The shipped "
            f"embeddings and surrogates would be ignored and this run would not reproduce.")
    res = solution.build(params, n_library=args.n_sequences, seed=args.seed)
    write_fasta(res["library"], out / "library.fasta")
    write_fasta(res["top"][: args.top_k], out / "top.fasta")
    print(f"wrote {len(res['library'])} sequences -> {out / 'library.fasta'}")
    print(f"wrote {len(res['top'][: args.top_k])} ranked candidates -> {out / 'top.fasta'}")


if __name__ == "__main__":
    main()
