"""A small conditional peptide language model. AGENT-EDITABLE (part of the generator).

Deliberately small: ~7M parameters over a 23-token vocabulary and a 52-token context. Peptides
are short and the labelled signal is tiny, so capacity beyond this buys memorisation of the AMP
databases — which the competition's 80%-identity rule then rejects. The compute budget belongs
in the *reinforcement* stage, not in parameter count.

Conditioning: every sequence is written as

    [BOS] [SRC] [LEN] [CHG] r1 r2 ... rn [EOS]

where SRC marks the corpus (natural background vs curated AMP), LEN the length decile and CHG
the net-charge bucket. Sampling then asks for what we want — AMP-like, 16-24 residues, +5 to +7
net charge — instead of filtering it out afterwards, which is how a generator keeps its yield.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

AA = "ACDEFGHIKLMNPQRSTVWY"
PAD, BOS, EOS = 0, 1, 2
AA_OFFSET = 3
SRC_BACKGROUND, SRC_AMP = 23, 24          # control tokens live above the residue block
LEN_BUCKETS = [(8, 12), (13, 16), (17, 20), (21, 25), (26, 32), (33, 40), (41, 50)]
LEN_OFFSET = 25                           # 25..31
CHG_BUCKETS = [(-99, 0), (0, 2), (2, 4), (4, 6), (6, 9), (9, 99)]
CHG_OFFSET = 32                           # 32..37
VOCAB = 38
MAX_LEN = 50
CONTEXT = MAX_LEN + 5

_AA2TOK = {a: i + AA_OFFSET for i, a in enumerate(AA)}
_TOK2AA = {v: k for k, v in _AA2TOK.items()}


def len_bucket(n):
    for i, (lo, hi) in enumerate(LEN_BUCKETS):
        if lo <= n <= hi:
            return LEN_OFFSET + i
    return LEN_OFFSET + len(LEN_BUCKETS) - 1


def charge_bucket(q):
    for i, (lo, hi) in enumerate(CHG_BUCKETS):
        if lo <= q < hi:
            return CHG_OFFSET + i
    return CHG_OFFSET + len(CHG_BUCKETS) - 1


def encode(seq, src_token, charge):
    return ([BOS, src_token, len_bucket(len(seq)), charge_bucket(charge)]
            + [_AA2TOK[a] for a in seq] + [EOS])


def decode(tokens):
    out = []
    for t in tokens:
        if t == EOS:
            break
        if t in _TOK2AA:
            out.append(_TOK2AA[t])
    return "".join(out)


class Block(nn.Module):
    def __init__(self, d, heads, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False, is_causal=True)
        x = x + a
        return x + self.mlp(self.ln2(x))

    def step(self, x_t, cache):
        """One decoding step, reusing cached keys and values.

        Generation was recomputing the whole prefix at every step and keeping only the last
        position: for a 55-token peptide that is 1,540 token-positions of attention and MLP where
        55 suffice. Since generation dominates the cost of a 50,000-sequence build, that waste is
        paid by every experiment in this project.

        This uses nn.MultiheadAttention's OWN parameters — in_proj_weight, in_proj_bias, out_proj —
        rather than replacing the module, so every existing checkpoint still loads unchanged and the
        arithmetic is the same arithmetic. `cache` is a mutable list per layer holding (k, v) of
        shape (B, heads, T, hd); it is appended to in place.
        """
        h = self.ln1(x_t)                                     # (B, 1, d)
        B, _, d = h.shape
        nh = self.attn.num_heads
        hd = d // nh
        qkv = F.linear(h, self.attn.in_proj_weight, self.attn.in_proj_bias)   # (B, 1, 3d)
        q, k, v = qkv.chunk(3, dim=-1)
        shape = lambda t: t.view(B, 1, nh, hd).transpose(1, 2)                # (B, nh, 1, hd)
        q, k, v = shape(q), shape(k), shape(v)
        if cache[0] is None:
            cache[0], cache[1] = k, v
        else:
            cache[0] = torch.cat([cache[0], k], dim=2)
            cache[1] = torch.cat([cache[1], v], dim=2)
        # Causal by construction: the cache holds exactly the positions up to and including t.
        att = torch.nn.functional.scaled_dot_product_attention(q, cache[0], cache[1])
        att = att.transpose(1, 2).reshape(B, 1, d)
        x_t = x_t + self.attn.out_proj(att)
        return x_t + self.mlp(self.ln2(x_t))


def helical_phase(T, device, n_control=4):
    """Where each position sits on the helical wheel, and on a beta-strand alternation.

    An alpha-helix advances about 100 degrees per residue, so residues i, i+3 and i+4 land on the
    same face — which is why amphipathicity, the property every antimicrobial peptide is built
    around, is a PERIODIC function of position with period 3.6. Absolute position embeddings can
    represent that, but only by learning 3.6-periodicity from data at every position independently.

    Handing the model sin/cos of the wheel angle makes "same face as residue i" a linear function of
    the encoding instead of something to memorise. The second harmonic sharpens the face boundary;
    the period-2 pair covers beta-strand amphipathicity, the other common AMP fold, where alternate
    residues face opposite sides.

    Control tokens sit before residue 0 and are given phase zero.
    """
    i = (torch.arange(T, device=device, dtype=torch.float32) - n_control).clamp(min=0.0)
    th = i * (100.0 * math.pi / 180.0)          # alpha-helix, 100 deg/residue
    be = i * math.pi                            # beta-strand, alternating faces
    return torch.stack([torch.sin(th), torch.cos(th),
                        torch.sin(2 * th), torch.cos(2 * th),
                        torch.sin(be), torch.cos(be)], dim=-1)


class PeptideLM(nn.Module):
    def __init__(self, d=256, layers=6, heads=8, dropout=0.0, helix_pe=False):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(CONTEXT, d)
        # Zero-initialised, so the model starts as an exact copy of the baseline and the comparison
        # measures whether the bias is USED, not whether the extra parameters perturbed the init.
        self.helix_pe = helix_pe
        if helix_pe:
            self.helix = nn.Linear(6, d, bias=False)
        self.blocks = nn.ModuleList([Block(d, heads, dropout) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)
        self.head.weight = self.tok.weight        # tied embeddings
        self.apply(self._init)
        if helix_pe:
            nn.init.zeros_(self.helix.weight)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))[None]
        if self.helix_pe:
            x = x + self.helix(helical_phase(T, idx.device))[None]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=idx.device), diagonal=1)
        for b in self.blocks:
            x = b(x, mask)
        return self.head(self.ln(x))

    def loss(self, idx):
        """Next-token cross-entropy, ignoring padding."""
        logits = self(idx[:, :-1])
        tgt = idx[:, 1:]
        return F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), ignore_index=PAD)

    @torch.no_grad()
    def sample(self, prompts, max_new=MAX_LEN, temperature=1.0, top_p=0.95, min_len=8,
               generator=None):
        """Batch nucleus sampling from a (B, 4) control-token prompt tensor.

        Returns (sequences, token tensor). EOS before `min_len` residues is suppressed so the
        model cannot satisfy the objective by emitting 8-mers.
        """
        device = prompts.device
        idx = prompts
        done = torch.zeros(len(idx), dtype=torch.bool, device=device)
        # Per-layer key/value caches. The first pass consumes the whole prompt; every later step
        # feeds a single token. See Block.step for why this matters.
        caches = [[None, None] for _ in self.blocks]
        pos_off = 0
        for step in range(max_new + 1):
            if step == 0:
                logits = self(idx)[:, -1, :].float()
                self._prime_caches(idx, caches)
                pos_off = idx.shape[1]
            else:
                logits = self._step_logits(idx[:, -1:], pos_off - 1, caches).float()
                pos_off += 1
            logits[:, PAD] = -float("inf")
            logits[:, BOS] = -float("inf")
            logits[:, SRC_BACKGROUND:] = -float("inf")     # control tokens only at the start
            if step < min_len:
                logits[:, EOS] = -float("inf")
            probs = F.softmax(logits / max(temperature, 1e-4), dim=-1)
            if top_p < 1.0:
                sp, si = torch.sort(probs, descending=True, dim=-1)
                cum = sp.cumsum(-1)
                sp[cum - sp > top_p] = 0.0
                sp /= sp.sum(-1, keepdim=True)
                pick = torch.multinomial(sp, 1, generator=generator)
                nxt = si.gather(-1, pick)
            else:
                nxt = torch.multinomial(probs, 1, generator=generator)
            nxt[done] = PAD
            idx = torch.cat([idx, nxt], dim=1)
            done |= (nxt.squeeze(-1) == EOS)
            if bool(done.all()):
                break
        seqs = [decode(row.tolist()[4:]) for row in idx]
        return seqs, idx

    @torch.no_grad()
    def _prime_caches(self, idx, caches):
        """Fill the caches from the prompt by replaying it one position at a time."""
        for c in caches:
            c[0] = c[1] = None
        T = idx.shape[1]
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        if self.helix_pe:
            x = x + self.helix(helical_phase(T, idx.device))[None]
        for t in range(T):
            h = x[:, t:t + 1]
            for b, c in zip(self.blocks, caches):
                h = b.step(h, c)

    @torch.no_grad()
    def _step_logits(self, tok, pos, caches):
        """Logits for one new token at absolute position `pos`, using the caches."""
        x = self.tok(tok) + self.pos(torch.tensor([pos], device=tok.device))[None]
        if self.helix_pe:
            ph = helical_phase(pos + 1, tok.device)[-1:].unsqueeze(0)
            x = x + self.helix(ph)
        for b, c in zip(self.blocks, caches):
            x = b.step(x, c)
        return self.head(self.ln(x))[:, -1, :]

    def sequence_logprob(self, idx):
        """Sum of log-probabilities of the generated residues (control prompt excluded)."""
        logits = self(idx[:, :-1])
        logp = F.log_softmax(logits.float(), dim=-1)
        tgt = idx[:, 1:]
        tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        mask = (tgt != PAD).float()
        mask[:, :3] = 0.0                     # the control tokens are given, not generated
        return (tok_lp * mask).sum(1), mask


def make_prompts(n, src=SRC_AMP, length_tok=None, charge_tok=None, device="cuda", rng=None):
    """Control-token prompts. Unspecified buckets are sampled uniformly over the plausible range."""
    g = rng
    lt = (torch.full((n,), length_tok) if length_tok is not None
          else torch.randint(LEN_OFFSET + 1, LEN_OFFSET + 5, (n,), generator=g))
    ct = (torch.full((n,), charge_tok) if charge_tok is not None
          else torch.randint(CHG_OFFSET + 2, CHG_OFFSET + 5, (n,), generator=g))
    return torch.stack([
        torch.full((n,), BOS), torch.full((n,), src), lt, ct
    ], dim=1).to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
