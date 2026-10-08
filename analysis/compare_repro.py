"""Reproduction check: our runs vs the released runs of the same model, paired by episode name (same question, graph,
personas and seed).

Compares per round:
- conviction c(t) = mean_i obar_i(t)^2;
- |net opinion| |n(t)|, n(t) = mean_i obar_i(t);
- objective accuracy (share of agents whose state matches the ground truth).

It also compares the fitted influence coefficients (own previous state, friendly drive, unfriendly drive), as in
consensus_claims.py.

    uv run --no-project --with numpy --with scikit-learn python analysis/compare_repro.py OURS_DIR data/models/qwen3.5-9b
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


def episodes(root: Path) -> dict[str, dict]:
    return {p.name: json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.rglob("*__rep*.json"))}


def truth_map(clean_runs: Path) -> dict[str, int]:
    gt = {}
    for r in json.loads(clean_runs.read_text(encoding="utf-8")):
        if r.get("ground_truth") in ("A", "B"):
            gt[r["statement"]] = +1 if r["ground_truth"] == "A" else -1
    return gt


def curves(ep: dict, gt: dict[str, int]) -> dict[str, np.ndarray]:
    raw = np.array(ep["spins_raw_history"], dtype=float)  # (T+1, N, K)
    o = raw.mean(axis=2)
    s = np.array(ep["spins_history"])
    out = {"conviction": (o ** 2).mean(axis=1), "abs_net": np.abs(o.mean(axis=1))}
    if ep["statement"] in gt:
        out["accuracy"] = (s == gt[ep["statement"]]).mean(axis=1)
    return out


def coefs(eps: list[dict]) -> np.ndarray:
    X, y = [], []
    for ep in eps:
        J, S = np.array(ep["J"]), np.array(ep["spins_history"])
        for t in range(1, S.shape[0]):
            for i in range(S.shape[1]):
                if S[t, i] == 0:
                    continue
                fr = float(np.sum(np.maximum(J[:, i], 0) * S[t - 1]))
                un = float(np.sum(np.minimum(J[:, i], 0) * -1 * S[t - 1]))  # drive from unfriendly neighbours
                X.append([S[t - 1, i], fr, un])
                y.append(int(S[t, i] > 0))
    return LogisticRegression(max_iter=1000).fit(np.array(X), np.array(y)).coef_[0]


def main(ours: str, theirs: str, clean_runs: str = "data/clean_runs.json") -> None:
    a, b = episodes(Path(ours)), episodes(Path(theirs))
    common = sorted(set(a) & set(b))
    if not common:
        raise SystemExit(f"no common episode names between {ours} and {theirs}")
    gt = truth_map(Path(clean_runs))
    print(f"{len(common)} paired episodes (ours {len(a)}, theirs {len(b)})")
    for key in ("conviction", "abs_net", "accuracy"):
        pa = [curves(a[n], gt).get(key) for n in common]
        pb = [curves(b[n], gt).get(key) for n in common]
        pairs = [(x, y) for x, y in zip(pa, pb) if x is not None and y is not None and len(x) == len(y)]
        if not pairs:
            continue
        A, B = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
        d = A[:, -1] - B[:, -1]
        rng = np.random.default_rng(0)
        boot = [rng.choice(d, len(d)).mean() for _ in range(2000)]
        print(f"\n{key} by round (mean over {len(pairs)} episodes)")
        print("  ours  :", np.round(A.mean(axis=0), 3))
        print("  theirs:", np.round(B.mean(axis=0), 3))
        print(f"  final-round difference ours - theirs: {d.mean():+.3f} "
              f"[{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}]")
    print("\ninfluence coefficients [self, friendly, unfriendly]")
    print("  ours  :", np.round(coefs([a[n] for n in common]), 3))
    print("  theirs:", np.round(coefs([b[n] for n in common]), 3))


if __name__ == "__main__":
    main(*sys.argv[1:])
