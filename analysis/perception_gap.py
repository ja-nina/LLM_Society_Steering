"""How wrong is a purely local world model, and how checkable is a shared one? (offline, released episodes)

For every agent update in the released episodes:
- local view: the stance share among the agent's neighbours (what its inbox shows), friendly and unfriendly ties alike;
- global truth: the stance share in the whole community at the same round.
Majority illusion = the local majority differs from the global majority (ties excluded).

Verifiability of a shared world model, per geometry: for an edge i -> j, the share of j's neighbours (other than i)
that i also observes directly. A claim by j about those agents can be checked by i against its own inbox.

    python -m analysis.perception_gap data/models/qwen3.5-9b
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def geometry(name: str) -> str:
    gid = name.split("__")[1]
    return gid if gid in ("square", "triangular") else "random"


def overlap(J: np.ndarray) -> float:
    A = J != 0
    vals = []
    for i, j in zip(*np.nonzero(A)):
        nj = A[j].copy()
        nj[i] = False
        if nj.sum():
            vals.append((A[i] & nj).sum() / nj.sum())
    return float(np.mean(vals))


def main(root: str) -> None:
    stats = defaultdict(lambda: defaultdict(list))
    for p in sorted(Path(root).rglob("*__rep*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        J = np.array(r["J"])
        A = J != 0
        S = np.array(r["spins_history"])
        key = (r["mode"], geometry(p.name))
        st = stats[key]
        st["overlap"].append(overlap(J))
        st["degree"].append(A.sum(1).mean())
        for t in range(1, S.shape[0]):
            s = S[t - 1]
            glob = (s > 0).mean()
            g_major = np.sign(glob - 0.5)
            for i in range(len(s)):
                nb = A[i] & (s != 0)
                if not nb.any():
                    continue
                loc = (s[nb] > 0).mean()
                st["abs_err"].append(abs(loc - glob))
                l_major = np.sign(loc - 0.5)
                if l_major != 0 and g_major != 0:
                    st["illusion"].append(l_major != g_major)
                    # minority-side agents: the global majority is against them
                    st["illusion_minority"].append(l_major != g_major) if s[i] != g_major else None
    print(f"{'mode':<11}{'geometry':<12}{'deg':>5}{'overlap':>9}{'|local-global|':>16}{'illusion':>10}{'illusion|minority':>19}")
    for (mode, geo), st in sorted(stats.items()):
        print(f"{mode:<11}{geo:<12}{np.mean(st['degree']):>5.1f}{np.mean(st['overlap']):>9.2f}"
              f"{np.mean(st['abs_err']):>16.3f}{np.mean(st['illusion']):>10.3f}{np.mean(st['illusion_minority']):>19.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
