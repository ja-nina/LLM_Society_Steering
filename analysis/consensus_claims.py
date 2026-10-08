"""Offline test on the released Physics-of-Agents data (no GPU): do messages that claim what the group believes
("most of us agree", "the consensus is", "everyone") move recipients beyond the stance they state?

Per (episode, agent i, step t >= 1), predict s_i(t) from:
  - base:  s_i(t-1), friendly drive sum_j J+_ji s_j(t-1), unfriendly drive sum_j J-_ji s_j(t-1)  (their model's terms)
  - +claims: friendly / unfriendly drives restricted to messages containing a consensus claim
             (sign = sender stance s_j(t-1), which is what the message argues for)
Logistic regression, cross-validated by episode; report held-out log-loss and the claim coefficients.

    uv run --with numpy --with scikit-learn python analysis/consensus_claims.py data/models/qwen3.5-9b
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

CLAIM = re.compile(r"\b(most of us|most (people|agents|colleagues|experts|members|others)|majority|consensus|"
                   r"everyone|everybody|we all|all of us|the (whole )?group|the panel|widely (agreed|accepted)|"
                   r"(nearly|almost) all|\d{1,3}\s?%)\b", re.I)


def rows(path: Path):
    r = json.loads(path.read_text(encoding="utf-8"))
    J = np.array(r["J"])
    S = np.array(r["spins_history"])  # (T+1, N)
    for t in range(1, S.shape[0]):
        msgs = r["messages_history"][t - 1]  # step t's messages, written from s(t-1); key "sender,receiver"
        prev = S[t - 1]
        for i in range(S.shape[1]):
            if S[t, i] == 0:
                continue
            fr = un = fr_c = un_c = 0.0
            n_claims = 0
            for k, m in msgs.items():
                snd, rcv = map(int, k.split(","))
                if rcv != i:
                    continue
                s, w = prev[snd], J[snd, i]
                claim = bool(CLAIM.search(m))
                n_claims += claim
                if w > 0:
                    fr += s
                    fr_c += s * claim
                elif w < 0:
                    un += s
                    un_c += s * claim
            yield (path.name, r["mode"], int(S[t, i] > 0), prev[i], fr, un, fr_c, un_c, n_claims)


def main(root: str) -> None:
    data = [x for p in sorted(Path(root).rglob("*__rep*.json")) for x in rows(p)]
    groups = np.array([d[0] for d in data])
    modes = np.array([d[1] for d in data])
    y = np.array([d[2] for d in data])
    X = np.array([d[3:8] for d in data], dtype=float)
    n_claims = np.array([d[8] for d in data])
    print(f"{len(y)} updates from {len(set(groups))} episodes; "
          f"{(n_claims > 0).mean():.1%} of updates had >= 1 consensus-claim message in the inbox")
    for mode in ("objective", "subjective"):
        m = modes == mode
        Xm, ym, gm = X[m], y[m], groups[m]
        res = {}
        for name, cols in (("base", [0, 1, 2]), ("+claims", [0, 1, 2, 3, 4])):
            losses, coefs = [], []
            for tr, te in GroupKFold(n_splits=5).split(Xm, ym, gm):
                clf = LogisticRegression(max_iter=1000).fit(Xm[tr][:, cols], ym[tr])
                losses.append(log_loss(ym[te], clf.predict_proba(Xm[te][:, cols])[:, 1], labels=[0, 1]))
                coefs.append(clf.coef_[0])
            res[name] = (np.mean(losses), np.mean(coefs, axis=0))
        b, c = res["base"], res["+claims"]
        print(f"\n[{mode}] n={m.sum()}  held-out log-loss base={b[0]:.4f}  +claims={c[0]:.4f}  "
              f"(diff {b[0] - c[0]:+.4f})")
        print(f"  base coefs   [self, friendly, unfriendly] = {np.round(b[1], 3)}")
        print(f"  +claims coefs [self, friendly, unfriendly, friendly_claim, unfriendly_claim] = {np.round(c[1], 3)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/models/qwen3.5-9b")
