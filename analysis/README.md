# Offline analyses on the released Physics-of-Agents data (no GPU)

## consensus_claims.py (2026-10-08)

Question: in the released Qwen3.5-9B episodes (2,400 episodes, 614,400 agent updates), do messages that claim what
the group believes move recipients beyond the stance they state?

Result:

- Genuine consensus claims ("most of us agree", "the majority believes") essentially do not occur. The 11.5% regex
  hits are almost all false positives ("... for everyone", "we all"). Agents have no memory and only see their
  neighbours, so they have no basis for a picture of the group.
- Adding the claim features changes held-out log-loss by 0.0001 (objective 0.2277 -> 0.2276; subjective
  0.3138 -> 0.3137).

Consequence: the perceived-group channel is absent from the original protocol. It must be created (agents keeping and
sharing world models) and intervened on. Perceived vs actual pressure cannot be separated on this data.

Side finding (logistic fit of s_i(t) on own previous state, friendly drive and unfriendly drive):

| Topic | Self | Friendly drive | Unfriendly drive |
|---|---|---|---|
| maths (objective) | 0.35 | 1.64 | 0.60 |
| politics (subjective) | 0.69 | 1.46 | 0.00 |

Agents still move toward unfriendly neighbours on maths. On political statements, unfriendly ties have no effect.
