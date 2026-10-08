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

## perception_gap.py (2026-10-08)

Question: how wrong would a purely local picture of the group be, and how checkable is a world model that a neighbour
shares? Qwen3.5-9B, all released episodes.

| Topic | Geometry | Mean degree | Overlap | Mean \|local − global\| share | Majority illusion | Illusion among minority-side agents |
|---|---|---|---|---|---|---|
| maths | random signed | 3.5 | 0.12 | 0.13 | 9 % | 19 % |
| maths | square lattice | 3.2 | 0.00 | 0.14 | 11 % | 17 % |
| maths | triangular lattice | 4.6 | 0.47 | 0.09 | 6 % | 43 % |
| politics | random signed | 3.5 | 0.12 | 0.19 | 16 % | 27 % |
| politics | square lattice | 3.2 | 0.00 | 0.17 | 14 % | 28 % |
| politics | triangular lattice | 4.6 | 0.47 | 0.15 | 11 % | 50 % |

- **Overlap:** for an edge i → j, the share of j's other neighbours that i also sees directly. It measures how much of
  a shared world model the receiver can check against its own inbox.
- **Majority illusion:** the agent's neighbourhood majority differs from the community majority.

Readings:
- A local view misjudges the group by 13–19 points on average, and is on the wrong side 6–16 % of the time.
- In the triangular lattice, minorities form domains: about half of the minority-side agents see a local majority on
  their own side.
- The three geometries span verifiability from none (square, 0.00) to high (triangular, 0.47), with random signed
  graphs in between (0.12).
