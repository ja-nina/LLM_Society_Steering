# Running the Physics-of-Agents generation on our cluster (Qwen3.5-9B on vLLM)

The generation code in `lib/datagen` runs **unchanged**. Two pieces connect it to our cluster:

- **The server:** `swmp submit` from the sibling repo `Social_World_Models_Personas` starts a vLLM server for one SLURM
  job, using the backend `configs/backend/vllm_qwen35_9b_l40.yaml` (Qwen/Qwen3.5-9B, thinking off), and runs the
  command after `--`.
- **The connection:** `slurm/with_endpoint.sh` reads the server URL from `$SWMP_ENDPOINTS`, exports
  `OPENAI_BASE_URL` (their OpenAI client reads it), changes into this repo, and runs the given command.

## Setup (once)

```bash
cd /BS/ForNet/work
git clone <your private repo url> LLM_Society_Steering && cd LLM_Society_Steering && git checkout steering
cd ../Social_World_Models_Personas && git pull && source .env.cluster
PY=$PWD/.venv/bin/python; S=/BS/ForNet/work/LLM_Society_Steering; BK=configs/backend/vllm_qwen35_9b_l40.yaml
```

## 1. Smoke run (about 10 minutes; checks the server, thinking off and parsing)

```bash
uv run swmp submit --backend $BK --time 01:00:00 -- bash $S/slurm/with_endpoint.sh \
  $PY -m lib.datagen.collect_energy --backend openai --model Qwen/Qwen3.5-9B --mode objective --limit 1 \
  --num-train-graphs 1 --num-seen-graphs 1 --num-fresh-graphs 0 --no-lattices --trajectories 1 --num-steps 2 --k 3 \
  --output-dir $SWMP_OUT/society/smoke
```

**Check:**
- `slurm-logs/swmp-<job>.out` ends with `[collect] done`;
- the episode JSONs in `$SWMP_OUT/society/smoke` have `spins_raw_history` entries of ±1 (0 means an unparsed answer);
- `messages_history` contains two-sentence messages, not reasoning text.

## 2. Reproduction slice (paired with their released Qwen3.5-9B episodes)

Their default graph bank (8 random graphs + 2 lattices; 4 trajectories) with the first 2 train and 2 test questions:
160 episodes per mode, roughly 270k calls.

```bash
for MODE in objective subjective; do
  uv run swmp submit --backend $BK --time 08:00:00 -- bash $S/slurm/with_endpoint.sh \
    $PY -m lib.datagen.collect_energy --backend openai --model Qwen/Qwen3.5-9B --mode $MODE --limit 2 \
    --output-dir $SWMP_OUT/society/repro/qwen3.5-9b/${MODE}_energy
done
```

## 3. Compare with their runs (login node, no GPU)

```bash
cd $S
for MODE in objective subjective; do
  uv run --no-project --with numpy --with scikit-learn python analysis/compare_repro.py \
    $SWMP_OUT/society/repro/qwen3.5-9b/${MODE}_energy data/models/qwen3.5-9b/${MODE}_energy
done
```

**Reproduction passes if:**
- the conviction, |net opinion| and accuracy curves are within the run-to-run spread;
- the final-round differences have CIs that include 0 or are small;
- the influence coefficients are of similar size.

Differences in serving (vLLM vs Together, sampling details) can cause small shifts; record them in this README.
