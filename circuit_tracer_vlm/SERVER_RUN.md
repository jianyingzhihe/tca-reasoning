# Server Quick Start

This folder contains a minimal server workflow so you can run attribution on a Linux GPU server.

## 1) Prepare env file

```bash
cd circuit_tracer_vlm
cp .env.example .env
```

Edit `.env`:

```env
HF_TOKEN=hf_xxx
HF_HOME=/data/hf_cache
HUGGINGFACE_HUB_CACHE=/data/hf_cache/hub
```

## 2) Create venv and install

```bash
bash scripts/server/setup_env.sh .venv
source .venv/bin/activate
```

## 3) Load env and verify access

```bash
source scripts/server/load_env.sh .env
python scripts/server/check_hf_access.py
```

You should see two `[OK]` lines for:
- `google/gemma-3-4b-it`
- `tianhux2/gemma3-4b-it-plt`

## 4) Run smoke attribution

```bash
bash scripts/server/run_gemma_smoke.sh demos/img/gemma/213.png ./outputs/gemma_demo_213.pt
```

If it succeeds, you can then increase:
- `--max_feature_nodes`
- `--batch_size`
- graph export options (`--slug`, `--graph_file_dir`, `--server`)

## Notes

- This project uses a local TransformerLens fork. Keep the repo layout unchanged.
- If your server disk is small, set `HF_HOME` to a large mount before downloading model/transcoder files.
- The first run may take a long time due to model and transcoder download.

## Daily use (avoid typing many commands)

After each SSH login, run one command:

```bash
source scripts/server/dev.sh
```

This command will:
- auto-create `.venv` if missing
- activate `.venv`
- load `.env`

Then you can run:

```bash
bash scripts/server/run_gemma_smoke.sh
```

Or one command for full check + smoke:

```bash
bash scripts/server/smoke_one_command.sh
```
