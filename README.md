# ASR Robustness Evaluation on AIP v2 — Accents × Noise

Compares two speech-recognition models (whisper-base, wav2vec2-base-960h) on
English speech across 3 accent groups × 3 noise conditions, reporting WER/CER
per cell — and drives the evaluation through Resaro's AIP v2 platform.

## Setup

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Common Voice (community mirror; Mozilla removed the original from HF):
# requires datasets==3.6.0 (pinned in requirements.txt) and a free HF account:
huggingface-cli login

# AIP platform credentials (for steps 4–5): copy .env.example to .env and
# fill AIP_BASE_URL, AIP_API_KEY, AIP_WORKSPACE_NAME from your credentials
# email. Also install the AIP SDK wheels: pip install path/to/wheels/*.whl
```

## Run (from repo root)

```bash
python src/prepare_data.py        # 1. download, filter, add noise -> data/manifest.csv  (~10-20 min)
python src/run_eval.py            # 2. both models over all clips  (CPU ~30-60 min; resumable)
python src/split_predictions.py   # 3. per-model prediction CSVs for the platform step
python src/run_asr_eval.py prep   # 4. upload -> quality checks -> promote golden (AIP)
python src/run_asr_eval.py score  # 5. external runs on AIP + canonical WER/CER
python src/run_asr_eval.py compare#    cross-model stratified summary
python src/analyze.py             # 6. per-cell table, heatmaps, worst failures
```

**Report numbers come from `run_asr_eval.py`** (`outputs/comparison_summary.csv`);
`analyze.py` produces the presentation artifacts (`results/wer_by_condition.csv`,
`heatmap_*.png`, `worst_failures.csv`). Steps 4–5 require platform credentials;
steps 1–3 and 6 run fully offline.

## Design decisions (details in report)

- 10 clips/group × 3 accents × 3 noise conditions = 90 rows; seed=42;
  duration 2–12 s; ≥3 reference words; non-silent (local QC)
- Accent mapping (Common Voice free-text): "United States English",
  "India and South Asia", "German"
- White-noise mixing at SNR 10 dB / 0 dB (seeded, reproducible)
- Shared text normalization before WER/CER (wav2vec2 outputs uppercase,
  unpunctuated text; raw comparison would punish formatting, not recognition)
- Empty model output scored as WER = 1.0
- AIP runs are created **without explicit metrics**: requesting metrics at
  creation fails required-column validation in external mode (report, Gap 3)

## Evidence

`logs/` contains terminal sessions for the quality-check PASS, golden
promotion, the 422 rejection with explicit metrics, and the successful runs.
