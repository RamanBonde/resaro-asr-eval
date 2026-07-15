# ASR Robustness Evaluation — Accents × Noise

Evaluates two speech recognition models (Whisper-base, wav2vec2-base-960h)
on English speech across 3 accent groups × 3 noise conditions, reporting
WER/CER per cell plus qualitative failure analysis.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Common Voice requires a free Hugging Face account:
# 1. Accept terms at https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0
# 2. huggingface-cli login
```

## Run (from repo root)

```bash
python src/prepare_data.py   # download, filter, QC, add noise  (~10-20 min)
python src/run_eval.py       # both models over all clips       (CPU: ~30-60 min; resumable)
python src/analyze.py        # WER/CER table, heatmaps, worst failures
```

Outputs land in `results/`: `wer_by_condition.csv`, `heatmap_*.png`,
`worst_failures.csv`, and raw `predictions.csv`.

## Design decisions (details in report)

- 60 clips/group, seed=42, duration 2–12 s, ≥3 words (data quality checks)
- White-noise mixing at SNR 10 dB / 0 dB (controlled, reproducible severity)
- Shared text normalization before WER (wav2vec2 has no casing/punctuation)
- Empty model output scored as WER=1.0

## TODO before submission (AIP integration)

- [ ] Register dataset + models via AIP SDK per its docs (wrap `models.py` classes)
- [ ] Register WER as custom metric on platform if missing
- [ ] Keep friction log -> report section "platform feedback"
