"""Central configuration — every experimental choice lives here, so the
report can reference one file and reviewers see all decisions at a glance."""

SEED = 42                      # fixed seed -> reproducible sampling
SAMPLE_RATE = 16_000           # both Whisper and wav2vec2 expect 16 kHz mono

# --- Accent groups ------------------------------------------------------
# Common Voice stores accents as free text. We match by substring.
# IMPORTANT: run prepare_data.py once with PRINT_ACCENTS=True to see the
# actual accent strings in the dataset version you download, then adjust
# these substrings if needed. Document the final mapping in the report.
ACCENT_GROUPS = {
    "us":     ["United States English"],
    "india":  ["India and South Asia"],
    "german": ["German"],
}
CLIPS_PER_GROUP = 60           # small n is fine per the briefing; state it as a limitation

# --- Noise conditions ---------------------------------------------------
# SNR in dB: higher = cleaner. 'clean' = original audio, no mixing.
NOISE_CONDITIONS = {
    "clean":  None,
    "snr10":  10,              # moderate background noise
    "snr0":   0,               # noise as loud as the speech
}

# --- Models -------------------------------------------------------------
WHISPER_SIZE = "base"                              # ~74M params, laptop-friendly
WAV2VEC2_NAME = "facebook/wav2vec2-base-960h"      # classic CTC baseline

# --- Paths --------------------------------------------------------------
DATA_DIR = "data"
AUDIO_DIR = f"{DATA_DIR}/audio"
MANIFEST = f"{DATA_DIR}/manifest.csv"
PREDICTIONS = "results/predictions.csv"
RESULTS_TABLE = "results/wer_by_condition.csv"
FAILURES = "results/worst_failures.csv"
