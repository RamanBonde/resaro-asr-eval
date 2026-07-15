"""Step 1: Build the evaluation dataset.

Downloads English clips from Mozilla Common Voice, filters them into
accent groups, applies basic data-quality checks, resamples to 16 kHz
mono, and generates noisy variants at fixed SNR levels.

Output: data/audio/*.wav + data/manifest.csv with columns:
    clip_id, path, reference_text, accent_group, noise_condition

NOTE: Common Voice on Hugging Face requires (free) account + accepting
the dataset terms once on its page, then `huggingface-cli login`.
"""

import os
import random
import numpy as np
import pandas as pd
import soundfile as sf
from datasets import load_dataset, Audio

from config import (SEED, SAMPLE_RATE, ACCENT_GROUPS, CLIPS_PER_GROUP,
                    NOISE_CONDITIONS, AUDIO_DIR, MANIFEST)

PRINT_ACCENTS = False   # set True on first run to inspect real accent labels

# --- Data quality checks (report these as your dataset QC criteria) ----
MIN_DURATION_S = 2.0    # too-short clips make WER unstable
MAX_DURATION_S = 12.0   # keeps runtime bounded
MIN_WORDS = 3           # reference must have enough words to grade


def quality_ok(audio_array: np.ndarray, sr: int, text: str) -> bool:
    """Return True if the clip passes basic quality checks."""
    duration = len(audio_array) / sr
    if not (MIN_DURATION_S <= duration <= MAX_DURATION_S):
        return False
    if text is None or len(text.split()) < MIN_WORDS:
        return False
    if np.max(np.abs(audio_array)) < 1e-4:   # near-silent / broken clip
        return False
    return True


def match_group(accent_field: str) -> str | None:
    """Map Common Voice's free-text accent field to one of our groups."""
    if not accent_field:
        return None
    for group, substrings in ACCENT_GROUPS.items():
        if any(s.lower() in accent_field.lower() for s in substrings):
            return group
    return None


def mix_noise(clean: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white Gaussian noise at a target SNR (dB).

    SNR = 10*log10(P_signal / P_noise). We scale unit-variance noise so
    that the ratio of signal power to noise power hits the target.
    """
    signal_power = np.mean(clean ** 2)
    noise = rng.standard_normal(len(clean))
    noise_power = np.mean(noise ** 2)
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / noise_power)
    noisy = clean + noise
    # prevent clipping
    peak = np.max(np.abs(noisy))
    if peak > 1.0:
        noisy = noisy / peak
    return noisy.astype(np.float32)


def main():
    random.seed(SEED)
    rng = np.random.default_rng(SEED)
    os.makedirs(AUDIO_DIR, exist_ok=True)

    # Streaming avoids downloading the full (huge) dataset.
    ds = load_dataset("mozilla-foundation/common_voice_17_0", "en",
                      split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    collected = {g: [] for g in ACCENT_GROUPS}
    seen_accents = set()

    for ex in ds:
        accent_field = ex.get("accents") or ex.get("accent") or ""
        seen_accents.add(accent_field)
        group = match_group(accent_field)
        if group is None or len(collected[group]) >= CLIPS_PER_GROUP:
            if all(len(v) >= CLIPS_PER_GROUP for v in collected.values()):
                break
            continue
        audio = ex["audio"]["array"].astype(np.float32)
        text = (ex.get("sentence") or "").strip()
        if quality_ok(audio, SAMPLE_RATE, text):
            collected[group].append((audio, text))

    if PRINT_ACCENTS:
        print("Accent labels seen:", sorted(a for a in seen_accents if a)[:100])

    # Save clean + noisy variants, build manifest
    rows = []
    for group, clips in collected.items():
        print(f"{group}: {len(clips)} clips collected")
        for i, (audio, text) in enumerate(clips):
            for cond, snr in NOISE_CONDITIONS.items():
                clip_id = f"{group}_{i:03d}_{cond}"
                path = os.path.join(AUDIO_DIR, clip_id + ".wav")
                out = audio if snr is None else mix_noise(audio, snr, rng)
                sf.write(path, out, SAMPLE_RATE)
                rows.append({"clip_id": clip_id, "path": path,
                             "reference_text": text,
                             "accent_group": group,
                             "noise_condition": cond})

    pd.DataFrame(rows).to_csv(MANIFEST, index=False)
    print(f"Wrote {len(rows)} rows to {MANIFEST}")


if __name__ == "__main__":
    main()
