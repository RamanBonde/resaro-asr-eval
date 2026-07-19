"""Build the evaluation dataset.

What this script does, in plain words:
  1. Downloads English audio clips from Mozilla Common Voice (streaming,
     so we don't download the whole huge dataset).
  2. Sorts the clips into accent groups (defined in config.py).
  3. Throws away bad clips (too short, too long, silent, too few words).
  4. Saves each clip as a clean version + noisy versions at fixed SNR levels.
  5. Writes a manifest.csv listing every saved file.

Output: data/audio/*.wav + data/manifest.csv with columns:
    clip_id, path, reference_text, accent_group, noise_condition

NOTE: We use the community mirror "fsicoli/common_voice_17_0" because
Mozilla removed Common Voice from Hugging Face in October 2025 (it moved
to the Mozilla Data Collective). Requires datasets==3.6.0, because newer
versions of the datasets library no longer run script-based datasets.
"""

import os
import random
import numpy as np
import pandas as pd
import soundfile as sf
from datasets import load_dataset, Audio, Features, Value

from config import (SEED, SAMPLE_RATE, ACCENT_GROUPS, CLIPS_PER_GROUP,
                    NOISE_CONDITIONS, AUDIO_DIR, MANIFEST)

PRINT_ACCENTS = False   # set True on first run to see the real accent labels

# --- Data quality rules (report these as your dataset QC criteria) ------
MIN_DURATION_S = 2.0    # too-short clips make WER unstable
MAX_DURATION_S = 12.0   # keeps runtime bounded
MIN_WORDS = 3           # reference text must have enough words to grade


def quality_ok(audio_array, sr, text):
    """Return True if the clip passes all the basic quality checks."""

    # Check 1: duration must be between MIN and MAX seconds.
    # Number of samples divided by sample rate = length in seconds.
    duration = len(audio_array) / sr
    if duration < MIN_DURATION_S:
        return False
    if duration > MAX_DURATION_S:
        return False

    # Check 2: the reference text must exist and have enough words.
    if text is None:
        return False
    words = text.split()
    if len(words) < MIN_WORDS:
        return False

    # Check 3: the audio must not be silent or broken.
    # If the loudest sample is basically zero, the clip is useless.
    loudest_sample = np.max(np.abs(audio_array))
    if loudest_sample < 0.0001:
        return False

    # All checks passed.
    return True


def match_group(accent_field):
    """Map Common Voice's free-text accent field to one of our groups.

    Example: if ACCENT_GROUPS = {"us": ["united states", "american"]},
    then an accent field like "American English" returns "us".
    Returns None if the accent does not match any group.
    """
    if not accent_field:
        return None

    accent_lower = accent_field.lower()

    # Go through every group and every keyword for that group.
    for group in ACCENT_GROUPS:
        keywords = ACCENT_GROUPS[group]
        for keyword in keywords:
            if keyword.lower() in accent_lower:
                return group

    # No keyword matched.
    return None


def all_groups_full(collected):
    """Return True if every accent group already has enough clips."""
    for group in collected:
        if len(collected[group]) < CLIPS_PER_GROUP:
            return False
    return True


def mix_noise(clean, snr_db, rng):
    """Add white Gaussian noise to a clip at a target SNR (in dB).

    SNR (signal-to-noise ratio) formula:
        SNR = 10 * log10(signal_power / noise_power)

    Plan: generate random noise, then scale it so the ratio between the
    signal power and the noise power hits the target SNR.
    """
    # Step 1: measure the power of the clean signal (power = mean of squares).
    signal_power = np.mean(clean ** 2)

    # Step 2: generate random noise, same length as the audio.
    noise = rng.standard_normal(len(clean))
    noise_power = np.mean(noise ** 2)

    # Step 3: figure out how much noise power we WANT.
    # Rearranging the SNR formula gives:
    #     noise_power = signal_power / 10^(SNR/10)
    target_noise_power = signal_power / (10 ** (snr_db / 10))

    # Step 4: scale the noise to have exactly that power.
    scale = np.sqrt(target_noise_power / noise_power)
    noise = noise * scale

    # Step 5: add the noise to the clean audio.
    noisy = clean + noise

    # Step 6: prevent clipping. WAV audio must stay between -1.0 and 1.0.
    # If we went over, shrink the whole clip back down.
    peak = np.max(np.abs(noisy))
    if peak > 1.0:
        noisy = noisy / peak

    return noisy.astype(np.float32)


def main():
    # Make results reproducible: same seed -> same random noise every run.
    random.seed(SEED)
    rng = np.random.default_rng(SEED)

    # Create the output folder if it doesn't exist yet.
    os.makedirs(AUDIO_DIR, exist_ok=True)

    # Load the dataset in streaming mode: clips arrive one at a time
    # instead of downloading everything first.
    ds = load_dataset("fsicoli/common_voice_17_0", "en",
                      split="test", streaming=True, trust_remote_code=True)

    # FIX for the CastError: the mirror's loader script declares one set
    # of columns, but the actual data files have a slightly different one
    # (extra columns like sentence_id / sentence_domain, and vote counts
    # stored as strings). Here we declare the TRUE schema ourselves —
    # copied straight from the "Couldn't cast <this>" part of the error.
    #
    # The Audio(...) entry also handles resampling every clip to our
    # sample rate (16 kHz), so no separate cast_column call is needed.
    real_features = Features({
        "client_id":       Value("string"),
        "path":            Value("string"),
        "sentence_id":     Value("string"),
        "sentence":        Value("string"),
        "sentence_domain": Value("string"),
        "up_votes":        Value("string"),
        "down_votes":      Value("string"),
        "age":             Value("string"),
        "gender":          Value("string"),
        "variant":         Value("string"),
        "locale":          Value("string"),
        "segment":         Value("string"),
        "accent":          Value("string"),
        "audio":           Audio(sampling_rate=SAMPLE_RATE),
    })
    ds = ds.cast(real_features)

    # 'collected' holds the clips we keep, one list per accent group.
    # Example: {"us": [], "uk": [], "india": []}
    collected = {}
    for group in ACCENT_GROUPS:
        collected[group] = []

    # Keep track of every accent label we see (useful for debugging).
    seen_accents = set()

    # ---- Main collection loop: go through clips one by one -------------
    for ex in ds:
        # The accent might be stored under "accents" or "accent",
        # depending on the dataset version. Try both.
        accent_field = ex.get("accents")
        if not accent_field:
            accent_field = ex.get("accent")
        if not accent_field:
            accent_field = ""

        seen_accents.add(accent_field)

        # Which of our groups does this clip belong to (if any)?
        group = match_group(accent_field)

        # Skip this clip if it doesn't match a group,
        # or if that group is already full.
        if group is None or len(collected[group]) >= CLIPS_PER_GROUP:
            # If EVERY group is full, we are done collecting.
            if all_groups_full(collected):
                break
            continue  # otherwise move on to the next clip

        # Get the audio samples and the reference sentence.
        audio = ex["audio"]["array"].astype(np.float32)
        text = ex.get("sentence")
        if text is None:
            text = ""
        text = text.strip()

        # Keep the clip only if it passes the quality checks.
        if quality_ok(audio, SAMPLE_RATE, text):
            collected[group].append((audio, text))

    if PRINT_ACCENTS:
        # Show (up to 100) accent labels so you can tune ACCENT_GROUPS.
        labels = sorted(a for a in seen_accents if a)
        print("Accent labels seen:", labels[:100])

    # ---- Save clean + noisy versions, and build the manifest -----------
    rows = []

    for group in collected:
        clips = collected[group]
        print(f"{group}: {len(clips)} clips collected")

        for i, (audio, text) in enumerate(clips):
            # For every clip, save one file per noise condition.
            # NOISE_CONDITIONS looks like: {"clean": None, "snr10": 10, ...}
            for cond in NOISE_CONDITIONS:
                snr = NOISE_CONDITIONS[cond]

                # Build a unique id like "us_007_clean".
                # {i:03d} means: the number padded to 3 digits (7 -> "007").
                clip_id = f"{group}_{i:03d}_{cond}"
                path = os.path.join(AUDIO_DIR, clip_id + ".wav")

                # snr is None for the clean condition -> save as-is.
                if snr is None:
                    out = audio
                else:
                    out = mix_noise(audio, snr, rng)

                sf.write(path, out, SAMPLE_RATE)

                # Remember this file for the manifest.
                rows.append({
                    "clip_id": clip_id,
                    "path": path,
                    "reference_text": text,
                    "accent_group": group,
                    "noise_condition": cond,
                })

    # Write the manifest CSV that the next steps will read.
    pd.DataFrame(rows).to_csv(MANIFEST, index=False)
    print(f"Wrote {len(rows)} rows to {MANIFEST}")


if __name__ == "__main__":
    main()