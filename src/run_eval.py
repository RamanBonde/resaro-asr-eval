"""Step 3: Run both models over every clip in the manifest.

The plan:
  1. Read the manifest (the list of clips that step 1 created).
  2. For each model, transcribe every clip.
  3. Save all predictions to results/predictions.csv.

Why save to a file? Running the models is SLOW and
expensive. Computing WER and making tables ("analysis") is fast and
cheap. By saving predictions to disk, you can re-run the analysis as
many times as you want without re-running the models.

The script resumes automatically. If it crashes or you stop it
halfway, just run it again — it skips everything already on disk.
"""

import os
import pandas as pd

from config import MANIFEST, PREDICTIONS
from models import WhisperWrapper, Wav2Vec2Wrapper


def main():
    manifest = pd.read_csv(MANIFEST)
    os.makedirs("results", exist_ok=True)

    # ---- Resume support -------------------------------------------------
    # 'done' will remember which (clip, model) pairs are already finished.
    # A set is used because checking "is this in the set?" is very fast.
    done = set()

    # 'rows' will hold every prediction as a dictionary. We start with
    # whatever is already on disk (if anything) and append new ones.
    rows = []

    if os.path.exists(PREDICTIONS):
        # A previous run left a predictions file. Load it.
        prev = pd.read_csv(PREDICTIONS)

        # Build the set of finished (clip_id, model) pairs.
        # zip pairs up the two columns row by row:
        #   clip_ids:  ["us_000_clean", "us_001_clean", ...]
        #   models:    ["whisper-base", "whisper-base", ...]
        #   zipped:    [("us_000_clean", "whisper-base"), ...]
        for pair in zip(prev["clip_id"], prev["model"]):
            done.add(pair)

        # Convert the old table back into a list of dictionaries,
        # so we can keep appending to it.
        rows = prev.to_dict("records")

        print(f"Resuming: {len(done)} predictions already on disk")

    # ---- Run the models --------------------------------------------------
    # Create one wrapper object for each model (this loads them into memory).
    models = [WhisperWrapper(), Wav2Vec2Wrapper()]

    for model in models:
        # iterrows() gives us the manifest one row at a time.
        # Each 'r' behaves like a dictionary: r["clip_id"], r["path"], etc.
        # (The underscore _ is the row number, which we don't need.)
        for _, r in manifest.iterrows():

            # Skip this clip if this model already transcribed it.
            key = (r["clip_id"], model.name)
            if key in done:
                continue

            # Transcribe the clip. If the model crashes on one weird
            # file, we log the error and keep going instead of losing
            # the whole run. The empty string will show up as a very
            # bad WER for that clip, which is honest.
            try:
                hyp = model.predict(r["path"])
            except Exception as e:
                print(f"FAILED {key}: {e}")
                hyp = ""

            # Store the prediction together with the clip's metadata,
            # so the analysis step has everything it needs in one file.
            # ("hypothesis" is the standard ASR term for the model's
            # output text, vs. the "reference" which is the true text.)
            rows.append({
                "clip_id": r["clip_id"],
                "model": model.name,
                "accent_group": r["accent_group"],
                "noise_condition": r["noise_condition"],
                "reference_text": r["reference_text"],
                "hypothesis_text": hyp,
            })

            # Checkpoint: every 10 predictions, save everything to disk.
            # (% is the remainder operator: len(rows) % 10 == 0 is True
            # when the count is 10, 20, 30, ...)
            if len(rows) % 10 == 0:
                pd.DataFrame(rows).to_csv(PREDICTIONS, index=False)
                print(f"{len(rows)} predictions saved...")

    # ---- Final save -------------------------------------------------------
    pd.DataFrame(rows).to_csv(PREDICTIONS, index=False)
    print(f"Done: {len(rows)} predictions -> {PREDICTIONS}")


if __name__ == "__main__":
    main()