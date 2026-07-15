"""Step 3: Run both models over every clip in the manifest.

Predictions are saved to results/predictions.csv so the (cheap) analysis
can be re-run any number of times without re-doing (expensive) inference.
Resumes automatically if interrupted.
"""

import os
import pandas as pd

from config import MANIFEST, PREDICTIONS
from models import WhisperWrapper, Wav2Vec2Wrapper


def main():
    manifest = pd.read_csv(MANIFEST)
    os.makedirs("results", exist_ok=True)

    done = set()
    if os.path.exists(PREDICTIONS):
        prev = pd.read_csv(PREDICTIONS)
        done = set(zip(prev["clip_id"], prev["model"]))
        rows = prev.to_dict("records")
        print(f"Resuming: {len(done)} predictions already on disk")
    else:
        rows = []

    models = [WhisperWrapper(), Wav2Vec2Wrapper()]

    for model in models:
        for _, r in manifest.iterrows():
            key = (r["clip_id"], model.name)
            if key in done:
                continue
            try:
                hyp = model.predict(r["path"])
            except Exception as e:                     # log, don't crash the run
                print(f"FAILED {key}: {e}")
                hyp = ""
            rows.append({"clip_id": r["clip_id"], "model": model.name,
                         "accent_group": r["accent_group"],
                         "noise_condition": r["noise_condition"],
                         "reference_text": r["reference_text"],
                         "hypothesis_text": hyp})
            if len(rows) % 25 == 0:                    # periodic checkpoint
                pd.DataFrame(rows).to_csv(PREDICTIONS, index=False)
                print(f"{len(rows)} predictions saved...")

    pd.DataFrame(rows).to_csv(PREDICTIONS, index=False)
    print(f"Done: {len(rows)} predictions -> {PREDICTIONS}")


if __name__ == "__main__":
    main()
