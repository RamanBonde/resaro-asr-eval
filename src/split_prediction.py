"""
split_predictions.py — reshape your one-file predictions into the per-model
CSVs that run_asr_eval.py expects.

Input : predictions.csv  (long format, columns:
          clip_id, model, accent_group, noise_condition,
          reference_text, hypothesis_text)
Output: predictions/<model>.csv  per model, columns: clip_id, hypothesis

Run:    python split_predictions.py
"""
import os
import pandas as pd

SRC = "results/predictions.csv"
OUT_DIR = "results/predictions"

def main():
    df = pd.read_csv(SRC)
    os.makedirs(OUT_DIR, exist_ok=True)

    for model, g in df.groupby("model"):
        out = g[["clip_id", "hypothesis_text"]].rename(
            columns={"hypothesis_text": "hypothesis"}
        )
        # keep empty hypotheses as empty strings (they are real failures,
        # e.g. wav2vec2 at snr0 — scored as WER ~1.0, which is correct)
        out["hypothesis"] = out["hypothesis"].fillna("")
        path = os.path.join(OUT_DIR, f"{model}.csv")
        out.to_csv(path, index=False)
        print(f"wrote {path}  ({len(out)} rows)")

if __name__ == "__main__":
    main()