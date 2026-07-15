"""Step 4: Grade and slice.

Computes WER and CER per (model x accent_group x noise_condition) cell,
writes the results table, renders a heatmap per model, and extracts the
worst failures for qualitative analysis.

Text normalization matters: wav2vec2 outputs UPPERCASE without
punctuation, Whisper outputs cased+punctuated text. Without a shared
normalization, WER would punish formatting, not recognition. State this
in the report — it is a real evaluation-design decision.
"""

import jiwer
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PREDICTIONS, RESULTS_TABLE, FAILURES

# One normalization applied to BOTH references and hypotheses.
NORMALIZE = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


def wer_of(ref: str, hyp: str) -> float:
    if not hyp:
        return 1.0                      # empty output counts as total error
    return jiwer.wer(ref, hyp,
                     reference_transform=NORMALIZE,
                     hypothesis_transform=NORMALIZE)


def main():
    df = pd.read_csv(PREDICTIONS).fillna({"hypothesis_text": ""})
    df["wer"] = [wer_of(r, h) for r, h in
                 zip(df["reference_text"], df["hypothesis_text"])]
    df["cer"] = [jiwer.cer(r.lower(), h.lower()) if h else 1.0 for r, h in
                 zip(df["reference_text"], df["hypothesis_text"])]

    # --- per-cell table (the core deliverable) --------------------------
    table = (df.groupby(["model", "accent_group", "noise_condition"])
               [["wer", "cer"]].mean().round(3).reset_index())
    table.to_csv(RESULTS_TABLE, index=False)
    print(table.to_string(index=False))

    # --- one heatmap per model ------------------------------------------
    for model_name, sub in table.groupby("model"):
        pivot = sub.pivot(index="accent_group", columns="noise_condition",
                          values="wer")
        pivot = pivot[["clean", "snr10", "snr0"]]        # logical order
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(pivot.values, cmap="Reds", vmin=0, vmax=1)
        ax.set_xticks(range(len(pivot.columns)), pivot.columns)
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{pivot.values[i, j]:.2f}",
                        ha="center", va="center")
        ax.set_title(f"WER by accent x noise — {model_name}")
        fig.colorbar(im, label="WER")
        fig.tight_layout()
        out = f"results/heatmap_{model_name}.png"
        fig.savefig(out, dpi=150)
        print(f"Saved {out}")

    # --- worst failures for qualitative analysis ------------------------
    worst = df.sort_values("wer", ascending=False).head(10)
    worst[["clip_id", "model", "accent_group", "noise_condition",
           "wer", "reference_text", "hypothesis_text"]].to_csv(
           FAILURES, index=False)
    print(f"Saved {FAILURES} — categorize these by hand for the report "
          f"(accent-driven? noise-driven? model-specific?)")


if __name__ == "__main__":
    main()
