""" Grade and slice.

What this script does:
  1. Loads the predictions from step 3.
  2. Computes WER and CER for every single prediction.
  3. Averages them per (model x accent_group x noise_condition) "cell"
     and saves that table — this is the core deliverable.
  4. Draws one heatmap image per model.
  5. Saves the 10 worst predictions.

About text normalization (IMPORTANT for the report):
  wav2vec2 outputs UPPERCASE TEXT WITHOUT PUNCTUATION, while Whisper
  outputs normal cased, punctuated text. If we compared them raw, WER
  would punish formatting differences, not actual recognition mistakes.
  So we lowercase everything, strip punctuation, etc. — and we apply the
  SAME normalization to both the reference and the model output.
"""

import jiwer
import pandas as pd

import matplotlib
matplotlib.use("Agg")   # "Agg" = draw to image files, no screen needed
import matplotlib.pyplot as plt

from config import PREDICTIONS, RESULTS_TABLE, FAILURES

# The normalization pipeline. jiwer.Compose chains steps together:
# the text goes through each step, top to bottom.
# This exact same pipeline is applied to BOTH references and hypotheses.
NORMALIZE = jiwer.Compose([
    jiwer.ToLowerCase(),              # "The Cat" -> "the cat"
    jiwer.RemovePunctuation(),        # "cat, dog." -> "cat dog"
    jiwer.RemoveMultipleSpaces(),     # "cat   dog" -> "cat dog"
    jiwer.Strip(),                    # remove spaces at start/end
    jiwer.ReduceToListOfListOfWords() # split into words for WER counting
])


def wer_of(ref, hyp):
    """Word Error Rate between the true text (ref) and model output (hyp).

    WER = (substitutions + deletions + insertions) / number of ref words.
    0.0 = perfect, 1.0 = everything wrong (it can even go above 1.0 if
    the model inserts lots of extra words).
    """
    # If the model produced nothing (e.g. it crashed in step 3),
    # count it as a total error rather than crashing here.
    if not hyp:
        return 1.0

    return jiwer.wer(ref, hyp,
                     reference_transform=NORMALIZE,
                     hypothesis_transform=NORMALIZE)


def cer_of(ref, hyp):
    """Character Error Rate — same idea as WER but counted per character.

    Useful alongside WER: a model that writes "recognise" instead of
    "recognize" gets a full word wrong (bad WER) but only one character
    wrong (good CER). Comparing the two tells you if errors are small
    spelling slips or completely wrong words.
    """
    if not hyp:
        return 1.0
    import re
    clean = lambda t: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t.lower())).strip()
    return jiwer.cer(clean(ref), clean(hyp))


def main():
    # ---- Load predictions, stop early if missing or empty ---------------
    try:
        df = pd.read_csv(PREDICTIONS)
    except FileNotFoundError:
        print(f"Predictions not found at {PREDICTIONS} — run step 3 first.")
        return
    except pd.errors.EmptyDataError:
        print(f"Predictions file at {PREDICTIONS} is empty — run step 3.")
        return

    if len(df) == 0:
        print("Predictions file has no rows — run step 3.")
        return

    # If a model produced empty output, the CSV stores it as NaN
    # (pandas' "missing value"). Replace those with empty strings so
    # our functions receive text, not NaN.
    df = df.fillna({"hypothesis_text": ""})

    # ---- Score every prediction -----------------------------------------
    # Build one score per row, then attach the scores as new columns.
    wer_scores = []
    cer_scores = []
    for i in range(len(df)):
        ref = df["reference_text"].iloc[i]   # .iloc[i] = value in row i
        hyp = df["hypothesis_text"].iloc[i]
        wer_scores.append(wer_of(ref, hyp))
        cer_scores.append(cer_of(ref, hyp))

    df["wer"] = wer_scores
    df["cer"] = cer_scores

    # ---- The per-cell results table (the core deliverable) --------------
    # groupby collects all rows that share the same (model, accent, noise)
    # combination into one group; .mean() then averages wer and cer
    # within each group. Result: one row per combination.
    grouped = df.groupby(["model", "accent_group", "noise_condition"])
    table = grouped[["wer", "cer"]].mean()

    # Round to 3 decimals and turn the group labels back into
    # normal columns (reset_index) so the CSV is easy to read.
    table = table.round(3)
    table = table.reset_index()

    table.to_csv(RESULTS_TABLE, index=False)
    print(table.to_string(index=False))

    # ---- One heatmap per model ------------------------------------------
    # groupby("model") here just splits the table into one sub-table
    # per model, so we can draw each model separately.
    for model_name, sub in table.groupby("model"):

        # pivot reshapes the sub-table into a grid:
        #   rows    = accent groups
        #   columns = noise conditions
        #   values  = the WER in each cell
        # That grid IS the heatmap data.
        pivot = sub.pivot(index="accent_group",
                          columns="noise_condition",
                          values="wer")

        # Put the columns in logical order (clean -> noisier),
        # instead of alphabetical.
        pivot = pivot[["clean", "snr10", "snr0"]]

        # Create an empty figure (6 x 4 inches).
        fig, ax = plt.subplots(figsize=(6, 4))

        # imshow paints the grid of numbers as colored squares.
        # cmap="Reds": higher value = darker red.
        # vmin/vmax pin the color scale to 0..1 so both models'
        # heatmaps use the SAME scale and can be compared fairly.
        im = ax.imshow(pivot.values, cmap="Reds", vmin=0, vmax=1)

        # Label the axes with the actual group / condition names.
        ax.set_xticks(range(len(pivot.columns)), pivot.columns)
        ax.set_yticks(range(len(pivot.index)), pivot.index)

        # Write the WER number inside each square, so readers don't
        # have to guess values from colors. (i = row, j = column.)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = pivot.values[i, j]
                ax.text(j, i, f"{value:.2f}",
                        ha="center", va="center")

        ax.set_title(f"WER by accent x noise — {model_name}")
        fig.colorbar(im, label="WER")   # the color legend on the side
        fig.tight_layout()              # avoid labels getting cut off

        out = f"results/heatmap_{model_name}.png"
        fig.savefig(out, dpi=150)
        print(f"Saved {out}")

    # ---- Worst failures, for qualitative analysis -----------------------
    # Sort all predictions from worst WER to best, keep the top 10.
    df_sorted = df.sort_values("wer", ascending=False)
    worst = df_sorted.head(10)

    # Keep only the columns useful for reading the failures by hand.
    columns_to_keep = ["clip_id", "model", "accent_group",
                       "noise_condition", "wer",
                       "reference_text", "hypothesis_text"]
    worst = worst[columns_to_keep]

    worst.to_csv(FAILURES, index=False)
    print(f"Saved {FAILURES} — categorize these by hand for the report "
          f"(accent-driven? noise-driven? model-specific?)")


if __name__ == "__main__":
    main()