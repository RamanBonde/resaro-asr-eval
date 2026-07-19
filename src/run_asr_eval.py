"""
run_asr_eval.py — Evaluate & compare two ASR models via the AIP v2 platform.

ASR evaluation is, underneath, a text comparison (predicted vs reference
transcript), so clips are mapped onto the platform's text schema
(gdi_text_v1) and scored in external mode:

    clip_id         -> input_id
    path            -> prompt          (placeholder; no audio field exists)
    reference_text  -> expected_output
    model transcript-> sut_response    (produced locally, attached per run)

WER/CER are computed locally with jiwer — the platform has no ASR scorer,
and requesting text metrics at run creation fails validation in external
mode (see report, Gap 3). Runs are therefore created without explicit
metrics; they carry the data, dimensions, and audit trail.

Run order:
    python run_asr_eval.py prep      # upload -> quality checks -> golden
    python run_asr_eval.py score     # two external runs + local WER/CER
    python run_asr_eval.py compare   # cross-model stratified summary
"""

import os
import sys
import json
import time
import pathlib

import pandas as pd

# ---------------------------------------------------------------------------
# 0. Config can be used.
# ---------------------------------------------------------------------------
MANIFEST_PATH = "data/manifest.csv"

# model_name -> predictions CSV (columns: clip_id, hypothesis)
MODELS = {
    "whisper-base": "results/predictions/whisper-base.csv",
    "wav2vec2-960h": "results/predictions/wav2vec2-base-960h.csv",
}

PROJECT_NAME = "asr-comparison"
OUTPUT_DIR = "outputs"
PLATFORM_METRICS = ["llm.bleu", "llm.rouge", "llm.exact_match"]

# We stash run IDs here between `score` and `compare` steps.
STATE_PATH = os.path.join(OUTPUT_DIR, "_run_ids.json")


# ---------------------------------------------------------------------------
# 1. Imports that may not be installed — fail with a helpful message.
# ---------------------------------------------------------------------------
def _require(module, hint):
    try:
        return __import__(module)
    except ImportError:
        sys.exit(f"Missing dependency '{module}'. Install it with: {hint}")


def connect():
    """Init the SDK from .env and return (aip, workspace)."""
    from dotenv import load_dotenv
    load_dotenv()
    import aip_sdk as aip

    base_url = os.environ.get("AIP_BASE_URL")
    if not base_url:
        sys.exit("Set AIP_BASE_URL in your .env (see onboarding.md §3).")

    if "AIP_API_KEY" in os.environ:
        aip.init(base_url, api_key=os.environ["AIP_API_KEY"])
        ws = aip.Workspace.get_by_name(os.environ["AIP_WORKSPACE_NAME"])
    elif "AIP_USERNAME" in os.environ and "AIP_PASSWORD" in os.environ:
        aip.init(base_url,
                 username=os.environ["AIP_USERNAME"],
                 password=os.environ["AIP_PASSWORD"])
        ws = aip.Workspace.get_by_name(os.environ.get("AIP_WORKSPACE_NAME", "Default"))
    else:
        sys.exit("Provide AIP_API_KEY, or AIP_USERNAME + AIP_PASSWORD, in .env.")

    print(f"Connected — workspace {ws.name} (id {ws.id})")
    return aip, ws


# ---------------------------------------------------------------------------
# 2. Text normalisation (standard ASR practice before scoring).
# ---------------------------------------------------------------------------
import re
_PUNCT = re.compile(r"[^\w\s]")

def normalise(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = _PUNCT.sub(" ", text)          # drop punctuation
    text = re.sub(r"\s+", " ", text)      # collapse whitespace
    return text.strip()


# ---------------------------------------------------------------------------
# 3. PREP — build the golden dataset on the platform.
# ---------------------------------------------------------------------------
def prep():
    aip, ws = connect()
    from aip_sdk import Dimension

    man = pd.read_csv(MANIFEST_PATH)
    required = {"clip_id", "path", "reference_text", "accent_group", "noise_condition"}
    missing = required - set(man.columns)
    if missing:
        sys.exit(f"manifest.csv missing columns: {missing}")

    # One row per clip is what we score.
    # (clip, noise_condition) that's fine — each becomes its own input_id.
    print(f"Manifest: {len(man)} rows")

    # Dimensions drive coverage + group_by analysis. Values come from the data.
    DIMENSIONS = [
        Dimension(name="accent_group", column="accent_group",
                  values=sorted(man["accent_group"].dropna().unique().tolist())),
        Dimension(name="noise_condition", column="noise_condition",
                  values=sorted(man["noise_condition"].dropna().unique().tolist())),
    ]

    project, _ = aip.Project.get_or_create(
        name=PROJECT_NAME,
        schema="gdi_text_v1",
        task_type="single_turn_llm",
        dimensions=DIMENSIONS,
        workspace_id=ws.id,
    )
    print(f"Project: {project.id} ({PROJECT_NAME})")

    # Raw upload — keep original column names; map to GDI in the next step.
    raw = pd.DataFrame({
        "clip_id": man["clip_id"],
        "path": man["path"],
        "reference_text": man["reference_text"],
        "accent_group": man["accent_group"],
        "noise_condition": man["noise_condition"],
    })
    dataset = project.upload_dataset(raw, name=f"asr-{int(time.time())}")
    version = dataset.latest_version()

    # Map raw columns -> GDI contract.
    version = dataset.map_version(
        version.id,
        column_mapping={
            "input_id": "clip_id",
            "prompt": "path",                 # placeholder: audio path, not text
            "expected_output": "reference_text",
        },
    )

    # Quality checks (server verdict gates promotion). We also compute them
    # locally so the report can show the PASS/WARN/FAIL evidence.
    df_mapped = raw.rename(columns={
        "clip_id": "input_id", "path": "prompt", "reference_text": "expected_output",
    })
    report = aip.run_quality_checks(
        df_mapped,
        checks=[
            aip.RowCountQualityCheck(),
            aip.DuplicateIdQualityCheck(),
            aip.SchemaCompletenessQualityCheck(project.schema_name),
            aip.VocabularyCoverageQualityCheck(DIMENSIONS),
        ],
    )
    dataset.upload_check_results(version.id, report)

    # Also fire the authoritative server-side checks, then promote.
    try:
        dataset.run_checks(version.id, label="pre-eval")
    except Exception as e:
        print(f"(server-side run_checks note: {e})")

    verdict = None
    try:
        verdict = dataset.quality_verdict(version.id)
        print(f"Quality verdict: {verdict}")
    except Exception:
        pass

    # Promote to golden. If verdict is WARN (e.g. small dev sample), force with a reason.
    try:
        version = dataset.promote(version.id)
    except Exception:
        version = dataset.promote(
            version.id, force=True,
            reason="Small dev sample; ASR text-proxy eval — audited WARN override.",
        )
    print(f"Golden: {dataset.id}@v{version.version}")

    _save_state({"project_id": project.id,
                 "dataset_ref": f"{dataset.id}@v{version.version}"})
    print("PREP done.")


# ---------------------------------------------------------------------------
# 4. SCORE — one external run per model.
# ---------------------------------------------------------------------------
def load_predictions(csv_path):
    """Return {clip_id: hypothesis}."""
    if not os.path.exists(csv_path):
        sys.exit(f"Predictions file not found: {csv_path}\n"
                 f"Run your ASR model over the clips first and write "
                 f"columns clip_id,hypothesis to this path.")
    p = pd.read_csv(csv_path)
    if not {"clip_id", "hypothesis"} <= set(p.columns):
        sys.exit(f"{csv_path} must have columns: clip_id, hypothesis")
    return dict(zip(p["clip_id"], p["hypothesis"]))


def score():
    aip, _ = connect()
    _require("aip_metrics", "uv pip install <the aip_metrics wheel>")
    import aip_metrics

    state = _load_state()
    run_ids = {}

    for model_name, pred_csv in MODELS.items():
        print(f"\n=== Scoring {model_name} ===")
        preds = load_predictions(pred_csv)

        with aip.run(
            project=state["project_id"],
            dataset=state["dataset_ref"],
            tags={"model": model_name, "modality": "asr", "mode": "external"},
        ) as run:
            df = run.dataset.pull()          # golden rows: input_id, prompt, expected_output, dims

            # Attach this model's transcript as the SUT response, by input_id.
            df["sut_response"] = df["input_id"].map(preds).fillna("")
            unmapped = (df["sut_response"] == "").sum()
            if unmapped:
                print(f"  WARNING: {unmapped} clips had no prediction (scored as empty).")

            # Platform scoring (built-in text metrics).
            scored = aip_metrics.score(df, scorers=run.required_scorers)
            run.upload(scored)
            run.poll_status(timeout=1200)
            print(f"  Run {run.id} — {run.url}")

            # Pull per-row platform scores for the report.
            page = run.results(page=1, page_size=1000)
            plat = pd.DataFrame([
                {"input_id": r.input_id, "scorer": r.scorer, "score": r.score}
                for r in page.results
            ])

            run_ids[model_name] = run.id

        # ---- LOCAL WER / CER (the platform has no native scorer for these) ----
        jiwer = _require("jiwer", "uv pip install jiwer")
        local_rows = []
        for _, row in df.iterrows():
            ref = normalise(row["expected_output"])
            hyp = normalise(row["sut_response"])
            if not ref:
                continue
            local_rows.append({
                "input_id": row["input_id"],
                "accent_group": row.get("accent_group"),
                "noise_condition": row.get("noise_condition"),
                "wer": jiwer.wer(ref, hyp),
                "cer": jiwer.cer(ref, hyp),
            })
        local = pd.DataFrame(local_rows)

        # Merge platform proxy scores with local WER/CER; one row per clip.
        if not plat.empty:
            wide = plat.pivot_table(index="input_id", columns="scorer",
                                    values="score", aggfunc="first").reset_index()
            merged = local.merge(wide, on="input_id", how="left")
        else:
            merged = local

        out_csv = os.path.join(OUTPUT_DIR, f"scores_{model_name}.csv")
        _ensure_dir()
        merged.to_csv(out_csv, index=False)
        print(f"  Wrote {out_csv}")

        # Quick stratified WER (this is the headline table for your report).
        if not local.empty:
            print("  WER by accent_group:")
            print(local.groupby("accent_group")["wer"].mean().round(3).to_string())
            print("  WER by noise_condition:")
            print(local.groupby("noise_condition")["wer"].mean().round(3).to_string())

    state["run_ids"] = run_ids
    _save_state(state)
    print("\nSCORE done. Run IDs:", run_ids)


# ---------------------------------------------------------------------------
# 5. COMPARE — diff the two runs + a combined stratified view.
# ---------------------------------------------------------------------------
def compare():
    aip, _ = connect()
    state = _load_state()
    run_ids = state.get("run_ids", {})
    if len(run_ids) < 2:
        sys.exit("Need two scored runs. Run `score` first.")

    names = list(run_ids.keys())
    a, b = run_ids[names[0]], run_ids[names[1]]
    print(f"Comparing {names[0]} ({a})  vs  {names[1]} ({b})")

    # Platform diff (per-input, per-scorer deltas) -> CSV.
    _ensure_dir()
    diff_csv = os.path.join(OUTPUT_DIR, "platform_diff.csv")
    try:
        aip.diff(a, b).export(path=diff_csv)
        print(f"Wrote {diff_csv}")
    except Exception as e:
        print(f"(platform diff note: {e})")

    # Combined local WER/CER comparison across models, stratified.
    frames = []
    for name in names:
        f = os.path.join(OUTPUT_DIR, f"scores_{name}.csv")
        if os.path.exists(f):
            d = pd.read_csv(f)
            d["model"] = name
            frames.append(d)
    if frames:
        allm = pd.concat(frames, ignore_index=True)
        print("\n=== WER by model x accent_group ===")
        print(allm.pivot_table(index="accent_group", columns="model",
                               values="wer", aggfunc="mean").round(3).to_string())
        print("\n=== WER by model x noise_condition ===")
        print(allm.pivot_table(index="noise_condition", columns="model",
                               values="wer", aggfunc="mean").round(3).to_string())
        summary = os.path.join(OUTPUT_DIR, "comparison_summary.csv")
        allm.to_csv(summary, index=False)
        print(f"\nWrote {summary}")

    # Platform-side stratified analysis (per run), for the dashboard/report.
    for name, rid in run_ids.items():
        try:
            run = aip.get_run(rid)
            an = run.analysis(group_by="accent_group")
            print(f"\n[{name}] platform scorer_stats: "
                  f"{json.dumps(an.get('scorer_stats', {}), default=str)[:400]}")
        except Exception as e:
            print(f"(analysis note for {name}: {e})")

    print("\nCOMPARE done.")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _ensure_dir():
    pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def _save_state(state):
    _ensure_dir()
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)

def _load_state():
    if not os.path.exists(STATE_PATH):
        sys.exit("No saved state — run `prep` first.")
    with open(STATE_PATH) as f:
        return json.load(f)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "prep":
        prep()
    elif cmd == "score":
        score()
    elif cmd == "compare":
        compare()
    elif cmd == "all":
        prep(); score(); compare()
    else:
        sys.exit("Usage: python run_asr_eval.py [prep|score|compare|all]")