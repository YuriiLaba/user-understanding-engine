"""Merge per-modality observations on wall-clock windows, then propose.

What text drives the merge is chosen with --merge-on:
  summary-only   embed + de-duplicate on summaries; transcriptions are ignored entirely
                 and written out empty (original behaviour)   -> merged+<mods>_<slug>
  summary        embed + de-duplicate on summaries; the kept observation's original
                 transcription is carried through unchanged   -> merged-keep+<mods>_<slug>
  transcription  embed + de-duplicate on transcriptions; the kept observation's original
                 summary is carried through unchanged         -> merged-tr+<mods>_<slug>
  both           embed + de-duplicate on "summary\\n\\ntranscription"; both fields of the
                 kept observation are carried through         -> merged-both+<mods>_<slug>

In every mode the output folder holds summaries.jsonl + transcriptions.jsonl with fresh
pair_idx, so PropositionGenerator and the scoring sweep treat it like any other experiment
(the folder's modality token is everything before the first "_").

One process handles every (user, modality set) job: the embedder and the generation model
are loaded once and reused. Jobs whose folder already holds propositions.csv are skipped;
a folder without it is treated as an interrupted run and regenerated.
"""

import json
import os
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from scripts.merging import merge_moving_window
from src.cli_utils import model_slug, resolve_provider
from src.helpers.convert_propositions import convert_propositions_to_csv
from src.proposition_generator import PropositionGenerator

load_dotenv()

MERGE_ON = ("summary-only", "summary", "transcription", "both")
FOLDER_PREFIX = {
    "summary-only": "merged",
    "summary": "merged-keep",
    "transcription": "merged-tr",
    "both": "merged-both",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--user-name", "--user-names", dest="user_names", nargs="+", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--input-dir", default="new_data", help="Root that holds <user>/<modality>_<slug>/ folders.")
    parser.add_argument("--output-dir", default="new_data", help="Root to write the merged folders into.")
    parser.add_argument("--modalities", nargs="+", default=["ax", "metadata", "ocr"],
                        help="A single modality set. Ignored when --mod-sets is given.")
    parser.add_argument("--mod-sets", nargs="+", default=None,
                        help='Several modality sets, each quoted: "ax metadata" "ax ocr" (or ax+ocr).')
    parser.add_argument("--merge-on", choices=MERGE_ON, default="summary-only",
                        help="Which text is embedded and de-duplicated, and whether transcriptions are "
                             "carried into the output. Default: summary-only")
    parser.add_argument("--window-seconds", type=int, default=300, help="Wall-clock window width in seconds.")
    parser.add_argument("--sim-threshold", type=float, default=0.75, help="Cosine threshold for dedup.")
    parser.add_argument("--k-items-per-batch", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=3)
    parser.add_argument("--source-slug", default=None, help="Read from <modality>_<source-slug>/ instead of model slug.")
    parser.add_argument("--long-term", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable cross-window dedup pass (pass 2). --no-long-term skips it.")
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=False,
                        help="Also build user_profile.json (gpt-4o, needs OPENAI_API_KEY). Scoring doesn't use it.")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _iso_to_epoch(ts: str) -> float:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[WARN] not found: {path}")
        return []
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def _load_pairs(input_dir: str, user: str, modality: str, slug: str) -> list[dict]:
    """Summaries joined with their transcriptions on pair_idx (missing transcription -> '')."""
    folder = Path(input_dir) / user / f"{modality}_{slug}"
    summaries = _read_jsonl(folder / "summaries.jsonl")
    transcriptions = {
        rec["pair_idx"]: (rec.get("transcription") or "")
        for rec in _read_jsonl(folder / "transcriptions.jsonl")
        if "pair_idx" in rec
    }
    for s in summaries:
        s["transcription"] = transcriptions.get(s.get("pair_idx"), "")
    return summaries


def _merge_text(summary: str, transcription: str, merge_on: str) -> str:
    if merge_on in ("summary-only", "summary"):
        return summary.strip()
    if merge_on == "transcription":
        return transcription.strip()
    return "\n\n".join(t for t in (summary.strip(), transcription.strip()) if t)


def _load_model(model_name: str):
    provider = resolve_provider(model_name)
    if provider is None:
        raise ValueError(f"Model '{model_name}' is not listed in SUPPORTED_MODELS.")
    if provider == "openai":
        from src.models.openai_model import OpenAIVisionModel
        return OpenAIVisionModel(model_identifier=model_name, api_key=os.getenv("OPENAI_API_KEY"))
    if provider == "sglang":
        from src.models.sglang_model import SGLangVisionModel
        return SGLangVisionModel(model_identifier=model_name)
    if provider == "mlx":
        from src.models.mlx_model import MLXVisionModel
        return MLXVisionModel(model_identifier=model_name)
    if provider == "mlx_lm":
        from src.models.mlx_lm_model import MLXModel
        return MLXModel(model_identifier=model_name)
    raise ValueError(f"Unknown provider: {provider}")


def _plan_jobs(args, slug: str) -> list[tuple[str, list[str], str]]:
    """(user, modalities, experiment_id) for every job that still needs running."""
    raw_sets = args.mod_sets or [" ".join(args.modalities)]
    mod_sets = [s.replace("+", " ").split() for s in raw_sets]
    jobs = []
    for user in args.user_names:
        for mods in mod_sets:
            experiment_id = os.path.join(
                args.output_dir, user, f"{FOLDER_PREFIX[args.merge_on]}+{'+'.join(mods)}_{slug}"
            )
            missing = [m for m in mods if not (Path(args.input_dir) / user / f"{m}_{slug}").is_dir()]
            if missing:
                print(f"--- skip {user} [{' '.join(mods)}]: no {', '.join(f'{m}_{slug}' for m in missing)}")
            elif os.path.exists(os.path.join(experiment_id, "propositions.csv")):
                print(f"--- skip (done): {experiment_id}")
            else:
                jobs.append((user, mods, experiment_id))
    return jobs


def run_job(args, slug: str, user: str, modalities: list[str], experiment_id: str, embedder, model) -> None:
    params = {
        "user_name": user,
        "model_name": args.model_name,
        "experiment_id": experiment_id,
        "modalities": modalities,
        "merge_on": args.merge_on,
        "window_seconds": args.window_seconds,
        "sim_threshold": args.sim_threshold,
        "k_items_per_batch": args.k_items_per_batch,
        "source_slug": slug,
        "long_term": args.long_term,
    }

    all_items: list[dict] = []
    per_mod_counts: dict[str, int] = {}
    per_mod_with_tr: dict[str, int] = {}
    for modality in modalities:
        count = with_tr = 0
        for s in _load_pairs(args.input_dir, user, modality, slug):
            ts = s.get("timestamp_range") or {}
            start_str = ts.get("start")
            summary = s.get("summary") or ""
            transcription = s.get("transcription") or ""
            text = _merge_text(summary, transcription, args.merge_on)
            if not start_str or not text:
                continue
            all_items.append({
                "proposition": text,                 # merge_moving_window embeds this key
                "batch_id": _iso_to_epoch(start_str),
                "summary": summary,
                "transcription": transcription,
                "timestamp_range": ts,
                "modality": modality,
                "source_pair_idx": s.get("pair_idx"),
            })
            count += 1
            with_tr += bool(transcription.strip())
        per_mod_counts[modality] = count
        per_mod_with_tr[modality] = with_tr

    for mod, cnt in per_mod_counts.items():
        print(f"  {mod:<12}: {cnt} observations ({per_mod_with_tr[mod]} with transcription)")
    print(f"  total    : {len(all_items)} observations")
    if not all_items:
        print("  [ERROR] No observations with timestamps and non-empty merge text found.")
        return

    t_merge = time.time()
    merged_texts, stats, src_indices = merge_moving_window(
        items=all_items,
        embedder=embedder,
        window_size=args.window_seconds,
        sim_threshold=args.sim_threshold,
        long_term=args.long_term,
        return_indices=True,
        show_progress_bar=False,
    )
    print(f"  merged       : {len(all_items)} → {len(merged_texts)}  "
          f"(windows={stats.get('merge_windows', '?')}, pass1={stats.get('merge_pass1_count', '?')}, "
          f"{time.time() - t_merge:.1f}s)")

    os.makedirs(experiment_id, exist_ok=True)
    with open(os.path.join(experiment_id, "params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    per_mod_kept = {m: sum(1 for k in src_indices if all_items[k]["modality"] == m) for m in modalities}
    with open(os.path.join(experiment_id, "merge_stats.json"), "w", encoding="utf-8") as f:
        json.dump({**stats, "kept_per_source_modality": per_mod_kept}, f, indent=2)

    summaries_path = os.path.join(experiment_id, "summaries.jsonl")
    transcriptions_path = os.path.join(experiment_id, "transcriptions.jsonl")
    with open(summaries_path, "w", encoding="utf-8") as sf, \
         open(transcriptions_path, "w", encoding="utf-8") as tf:
        for i, (text, src_idx) in enumerate(zip(merged_texts, src_indices)):
            src = all_items[src_idx]
            # The merged text replaces only the field it was built from; the other field is
            # the kept observation's original. In "both" mode the merged text is a concat of
            # the two, so both originals are written as-is.
            summary = text if args.merge_on in ("summary-only", "summary") else src["summary"]
            if args.merge_on == "summary-only":
                transcription = ""
            elif args.merge_on == "transcription":
                transcription = text
            else:
                transcription = src["transcription"]
            meta = {
                "pair_idx": i,
                "modality": src["modality"],
                "source_pair_idx": src["source_pair_idx"],
            }
            sf.write(json.dumps({
                **meta, "summary": summary, "timestamp_range": src["timestamp_range"],
            }, ensure_ascii=False) + "\n")
            tf.write(json.dumps({**meta, "transcription": transcription}, ensure_ascii=False) + "\n")

    # The generator appends, so drop any output left by an interrupted run of this job.
    propositions_path = os.path.join(experiment_id, "propositions.jsonl")
    for stale in (propositions_path, os.path.join(experiment_id, "logs.log")):
        if os.path.exists(stale):
            os.remove(stale)

    generator = PropositionGenerator(
        model=model,
        user_name=user,
        k_items_per_batch=args.k_items_per_batch,
        chunk_size=args.chunk_size,
        log_filename=os.path.join(experiment_id, "logs.log"),
    )
    generator.process(
        summaries_path=summaries_path,
        transcriptions_path=transcriptions_path,
        output_path=propositions_path,
    )
    for handler in generator.logger.handlers:
        handler.close()

    csv_path = os.path.join(experiment_id, "propositions.csv")
    if args.profile:
        # Profile first, CSV marker last: propositions.csv is what marks a job as done.
        from src.user_profile_generator import generate_user_profile
        tmp_csv = csv_path + ".tmp"
        convert_propositions_to_csv(propositions_path, tmp_csv)
        generate_user_profile(tmp_csv, os.path.join(experiment_id, "user_profile.json"))
        os.replace(tmp_csv, csv_path)
    else:
        convert_propositions_to_csv(propositions_path, csv_path)


def main() -> None:
    args = parse_args()
    slug = args.source_slug or model_slug(args.model_name)

    print("=== Run parameters ===")
    for k in ("user_names", "model_name", "merge_on", "window_seconds", "sim_threshold",
              "k_items_per_batch", "chunk_size", "long_term", "profile"):
        print(f"  {k:<18}: {getattr(args, k)}")
    print(f"  {'mod_sets':<18}: {args.mod_sets or [' '.join(args.modalities)]}")
    print("======================")

    jobs = _plan_jobs(args, slug)
    print(f"\n{len(jobs)} job(s) to run")
    if not jobs:
        return

    t_total = time.time()
    from sentence_transformers import SentenceTransformer
    import torch
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(args.embedding_model, device=device)
    print(f"Embedder [{args.embedding_model}] on {device}")

    model = _load_model(args.model_name)
    print(f"Loaded in {time.time() - t_total:.1f}s")

    failed = []
    for n, (user, mods, experiment_id) in enumerate(jobs, 1):
        print(f"\n>>> job [{n}/{len(jobs)}] {user} [{' '.join(mods)}]  -> {experiment_id}")
        t_job = time.time()
        try:
            run_job(args, slug, user, mods, experiment_id, embedder, model)
        except Exception as e:  # noqa: BLE001 - one bad job shouldn't stop the rest
            print(f"  [FAILED] {type(e).__name__}: {e}")
            failed.append(experiment_id)
            continue
        print(f"  done in {time.time() - t_job:.1f}s")

    if hasattr(model, "close"):
        model.close()

    print(f"\n{'='*50}")
    print(f"  {len(jobs) - len(failed)}/{len(jobs)} job(s) done  —  total elapsed: {time.time() - t_total:.1f}s")
    for f in failed:
        print(f"  FAILED: {f}")
    print(f"{'='*50}\n")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
