from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


DATASETS = ("amazon_game", "yelp", "citeulike_t")


def metric_at(result, metric_index: int, topn: list[int], cutoff: int = 20) -> float:
    if result is None:
        raise ValueError("missing best result")
    return float(result[metric_index][topn.index(cutoff)])


def schedule_tag(args: dict) -> str:
    scheduler = args.get("scheduler", "mas_snr")
    if scheduler == "fcps":
        tag = f"fcps_a{float(args.get('fcps_alpha', 3.0)):g}"
        if args.get("fcps_calibration", "snr_matched") != "snr_matched":
            tag += "_manual"
        return tag
    return scheduler


def read_result(path: Path) -> dict:
    payload = json.loads(path.read_text())
    args = payload.get("args", {})
    dataset = args.get("dataset")
    seed = args.get("random_seed")
    if dataset not in DATASETS or seed is None:
        raise ValueError(f"{path}: missing supported dataset or random_seed")
    topn_raw = args.get("topN", "[10, 20, 50, 100]")
    topn = ast.literal_eval(topn_raw) if isinstance(topn_raw, str) else list(topn_raw)
    return {
        "tag": schedule_tag(args),
        "dataset": dataset,
        "seed": int(seed),
        "best_epoch": int(payload.get("best_epoch", 0)),
        "val_recall20": metric_at(payload.get("best_valid"), 1, topn),
        "recall20": metric_at(payload.get("best_test"), 1, topn),
        "ndcg20": metric_at(payload.get("best_test"), 2, topn),
        "path": str(path),
    }


def compare(summary: list[dict]) -> None:
    by_key = {(row["tag"], row["dataset"]): row for row in summary}
    fcps_tags = sorted({row["tag"] for row in summary if row["tag"].startswith("fcps_a")})
    if not fcps_tags or not any(tag == "mas_snr" for tag, _ in by_key):
        return
    print("\nMASS vs FCPS (FCPS alpha selected per dataset by mean validation Recall@20; test metrics shown)")
    for dataset in DATASETS:
        mas = by_key.get(("mas_snr", dataset))
        candidates = [by_key[(tag, dataset)] for tag in fcps_tags if (tag, dataset) in by_key]
        if mas is None or not candidates:
            continue
        best = max(candidates, key=lambda row: row["val_recall20_mean"])
        gain_r = 100.0 * (mas["recall20_mean"] / best["recall20_mean"] - 1.0)
        gain_n = 100.0 * (mas["ndcg20_mean"] / best["ndcg20_mean"] - 1.0)
        swept = ", ".join(f"{row['tag']}:{row['val_recall20_mean']:.4f}" for row in candidates)
        print(
            f"  {dataset}: MASS R@20={mas['recall20_mean']:.4f} N@20={mas['ndcg20_mean']:.4f} | "
            f"{best['tag']} R@20={best['recall20_mean']:.4f} N@20={best['ndcg20_mean']:.4f} | "
            f"MASS vs FCPS: R@20 {gain_r:+.1f}%, N@20 {gain_n:+.1f}%"
        )
        print(f"    validation R@20 over swept alphas: {swept}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--tag", action="append", default=None,
                        help="Keep only these schedule tags (repeatable), e.g. --tag mas_snr --tag fcps_a3")
    parser.add_argument("--strict", action="store_true",
                        help="Require exactly seeds 1-5 on every dataset for every schedule tag kept")
    args = parser.parse_args()

    rows = []
    for path in sorted(args.root.rglob("*.json")):
        try:
            rows.append(read_result(path))
        except (ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            print(f"Skipping {path}: {exc}")
    if args.tag:
        rows = [row for row in rows if row["tag"] in set(args.tag)]

    grouped: dict[tuple[str, str], dict[int, dict]] = {}
    for row in rows:
        seed_rows = grouped.setdefault((row["tag"], row["dataset"]), {})
        if row["seed"] in seed_rows:
            raise SystemExit(f"Duplicate result for {row['tag']} {row['dataset']} seed {row['seed']}: "
                             f"{seed_rows[row['seed']]['path']} and {row['path']}")
        seed_rows[row["seed"]] = row

    expected = {1, 2, 3, 4, 5}
    tags = sorted({tag for tag, _ in grouped}, key=lambda t: (t != "mas_snr", t))
    if args.strict and args.tag:
        missing = sorted(set(args.tag) - set(tags))
        if missing:
            raise SystemExit(f"No results found for tag(s): {missing}")
    if args.strict and not tags:
        raise SystemExit("No results found")
    summary = []
    for tag in tags:
        for dataset in DATASETS:
            seed_rows = grouped.get((tag, dataset), {})
            if args.strict and set(seed_rows) != expected:
                raise SystemExit(f"{tag} {dataset}: expected seeds 1-5, found {sorted(seed_rows)}")
            if not seed_rows:
                print(f"{tag} {dataset}: no results found")
                continue
            ordered = [seed_rows[s] for s in sorted(seed_rows)]
            val = [row["val_recall20"] for row in ordered]
            recalls = [row["recall20"] for row in ordered]
            ndcgs = [row["ndcg20"] for row in ordered]
            item = {
                "tag": tag,
                "dataset": dataset,
                "n": len(ordered),
                "seeds": ",".join(str(row["seed"]) for row in ordered),
                "val_recall20_mean": mean(val),
                "recall20_mean": mean(recalls),
                "recall20_pstdev": pstdev(recalls),
                "ndcg20_mean": mean(ndcgs),
                "ndcg20_pstdev": pstdev(ndcgs),
            }
            summary.append(item)
            print(
                f"{tag} {dataset}: n={item['n']} "
                f"Recall@20={item['recall20_mean']:.4f}+/-{item['recall20_pstdev']:.4f} "
                f"NDCG@20={item['ndcg20_mean']:.4f}+/-{item['ndcg20_pstdev']:.4f} "
                f"(val Recall@20={item['val_recall20_mean']:.4f})"
            )

    compare(summary)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fields = ["tag", "dataset", "n", "seeds", "val_recall20_mean", "recall20_mean", "recall20_pstdev",
                  "ndcg20_mean", "ndcg20_pstdev"]
        with args.out.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(summary)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
