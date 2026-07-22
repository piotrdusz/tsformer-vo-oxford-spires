#!/usr/bin/env python3

import argparse
import csv
import os
import shutil
from bisect import bisect_left, bisect_right
from pathlib import Path


def find_gt_file(sequence_root):
    for name in ("gt-tum.txt", "ground_truth.txt", "groundtruth.txt"):
        candidate = sequence_root / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No GT file found in {sequence_root}")


def parse_image_timestamp_ns(image_name):
    stem = image_name.rsplit(".", 1)[0]
    sec_str, nsec_str = stem.split(".", 1)
    nsec_str = (nsec_str + "0" * 9)[:9]
    return int(sec_str) * 1_000_000_000 + int(nsec_str)


def load_sequence_index(sequence_dir):
    images = [p.name for p in sequence_dir.glob("*.jpg")]
    index = []
    for name in images:
        try:
            ts_ns = parse_image_timestamp_ns(name)
        except (ValueError, IndexError):
            continue
        index.append((ts_ns, name))

    index.sort(key=lambda x: x[0])
    timestamps = [x[0] for x in index]
    names = [x[1] for x in index]
    return timestamps, names


def ensure_unique_dir(parent, desired_name):
    out_dir = parent / desired_name
    if not out_dir.exists():
        return out_dir

    suffix = 2
    while True:
        candidate = parent / f"{desired_name}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def copy_or_link(src, dst, mode):
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        os.symlink(src.resolve(), dst)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def process_spec(csv_path, source_root, output_parent, mode="copy", dry_run=False):
    sequence_dir = source_root / "sequence"
    if not sequence_dir.is_dir():
        raise FileNotFoundError(f"Missing sequence dir: {sequence_dir}")

    gt_file = find_gt_file(source_root)
    timestamps, names = load_sequence_index(sequence_dir)
    if not timestamps:
        raise RuntimeError(f"No valid JPG timestamps found in {sequence_dir}")

    created = 0
    skipped = 0

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"bag_name", "start_time_ns", "end_time_ns"}
        if not required_cols.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"CSV {csv_path} must contain columns: {sorted(required_cols)}"
            )

        for row in reader:
            bag_name = row["bag_name"].strip()
            start_ns = int(row["start_time_ns"])
            end_ns = int(row["end_time_ns"])

            left = bisect_left(timestamps, start_ns)
            right = bisect_right(timestamps, end_ns)
            subset_names = names[left:right]

            if not subset_names:
                skipped += 1
                print(f"[skip] {bag_name}: no frames in [{start_ns}, {end_ns}]")
                continue

            subset_dir = ensure_unique_dir(output_parent, bag_name)
            subset_seq_dir = subset_dir / "sequence"

            if dry_run:
                print(
                    f"[dry-run] {subset_dir} <- {len(subset_names)} frames, gt={gt_file.name}"
                )
                created += 1
                continue

            subset_seq_dir.mkdir(parents=True, exist_ok=False)
            shutil.copy2(gt_file, subset_dir / gt_file.name)

            for image_name in subset_names:
                src = sequence_dir / image_name
                dst = subset_seq_dir / image_name
                copy_or_link(src, dst, mode)

            created += 1
            print(f"[ok] {subset_dir}: {len(subset_names)} frames")

    return created, skipped


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create Oxford subsequences from timestamp CSV files. "
            "Each CSV row becomes one subsequence directory compatible with evaluate_tsformer_oxford.py."
        )
    )
    parser.add_argument(
        "--spec",
        action="append",
        nargs=3,
        metavar=("CSV", "SOURCE_ROOT", "OUTPUT_GROUP"),
        required=True,
        help=(
            "One generation spec: CSV path, source sequence root (contains sequence/ and GT), "
            "and output group name. Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="oxford/subsequences",
        help="Root output directory. Subfolders are created using OUTPUT_GROUP from each --spec.",
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "hardlink", "symlink"],
        default="copy",
        help="How to place selected images into subsequences.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be created.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    total_created = 0
    total_skipped = 0

    for csv_arg, source_arg, group_name in args.spec:
        csv_path = Path(csv_arg)
        source_root = Path(source_arg)
        group_root = output_root / group_name
        group_root.mkdir(parents=True, exist_ok=True)

        print(f"\n=== spec: csv={csv_path} source={source_root} group={group_name} ===")
        created, skipped = process_spec(
            csv_path=csv_path,
            source_root=source_root,
            output_parent=group_root,
            mode=args.mode,
            dry_run=args.dry_run,
        )
        total_created += created
        total_skipped += skipped

    print(
        f"\nDone. created={total_created}, skipped={total_skipped}, output_root={output_root}"
    )


if __name__ == "__main__":
    main()