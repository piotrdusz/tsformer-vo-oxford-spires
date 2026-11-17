#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path
from statistics import median


SEQUENCES = {
    "2024-03-14-blenheim-palace-01": "blenheim_palace_01",
    "2024-03-14-blenheim-palace-02": "blenheim_palace_02",
    "2024-03-14-blenheim-palace-05": "blenheim_palace_05",
    "2024-05-20-bodleian-library-02": "bodleian_library_02",
    "2024-03-12-keble-college-02": "keble_college_02",
    "2024-03-12-keble-college-03": "keble_college_03",
    "2024-03-12-keble-college-04": "keble_college_04",
    "2024-03-12-keble-college-05": "keble_college_05",
    "2024-03-13-observatory-quarter-01": "observatory_quarter_01",
    "2024-03-13-observatory-quarter-02": "observatory_quarter_02",
    "2024-03-18-christ-church-02": "christ_church_02",
    "2024-03-20-christ-church-05": "christ_church_05",
}
TARGET_RATE_HZ = 5.0


def timestamp_ns(image_path):
    seconds, fraction = image_path.stem.split(".", 1)
    nanoseconds = (fraction + "000000000")[:9]
    return int(seconds) * 1_000_000_000 + int(nanoseconds)


def inspect_sequence(source_dir, output_dir):
    image_dir = source_dir / "raw" / "cam0"
    gt_file = source_dir / "processed" / "trajectory" / "gt-tum.txt"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing front-camera image directory: {image_dir}")
    if not gt_file.is_file():
        raise FileNotFoundError(f"Missing ground-truth trajectory: {gt_file}")
    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists: {output_dir}. Move it or choose a fresh output root."
        )

    images = sorted(image_dir.glob("*.jpg"), key=timestamp_ns)
    if len(images) < 2:
        raise RuntimeError(f"Expected at least two timestamp-named JPG images in {image_dir}")

    timestamps = [timestamp_ns(image) for image in images]
    intervals = [later - earlier for earlier, later in zip(timestamps, timestamps[1:])]
    if any(interval <= 0 for interval in intervals):
        raise ValueError(f"Image timestamps must be unique and increasing in {image_dir}")

    source_rate_hz = 1_000_000_000 / median(intervals)
    stride = max(1, round(source_rate_hz / TARGET_RATE_HZ))
    output_rate_hz = source_rate_hz / stride
    if not 0.85 * TARGET_RATE_HZ <= output_rate_hz <= 1.15 * TARGET_RATE_HZ:
        raise ValueError(
            f"Cannot convert {source_rate_hz:.2f} Hz recording to approximately "
            f"{TARGET_RATE_HZ:g} Hz using an integer frame stride: {source_dir}"
        )

    return images, gt_file, stride, source_rate_hz


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare this experiment's Oxford Spires cam0 sequences at approximately "
            "5 Hz and copy their TUM ground-truth trajectories."
        )
    )
    parser.add_argument(
        "--sequences-root",
        type=Path,
        required=True,
        help="Oxford Spires sequences directory containing the dated sequence folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("oxford/5hz"),
        help="Output directory (default: oxford/5hz).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report detected rates and frame counts without writing files.",
    )
    args = parser.parse_args()

    prepared = []
    for source_name, output_name in SEQUENCES.items():
        source_dir = args.sequences_root / source_name
        output_dir = args.output_root / output_name
        images, gt_file, stride, source_rate_hz = inspect_sequence(source_dir, output_dir)
        selected_images = images[stride - 1 :: stride]
        prepared.append((output_dir, selected_images, gt_file))
        print(
            f"{source_name}: {source_rate_hz:.2f} Hz, stride={stride}, "
            f"{len(selected_images)}/{len(images)} frames -> {output_dir}"
        )

    if args.dry_run:
        return

    for output_dir, images, gt_file in prepared:
        image_output_dir = output_dir / "sequence"
        image_output_dir.mkdir(parents=True)
        shutil.copy2(gt_file, output_dir / "gt-tum.txt")
        for image in images:
            shutil.copy2(image, image_output_dir / image.name)

    print(f"\nPrepared {len(prepared)} sequences under {args.output_root}.")


if __name__ == "__main__":
    main()
