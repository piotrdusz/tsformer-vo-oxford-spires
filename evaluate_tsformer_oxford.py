#!/usr/bin/env python3

import argparse
import csv
import glob
import json
import os
import pickle
import queue
import subprocess
import zipfile

import numpy as np
import torch
import torch.nn as nn
from functools import partial
from scipy.spatial.transform import Rotation
from torchvision import transforms
from tqdm import tqdm

from datasets.oxford import Oxford
from datasets.utils import euler_to_rotation
from timesformer.models.vit import VisionTransformer


def post_processing(pred_poses, args):
    if args["window_size"] == 2:
        pred_poses = pred_poses.squeeze(1)
        return np.asarray(pred_poses)

    num_batchs = pred_poses.shape[0]
    q = queue.Queue(args["window_size"] - 1)
    idx = 0
    poses = []

    while not q.full() and idx < num_batchs:
        q.put(pred_poses[idx, :, :])
        idx += 1

    while idx < num_batchs:
        if idx == (args["window_size"] - 1):
            poses.append(q.queue[0][0, :])
            avg_pose = (q.queue[0][1, :] + q.queue[1][0, :]) / 2
            poses.append(avg_pose)

            if args["window_size"] == 4:
                avg_pose = (q.queue[0][2, :] + q.queue[1][1, :] + q.queue[2][0, :]) / 3
                poses.append(avg_pose)

        elif idx < (num_batchs - 1):
            if args["window_size"] == 3:
                avg_pose = (q.queue[0][1, :] + q.queue[1][0, :]) / 2
                poses.append(avg_pose)
            elif args["window_size"] == 4:
                avg_pose = (q.queue[0][2, :] + q.queue[1][1, :] + q.queue[2][0, :]) / 3
                poses.append(avg_pose)

        else:
            if args["window_size"] == 3:
                poses.append(q.queue[1][1, :])
            elif args["window_size"] == 4:
                avg_pose = (q.queue[1][2, :] + q.queue[2][1, :]) / 2
                poses.append(avg_pose)
                poses.append(q.queue[2][2, :])
            idx += 1

        if idx < (num_batchs - 1):
            idx += 1
            _ = q.get()
            q.put(pred_poses[idx, :, :])

    return np.asarray(poses)


def recover_absolute_poses(poses):
    predicted_poses = []

    mean_angles = np.array([1.7061e-5, 9.5582e-4, -5.5258e-5])
    std_angles = np.array([2.8256e-3, 1.7771e-2, 3.2326e-3])
    mean_t = np.array([-8.6736e-5, -1.6038e-2, 9.0033e-1])
    std_t = np.array([2.5584e-2, 1.8545e-2, 3.0352e-1])

    T = np.eye(4)
    for i in range(len(poses) - 1):
        angles = poses[i, :3]
        t = poses[i, 3:]

        [x, y, z] = np.multiply(angles, std_angles) + mean_angles
        t = np.multiply(t, std_t) + mean_t
        R = np.asarray(euler_to_rotation(x, y, z, seq="zyx"))

        T_r = np.concatenate(
            (
                np.concatenate([R, np.reshape(t, (3, 1))], axis=1),
                [[0.0, 0.0, 0.0, 1.0]],
            ),
            axis=0,
        )
        T = np.dot(T, T_r)
        predicted_poses.append(T.copy())

    return predicted_poses


def save_poses_12col(poses_4x4, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for pose in poses_4x4:
            vals = pose.flatten()[:12]
            f.write(" ".join(str(x) for x in vals) + "\n")


def read_tum_file(gt_path):
    rows = []
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 8:
                ts = float(parts[0])
                vals = [float(x) for x in parts[1:]]
                rows.append((ts, vals))
    return rows


def extract_image_timestamps(sequence_dir):
    image_files = sorted(glob.glob(os.path.join(sequence_dir, "*.jpg")))
    timestamps = []
    for p in image_files:
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            timestamps.append(float(stem))
        except ValueError:
            continue
    return image_files, timestamps


def nearest_gt_timestamps_for_images(image_ts, gt_rows):
    if not image_ts or not gt_rows:
        return []
    gt_ts = np.array([r[0] for r in gt_rows], dtype=np.float64)
    matched = []
    for ts in image_ts:
        idx = int(np.argmin(np.abs(gt_ts - ts)))
        matched.append(float(gt_ts[idx]))
    return matched


def poses12_to_tum_with_timestamps(poses12_file, timestamps, output_tum_file):
    poses12 = []
    with open(poses12_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            poses12.append([float(x) for x in parts])

    n = min(len(poses12), len(timestamps))
    os.makedirs(os.path.dirname(output_tum_file), exist_ok=True)
    with open(output_tum_file, "w") as f:
        for i in range(n):
            p = poses12[i]
            if len(p) != 12:
                continue
            R = np.array(
                [
                    [p[0], p[1], p[2]],
                    [p[4], p[5], p[6]],
                    [p[8], p[9], p[10]],
                ]
            )
            t = np.array([p[3], p[7], p[11]])
            q = Rotation.from_matrix(R).as_quat()
            f.write(
                f"{timestamps[i]:.9f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n"
            )
    return n


def write_gt_overlap(gt_rows, image_ts, output_gt_file):
    if not gt_rows or not image_ts:
        return 0
    t_min = min(image_ts)
    t_max = max(image_ts)
    filtered = [(ts, vals) for ts, vals in gt_rows if (t_min <= ts <= t_max)]

    os.makedirs(os.path.dirname(output_gt_file), exist_ok=True)
    with open(output_gt_file, "w") as f:
        for ts, vals in filtered:
            f.write(
                f"{ts:.9f} {vals[0]:.9f} {vals[1]:.9f} {vals[2]:.9f} "
                f"{vals[3]:.9f} {vals[4]:.9f} {vals[5]:.9f} {vals[6]:.9f}\n"
            )
    return len(filtered)


def run_evo_metric(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    ok = proc.returncode == 0
    stderr_lines = [ln for ln in proc.stderr.split("\n") if ln.strip()]
    stdout_lines = [ln for ln in proc.stdout.split("\n") if ln.strip()]
    msg = ""
    if not ok:
        msg = stderr_lines[-1] if stderr_lines else (stdout_lines[-1] if stdout_lines else "unknown evo error")
    return ok, msg


def run_evo_traj_plot(gt_file, traj_file, plot_file, t_max_diff):
    cmd = [
        "evo_traj",
        "tum",
        traj_file,
        "--ref",
        gt_file,
        "-a",
        "--save_plot",
        plot_file,
        "--plot_mode=xy",
        "--t_max_diff",
        str(t_max_diff),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    ok = proc.returncode == 0
    stderr_lines = [ln for ln in proc.stderr.split("\n") if ln.strip()]
    stdout_lines = [ln for ln in proc.stdout.split("\n") if ln.strip()]
    msg = ""
    if not ok:
        msg = stderr_lines[-1] if stderr_lines else (stdout_lines[-1] if stdout_lines else "unknown evo_traj error")
    return ok, msg


def parse_rmse_from_zip(zip_path):
    if not os.path.exists(zip_path):
        return None
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open("stats.json") as fh:
                stats = json.load(fh)
                return stats.get("rmse")
    except Exception:
        return None

def parse_max_from_zip(zip_path):
    if not os.path.exists(zip_path):
        return None
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open("stats.json") as fh:
                stats = json.load(fh)
                return stats.get("max")
    except Exception:
        return None


def _find_gt_file(base_dir):
    gt_candidates = [
        "ground_truth.txt",
        "gt-tum.txt",
        "groundtruth.txt",
    ]
    for name in gt_candidates:
        p = os.path.join(base_dir, name)
        if os.path.isfile(p):
            return p
    return None


def discover_subsequences(sequence_roots, include_names=None):
    found = []
    include_set = set(include_names) if include_names else None

    for root in sequence_roots:
        if not os.path.isdir(root):
            continue

        root_name = os.path.basename(root.rstrip("/"))
        direct_seq_dir = os.path.join(root, "sequence")
        direct_gt_file = _find_gt_file(root)

        # Support direct sequence folders, e.g. oxford/5hz/christ_church_02
        if os.path.isdir(direct_seq_dir) and direct_gt_file:
            if include_set is None or root_name in include_set:
                found.append((os.path.dirname(root), root_name, root, direct_seq_dir, direct_gt_file))
            continue

        # Support parent roots containing many sequence folders.
        for name in sorted(os.listdir(root)):
            subdir = os.path.join(root, name)
            if not os.path.isdir(subdir):
                continue
            if include_set is not None and name not in include_set:
                continue

            seq_dir = os.path.join(subdir, "sequence")
            gt_file = _find_gt_file(subdir)
            if os.path.isdir(seq_dir) and gt_file:
                found.append((root, name, subdir, seq_dir, gt_file))

    return found


def build_model_and_args(checkpoint_path, checkpoint_name, device):
    args_path = os.path.join(checkpoint_path, "args.pkl")
    with open(args_path, "rb") as f:
        args = pickle.load(f)
    model_params = args["model_params"]

    model = VisionTransformer(
        img_size=model_params["image_size"],
        num_classes=model_params["num_classes"],
        patch_size=model_params["patch_size"],
        embed_dim=model_params["dim"],
        depth=model_params["depth"],
        num_heads=model_params["heads"],
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        num_frames=model_params["num_frames"],
        attention_type=model_params["attention_type"],
    )

    ckpt_file = os.path.join(checkpoint_path, f"{checkpoint_name}.pth")
    checkpoint = torch.load(ckpt_file, map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    preprocess = transforms.Compose(
        [
            transforms.Resize((model_params["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.34721234, 0.36705238, 0.36066107],
                std=[0.30737526, 0.31515116, 0.32020183],
            ),
        ]
    )

    return model, args, preprocess


def run_inference_for_subsequence(model, args, preprocess, seq_dir, gt_file, device):
    dataset = Oxford(
        data_path=seq_dir,
        gt_path=gt_file,
        window_size=args["window_size"],
        overlap=args["overlap"],
        transform=preprocess,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    pred_poses = torch.zeros((1, args["window_size"] - 1, 6), device=device)
    with torch.no_grad():
        with tqdm(loader, unit="batch", leave=False) as batches:
            for images, _ in batches:
                images = images.to(device)
                pred_pose = model(images.float())
                pred_pose = torch.reshape(pred_pose, (args["window_size"] - 1, 6)).to(device)
                pred_pose = pred_pose.unsqueeze(dim=0)
                pred_poses = torch.concat((pred_poses, pred_pose), dim=0)

    pred_poses = pred_poses[1:, :, :].cpu().detach().numpy()
    poses_rel = post_processing(pred_poses, args)
    poses_abs = recover_absolute_poses(poses_rel)
    return pred_poses, poses_abs


def main():
    parser = argparse.ArgumentParser(description="Evaluate TS-Former on Oxford subsequences with APE/RPE CSV output.")
    parser.add_argument("--checkpoint-path", default="checkpoints/test_no_leakage")
    parser.add_argument("--checkpoint-name", default="checkpoint_best")
    parser.add_argument(
        "--sequence-roots",
        nargs="+",
        default=["oxford/subsequences/church_02", "oxford/subsequences/church_05"],
    )
    parser.add_argument(
        "--include-names",
        nargs="+",
        default=None,
        help="Optional list of sequence/subsequence folder names to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        default="comparison/tsformer_eval/test_no_leakage_checkpoint_best",
    )
    parser.add_argument("--t-max-diff", type=float, default=0.5)
    args_cli = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Checkpoint: {args_cli.checkpoint_path}/{args_cli.checkpoint_name}.pth")

    model, model_args, preprocess = build_model_and_args(
        args_cli.checkpoint_path, args_cli.checkpoint_name, device
    )

    subsequences = discover_subsequences(args_cli.sequence_roots, include_names=args_cli.include_names)
    if not subsequences:
        print("No valid subsequences found.")
        return

    os.makedirs(args_cli.output_dir, exist_ok=True)
    csv_path = os.path.join(args_cli.output_dir, "metrics_summary.csv")

    rows = []
    for root, sub_name, sub_dir, seq_dir, gt_file in subsequences:
        seq_id = os.path.basename(root.rstrip("/"))
        run_name = f"{seq_id}_{sub_name}"
        print(f"\n=== {run_name} ===")

        out_dir = os.path.join(args_cli.output_dir, seq_id, sub_name)
        os.makedirs(out_dir, exist_ok=True)

        image_files, image_ts = extract_image_timestamps(seq_dir)
        gt_rows = read_tum_file(gt_file)
        matched_gt_ts = nearest_gt_timestamps_for_images(image_ts, gt_rows)

        status = "ok"
        notes = ""
        ape_max = None
        ape_rmse = None
        rpe_max = None
        rpe_rmse = None
        pred_count = 0
        gt_overlap_count = 0
        plot_file = ""

        if len(image_files) < 2:
            status = "skip"
            notes = "too few images"
        elif len(gt_rows) < 2:
            status = "skip"
            notes = "too few gt entries"
        else:
            pred_npy, poses_abs = run_inference_for_subsequence(
                model, model_args, preprocess, seq_dir, gt_file, device
            )
            np.save(os.path.join(out_dir, "pred_poses_raw.npy"), pred_npy)

            poses12_file = os.path.join(out_dir, "pred_poses_12col.txt")
            save_poses_12col(poses_abs, poses12_file)

            gt_overlap_file = os.path.join(out_dir, "gt_overlap.txt")
            gt_overlap_count = write_gt_overlap(gt_rows, image_ts, gt_overlap_file)

            pred_tum_file = os.path.join(out_dir, "pred_tum.txt")
            pred_count = poses12_to_tum_with_timestamps(
                poses12_file,
                matched_gt_ts[1:] if len(matched_gt_ts) > 1 else matched_gt_ts,
                pred_tum_file,
            )

            ape_zip = os.path.join(out_dir, "ape_metrics.zip")
            rpe_zip = os.path.join(out_dir, "rpe_metrics.zip")

            if pred_count < 2 or gt_overlap_count < 2:
                status = "skip"
                notes = "not enough overlap after timestamp filtering"
            else:
                cmd_ape = [
                    "evo_ape",
                    "tum",
                    gt_overlap_file,
                    pred_tum_file,
                    "-a",
                    "--t_max_diff",
                    str(args_cli.t_max_diff),
                    "--save_results",
                    ape_zip,
                ]
                cmd_rpe = [
                    "evo_rpe",
                    "tum",
                    gt_overlap_file,
                    pred_tum_file,
                    "-a",
                    "--t_max_diff",
                    str(args_cli.t_max_diff),
                    "--save_results",
                    rpe_zip,
                ]

                ok_ape, msg_ape = run_evo_metric(cmd_ape)
                ok_rpe, msg_rpe = run_evo_metric(cmd_rpe)

                ape_max = parse_max_from_zip(ape_zip) if ok_ape else None
                rpe_max = parse_max_from_zip(rpe_zip) if ok_rpe else None
                ape_rmse = parse_rmse_from_zip(ape_zip) if ok_ape else None
                rpe_rmse = parse_rmse_from_zip(rpe_zip) if ok_rpe else None

                if not ok_ape or not ok_rpe:
                    status = "partial" if (ok_ape or ok_rpe) else "fail"
                    notes = "; ".join(x for x in [msg_ape, msg_rpe] if x)

                plot_file = os.path.join(out_dir, f"{run_name}_xy_plot.pdf")
                ok_plot, msg_plot = run_evo_traj_plot(
                    gt_overlap_file,
                    pred_tum_file,
                    plot_file,
                    args_cli.t_max_diff,
                )
                if not ok_plot:
                    if notes:
                        notes += "; "
                    notes += msg_plot
                    if status == "ok":
                        status = "partial"


        print(
            f"images={len(image_files)} pred={pred_count} gt_overlap={gt_overlap_count} "
            f"APE_RMSE={ape_rmse} RPE_RMSE={rpe_rmse} status={status}"
        )

        rows.append(
            {
                "sequence_root": root,
                "subsequence": sub_name,
                "run_name": run_name,
                "num_images": len(image_files),
                "num_pred_poses": pred_count,
                "num_gt_overlap": gt_overlap_count,
                "ape_max_m": ape_max,
                "ape_rmse_m": ape_rmse,
                "rpe_max_m": rpe_max,
                "rpe_rmse_m": rpe_rmse,
                "status": status,
                "notes": notes,
            }
        )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sequence_root",
                "subsequence",
                "run_name",
                "num_images",
                "num_pred_poses",
                "num_gt_overlap",
                "ape_max_m",
                "ape_rmse_m",
                "rpe_max_m",
                "rpe_rmse_m",
                "status",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV summary: {csv_path}")


if __name__ == "__main__":
    main()
