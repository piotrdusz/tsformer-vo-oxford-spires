#!/usr/bin/env python3
"""
Convert predicted poses from 12-value format to TUM format using timestamps from ground truth.
The 12 values represent a flattened 4x3 pose matrix [R | t] where R is 3x3 rotation and t is translation.
"""

import numpy as np
from scipy.spatial.transform import Rotation
import sys

def matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion (x, y, z, w)"""
    rot = Rotation.from_matrix(R)
    quat = rot.as_quat()  # returns [x, y, z, w]
    return quat

def convert_poses_to_tum(ground_truth_file, predicted_poses_file, output_file):
    """
    Convert predicted poses to TUM format using timestamps from ground truth.
    
    Args:
        ground_truth_file: Path to ground truth TUM file (for timestamps)
        predicted_poses_file: Path to predicted poses file (12 values per row)
        output_file: Path to output TUM format file
    """
    
    # Read ground truth to extract timestamps
    gt_timestamps = []
    with open(ground_truth_file, 'r') as f:
        for line in f:
            values = line.strip().split()
            if values:  # Skip empty lines
                gt_timestamps.append(float(values[0]))
    
    print(f"Read {len(gt_timestamps)} ground truth timestamps")
    
    # Read predicted poses
    pred_poses = []
    with open(predicted_poses_file, 'r') as f:
        for line in f:
            values = line.strip().split()
            if values:  # Skip empty lines
                pred_poses.append([float(v) for v in values])
    
    print(f"Read {len(pred_poses)} predicted pose entries")
    
    # Check dimensions match
    if len(gt_timestamps) != len(pred_poses):
        print(f"WARNING: Different number of timestamps ({len(gt_timestamps)}) and poses ({len(pred_poses)})")
        min_len = min(len(gt_timestamps), len(pred_poses))
        gt_timestamps = gt_timestamps[:min_len]
        pred_poses = pred_poses[:min_len]
        print(f"Using first {min_len} entries")
    
    # Convert to TUM format and save
    with open(output_file, 'w') as f:
        for timestamp, pose_values in zip(gt_timestamps, pred_poses):
            if len(pose_values) != 12:
                print(f"WARNING: Expected 12 values, got {len(pose_values)}")
                continue
            
            # Reshape to get R (3x3) and t (3x1)
            # Values are: R[0,0], R[0,1], R[0,2], t[0], R[1,0], R[1,1], R[1,2], t[1], R[2,0], R[2,1], R[2,2], t[2]
            R = np.array([
                [pose_values[0], pose_values[1], pose_values[2]],
                [pose_values[4], pose_values[5], pose_values[6]],
                [pose_values[8], pose_values[9], pose_values[10]]
            ])
            t = np.array([pose_values[3], pose_values[7], pose_values[11]])
            
            # Convert rotation matrix to quaternion
            quat = matrix_to_quaternion(R)
            
            # Write in TUM format: timestamp x y z qx qy qz qw
            line = f"{timestamp} {t[0]} {t[1]} {t[2]} {quat[0]} {quat[1]} {quat[2]} {quat[3]}\n"
            f.write(line)
    
    print(f"Saved {len(gt_timestamps)} poses to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python convert_poses_to_tum.py <ground_truth_file> <predicted_poses_file> <output_file>")
        print("Example: python convert_poses_to_tum.py oxford/ground_truth.txt checkpoints/oxford_finetune_model2/checkpoint_best/pred_poses/oxford.txt checkpoints/oxford_finetune_model2/checkpoint_best/pred_poses/oxford_tum.txt")
        sys.exit(1)
    
    gt_file = sys.argv[1]
    pred_file = sys.argv[2]
    out_file = sys.argv[3]
    
    convert_poses_to_tum(gt_file, pred_file, out_file)
