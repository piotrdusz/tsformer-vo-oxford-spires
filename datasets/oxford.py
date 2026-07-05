import glob
import os
import pandas as pd
import numpy as np
from PIL import Image
from datasets.utils import rotation_to_euler
import torch
from torchvision import transforms


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """Convert a unit quaternion to a 3x3 rotation matrix."""
    R = np.array([
        [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),      2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),      1 - 2*(qx**2 + qz**2),  2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),      2*(qy*qz + qx*qw),      1 - 2*(qx**2 + qy**2)],
    ])
    return R


class Oxford(torch.utils.data.Dataset):
    """
    Dataloader for Oxford RobotCar dataset.

    Ground truth file format (one line per pose):
        timestamp x y z qx qy qz qw

    Images are JPEGs named by their timestamp (e.g. 1710926321.899711842.jpg)
    stored in a flat directory.  Each image is matched to the nearest
    ground-truth timestamp.

    Arguments:
        data_path  {str}  -- path to the folder containing the JPEG frames
        gt_path    {str}  -- path to ground_truth.txt
        window_size {int} -- number of frames per window (same as training)
        overlap    {int}  -- overlap between consecutive windows
        transform         -- torchvision transform applied to each frame
    """

    def __init__(self,
                 data_path="oxford/sequence",
                 gt_path="oxford/ground_truth.txt",
                 window_size=3,
                 overlap=1,
                 transform=None):

        self.data_path = data_path
        self.gt_path = gt_path
        self.window_size = window_size
        self.overlap = overlap
        self.transform = transform

        # Normalization constants from KITTI (model was trained on KITTI).
        # These are applied to the GT so that it is in the same normalised
        # space as the model output.  Replace with Oxford-specific values if
        # you re-train on Oxford data.
        self.mean_angles = np.array([1.7061e-5,  9.5582e-4, -5.5258e-5])
        self.std_angles  = np.array([2.8256e-3,  1.7771e-2,  3.2326e-3])
        self.mean_t      = np.array([-8.6736e-5, -1.6038e-2,  9.0033e-1])
        self.std_t       = np.array([2.5584e-2,   1.8545e-2,  3.0352e-1])

        gt_dict = self._read_gt()
        frames, gt = self._match_frames_to_gt(gt_dict)

        # Build a flat dataframe with one row per frame.
        # Columns 0-6: [x, y, z, qx, qy, qz, qw]
        data = pd.DataFrame({"gt": gt})
        data = data["gt"].apply(pd.Series)
        data["frames"] = frames
        self.data = data
        self.windowed_data = self._create_windowed_dataframe(data)

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.windowed_data["w_idx"].unique())

    def __getitem__(self, idx):
        data = self.windowed_data.loc[self.windowed_data["w_idx"] == idx, :]

        # --- load images ---
        imgs = []
        for fname in data["frames"].values:
            img = Image.open(fname).convert("RGB")
            img = self.transform(img)
            imgs.append(img.unsqueeze(0))
        imgs = np.concatenate(imgs, axis=0)       # (T, C, H, W)
        imgs = imgs.transpose(1, 0, 2, 3)         # (C, T, H, W)

        # --- compute relative poses [window_size-1, 6] ---
        gt_vals = data.loc[:, [i for i in range(7)]].values  # (T, 7)
        y_list = []
        pose_prev = None
        for gt_row in gt_vals:
            x_t, y_t, z_t = gt_row[0], gt_row[1], gt_row[2]
            qx, qy_q, qz_q, qw = gt_row[3], gt_row[4], gt_row[5], gt_row[6]
            R = quaternion_to_rotation_matrix(qx, qy_q, qz_q, qw)
            t = np.array([x_t, y_t, z_t])
            pose = np.vstack([np.hstack([R, t.reshape(3, 1)]),
                              [0., 0., 0., 1.]])

            if pose_prev is not None:
                rel = np.dot(np.linalg.inv(pose_prev), pose)
                angles = rotation_to_euler(rel[:3, :3], seq='zyx')
                t_rel  = rel[:3, 3]
                angles = (np.asarray(angles) - self.mean_angles) / self.std_angles
                t_rel  = (np.asarray(t_rel)  - self.mean_t)      / self.std_t
                y_list.append(list(angles) + list(t_rel))

            pose_prev = pose

        y = np.asarray(y_list).flatten()
        return imgs, y

    # ------------------------------------------------------------------
    def _read_gt(self):
        """Return {timestamp_float: [x, y, z, qx, qy, qz, qw]}."""
        gt_dict = {}
        with open(self.gt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 8:
                    ts = float(parts[0])
                    gt_dict[ts] = [float(v) for v in parts[1:]]
        return gt_dict

    def _match_frames_to_gt(self, gt_dict):
        """
        For every JPEG in data_path, find the nearest GT timestamp.
        Returns parallel lists (frame_paths, gt_values).
        """
        image_files = sorted(glob.glob(os.path.join(self.data_path, "*.jpg")))
        gt_timestamps = np.array(sorted(gt_dict.keys()))

        frames, gt = [], []
        for img_path in image_files:
            ts_str = os.path.splitext(os.path.basename(img_path))[0]
            ts = float(ts_str)
            nearest_idx = np.argmin(np.abs(gt_timestamps - ts))
            nearest_ts  = gt_timestamps[nearest_idx]
            frames.append(img_path)
            gt.append(gt_dict[nearest_ts])
        return frames, gt

    def _create_windowed_dataframe(self, df):
        windowed_df = pd.DataFrame()
        w_idx = 0
        row_idx = 0
        while row_idx + self.window_size <= len(df):
            rows = df.iloc[row_idx:(row_idx + self.window_size)].copy()
            rows["w_idx"] = w_idx
            row_idx += self.window_size - self.overlap
            w_idx   += 1
            windowed_df = pd.concat([windowed_df, rows], ignore_index=True)
        windowed_df.reset_index(drop=True)
        return windowed_df
