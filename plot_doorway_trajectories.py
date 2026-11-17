import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# --- PUBLICATION PLOT SETTINGS ---
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 22,
    'lines.linewidth': 2.5
})

# --- DIRECTORIES ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Map output figure suffix to TSformer bag folder and full-sequence GT file.
SEQUENCE_CONFIG = {
    'christ-church-02': {
        'result_dir': os.path.join(BASE_DIR, 'comparison', 'tsformer_eval', 'test_no_leakage_checkpoint_best', 'church_02'),
        'full_gt_file': os.path.join(BASE_DIR, 'oxford', '5hz', 'christ_church_02', 'gt-tum.txt'),
    },
    'christ-church-05': {
        'result_dir': os.path.join(BASE_DIR, 'comparison', 'tsformer_eval', 'test_no_leakage_checkpoint_best', 'church_05'),
        'full_gt_file': os.path.join(BASE_DIR, 'oxford', '5hz', 'christ_church_05', 'gt-tum.txt'),
    },
}

def load_tum_data(filepath):
    """Load trajectory data from a TUM format txt file."""
    try:
        data = np.loadtxt(filepath)
        if len(data) == 0:
            return None
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def align_trajectories(traj_times, traj_pts, gt_times, gt_pts, time_thresh=0.05):
    """
    Aligns trajectory to ground truth using timestamp synchronization 
    and Kabsch (Umeyama) rigid transformation.
    """
    matched_traj = []
    matched_gt = []
    
    for i, t in enumerate(traj_times):
        idx = np.abs(gt_times - t).argmin()
        if np.abs(gt_times[idx] - t) < time_thresh:
            matched_traj.append(traj_pts[i])
            matched_gt.append(gt_pts[idx])
            
    if len(matched_traj) < 3:
        return None, None
        
    matched_traj = np.array(matched_traj)
    matched_gt = np.array(matched_gt)
    
    centroid_traj = np.mean(matched_traj, axis=0)
    centroid_gt = np.mean(matched_gt, axis=0)
    
    centered_traj = matched_traj - centroid_traj
    centered_gt = matched_gt - centroid_gt
    
    H = np.dot(centered_traj.T, centered_gt)
    U, S, Vt = np.linalg.svd(H)
    rot = np.dot(Vt.T, U.T)
    
    if np.linalg.det(rot) < 0:
        Vt[2, :] *= -1
        rot = np.dot(Vt.T, U.T)
        
    trans = centroid_gt - np.dot(centroid_traj, rot.T)
    
    return rot, trans

def main():
    for seq_name, seq_cfg in SEQUENCE_CONFIG.items():
        seq_dir = seq_cfg['result_dir']
        full_gt_file = seq_cfg['full_gt_file']
        print(f"Processing sequence: {seq_name}")

        if not os.path.isdir(seq_dir):
            print(f"Directory not found: {seq_dir}")
            continue

        full_gt_data = load_tum_data(full_gt_file)
        if full_gt_data is None:
            print(f"Full GT file not found or invalid: {full_gt_file}")
            continue
        full_gt_pts = full_gt_data[:, 1:4]

        bag_dirs = sorted(glob.glob(os.path.join(seq_dir, 'oxford_doorway_bag_*')))
        if not bag_dirs:
            print(f"No doorway bag directories found in: {seq_dir}")
            continue

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        # fig.suptitle(f'TSformer Doorway Trajectories: {seq_name}', weight='bold')
        ax.set_title('TSformer', fontweight='bold', pad=15)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')

        ax.plot(full_gt_pts[:, 0], full_gt_pts[:, 1], color='lightgray', linestyle='--',
            linewidth=1.8, label='Global GT', zorder=1)

        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        plotted_any = False

        for bag_dir in bag_dirs:
            bag_name = os.path.basename(bag_dir)
            pred_file = os.path.join(bag_dir, 'pred_tum.txt')
            gt_file = os.path.join(bag_dir, 'gt_overlap.txt')

            traj_data = load_tum_data(pred_file)
            gt_data = load_tum_data(gt_file)

            if traj_data is None or gt_data is None:
                print(f"Skipping {bag_name}: missing or invalid pred/gt file")
                continue

            traj_times = traj_data[:, 0]
            traj_pts = traj_data[:, 1:4]
            gt_times = gt_data[:, 0]
            gt_pts = gt_data[:, 1:4]

            R_mat, t_vec = align_trajectories(traj_times, traj_pts, gt_times, gt_pts)
            if R_mat is None:
                print(f"Skipping {bag_name}: not enough timestamp overlap for alignment")
                continue

            aligned_pts = np.dot(traj_pts, R_mat.T) + t_vec

            min_x = min(min_x, np.min(aligned_pts[:, 0]))
            max_x = max(max_x, np.max(aligned_pts[:, 0]))
            min_y = min(min_y, np.min(aligned_pts[:, 1]))
            max_y = max(max_y, np.max(aligned_pts[:, 1]))

            try:
                trans_idx = int(bag_name.split('_bag_')[-1]) + 1
            except ValueError:
                trans_idx = '?'

            ax.plot(aligned_pts[:, 0], aligned_pts[:, 1],
                    label=f'Transition {trans_idx}', linewidth=2.2, zorder=2)
            plotted_any = True

        if not plotted_any:
            print(f"No valid trajectories plotted for: {seq_name}")
            plt.close(fig)
            continue

        margin = 10.0
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - 2.0, max_y + margin + 2.0)
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        output_filename = f'doorway_transitions_zoom_{seq_name}.png'
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {output_filename}")
        
        # 3. MEMORY MANAGEMENT: Safely close the figure before the next sequence starts
        plt.close(fig)

if __name__ == '__main__':
    main()