import matplotlib.pyplot as plt
import numpy as np
import queue
import pickle
import os
from datasets.kitti import KITTI
from datasets.oxford import Oxford
from datasets.utils import euler_to_rotation


def save_trajectory(poses, sequence, save_dir):
    """
    Save predicted poses in .txt file
    Args:
        poses {ndarray}: list with all 4x4 pose matrix
        sequence {str}: sequence of KITTI dataset
        save_dir {str}: path to save pose
    """
    # create directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    output_filename = os.path.join(save_dir, "{}.txt".format(sequence))
    with open(output_filename, "w") as f:
        for pose in poses:
            pose = pose.flatten()[:12]
            line = " ".join([str(x) for x in pose]) + "\n"
            f.write(line)


def post_processing(pred_poses, args):

    if args["window_size"] == 2:
        pred_poses = pred_poses.squeeze(1)
        return np.asarray(pred_poses)

    num_batchs = pred_poses.shape[0]

    # get poses in overlaped frames
    q = queue.Queue(args["window_size"]-1) #The max size is 5.
    idx = 0
    poses = []

    while not q.full():
        q.put(pred_poses[idx, :, :])
        idx = idx + 1

    while idx < num_batchs:
        # process first full queue
        if idx == (args["window_size"]-1):
            poses.append(q.queue[0][0, :])

            # implemented for specific case window_size = 3 and overlap = 2
            avg_pose = (q.queue[0][1, :] + q.queue[1][0, :])/2
            poses.append(avg_pose)

            if args["window_size"] == 4:  
                # implemented for specific case window_size = 4 and overlap = 3
                avg_pose = (q.queue[0][2, :] + q.queue[1][1, :] + q.queue[2][0, :])/3
                poses.append(avg_pose)

        elif idx < (num_batchs - 1):
            if args["window_size"] == 3:
                # implemented for specific case window_size = 3 and overlap = 2
                avg_pose = (q.queue[0][1, :] + q.queue[1][0, :])/2
                poses.append(avg_pose)

            elif args["window_size"] == 4:  
                # implemented for specific case window_size = 4 and overlap = 3
                avg_pose = (q.queue[0][2, :] + q.queue[1][1, :] + q.queue[2][0, :])/3
                poses.append(avg_pose)

        # process last full queue (idx == num_batchs-1)
        else:
            if args["window_size"] == 3:
                # implemented for specific case window_size = 3 and overlap = 2
                poses.append(q.queue[1][1,:])

            elif args["window_size"] == 4:
                # implemented for specific case window_size = 4 and overlap = 2
                avg_pose = (q.queue[1][2, :] + q.queue[2][1, :])/2
                poses.append(avg_pose)
                poses.append(q.queue[2][2,:])
        
            idx = idx + 1

        # update queue
        if idx < (num_batchs-1):
            idx = idx + 1
            first = q.get() #dequeue first element
            q.put(pred_poses[idx, :, :])

    return np.asarray(poses)


def recover_trajectory_and_poses(poses):

    predicted_poses = []
    # recover predicted trajectory
    predicted_trajectory = []
    for i in range(len(poses)-1):
        if i == 0:
            T = np.eye(4)

        angles = poses[i, :3]
        t = poses[i, 3:]

        # undo normalization
        mean_angles = np.array([1.7061e-5, 9.5582e-4, -5.5258e-5])
        std_angles = np.array([2.8256e-3, 1.7771e-2, 3.2326e-3])
        mean_t = np.array([-8.6736e-5, -1.6038e-2, 9.0033e-1])
        std_t = np.array([2.5584e-2, 1.8545e-2, 3.0352e-1])

        [x, y, z] = np.multiply(angles, std_angles) + mean_angles
        t = np.multiply(t, std_t) + mean_t
        R = np.asarray(euler_to_rotation(x, y, z, seq = 'zyx'))

        T_r = np.concatenate((np.concatenate([R, np.reshape(t, (3,1))], axis=1) , [[0.0, 0.0, 0.0, 1.0]] ), axis=0)
        T_abs = np.dot(T,T_r)
        T = T_abs

        predicted_poses.append(T)
        predicted_trajectory.append(T_abs[:3, 3])

    return predicted_poses, predicted_trajectory
  

if __name__ == "__main__":
  
    ckpt_path = "checkpoints/oxford_finetune_model3/"
    ckpt_name = "checkpoint_best"

    # Set dataset to "kitti" or "oxford"
    dataset_name = "oxford"

    kitti_sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    oxford_data_path = "oxford/sequence"
    oxford_gt_path   = "oxford/ground_truth.txt"

    # read hyperparameters and configuration
    with open(os.path.join(ckpt_path, "args.pkl"), 'rb') as f:
        args = pickle.load(f)
    f.close()

    ckpt_path = os.path.join(ckpt_path, ckpt_name)
    args["checkpoint_path"] = ckpt_path

    sequences = kitti_sequences if dataset_name == "kitti" else ["oxford"]

    # plot trajectory and ground truth
    for sequence in sequences:
        # read ground test data and predicted poses
        pred_path = os.path.join(args["checkpoint_path"], "pred_poses_{}.npy".format(sequence))
        pred_poses = np.load(pred_path)

        # post processing and recover trajectory
        poses = post_processing(pred_poses, args)
        pred_poses, pred_trajectory = recover_trajectory_and_poses(poses)

        save_trajectory(pred_poses, sequence,
                        save_dir=os.path.join(args["checkpoint_path"], "pred_poses"))

        # get ground truth trajectories
        if dataset_name == "kitti":
            test_data = KITTI(sequences=[sequence], window_size=args["window_size"])
            gt_poses = test_data.windowed_data.loc[
                test_data.windowed_data["sequence"] == sequence, [3, 7, 11]
            ]
            gt_x = [p[0] for p in gt_poses.values]
            gt_z = [p[2] for p in gt_poses.values]
        else:  # oxford
            test_data = Oxford(data_path=oxford_data_path,
                               gt_path=oxford_gt_path,
                               window_size=args["window_size"],
                               overlap=args["overlap"])
            # Oxford GT columns: 0=x, 1=y, 2=z  (world-frame positions)
            gt_poses = test_data.windowed_data.loc[:, [0, 1]]
            gt_x = gt_poses[0].values
            gt_y = gt_poses[1].values

        plt.figure()
        pred_trajectory = np.asarray(pred_trajectory)
        plt.plot([p[0] for p in pred_trajectory], [p[1] for p in pred_trajectory], "b")  # estimated
        plt.plot(gt_x, gt_y, "r")  # ground truth
        plt.grid()
        plt.title("VO - Seq {}".format(sequence))
        plt.xlabel("Translation in x direction [m]")
        plt.ylabel("Translation in y direction [m]")
        plt.legend(["estimated", "ground truth"])

        # create plots folder
        if not os.path.exists(os.path.join(args["checkpoint_path"], "plots")):
            os.makedirs(os.path.join(args["checkpoint_path"], "plots"))
        plt.savefig(os.path.join(args["checkpoint_path"], "plots", "pred_traj_{}.png".format(sequence)))
  
