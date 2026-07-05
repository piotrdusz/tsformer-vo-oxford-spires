from torchvision import transforms
from datasets.kitti import KITTI
from datasets.oxford import Oxford
import pickle
import torch
from timesformer.models.vit import VisionTransformer
from functools import partial
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm


checkpoint_path = "checkpoints/oxford_finetune_model3"
checkpoint_name = "checkpoint_best"

# Set dataset to "kitti" or "oxford"
dataset_name = "oxford"

# KITTI: list of sequence IDs to evaluate
kitti_sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
# Oxford: single sequence (paths are set in the Oxford class defaults)
oxford_data_path = "oxford/sequence"
oxford_gt_path   = "oxford/ground_truth.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"


# read hyperparameters and configuration
with open(os.path.join(checkpoint_path, "args.pkl"), 'rb') as f:
    args = pickle.load(f)
f.close()
model_params = args["model_params"]
args["checkpoint_path"] = checkpoint_path
print(args)

# preprocessing operation
preprocess = transforms.Compose([
    transforms.Resize((model_params["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.34721234, 0.36705238, 0.36066107],
        std=[0.30737526, 0.31515116, 0.32020183]),
])

# build and load model
model = VisionTransformer(img_size=model_params["image_size"],
                          num_classes=model_params["num_classes"],
                          patch_size=model_params["patch_size"],
                          embed_dim=model_params["dim"],
                          depth=model_params["depth"],
                          num_heads=model_params["heads"],
                          mlp_ratio=4,
                          qkv_bias=True,
                          norm_layer=partial(nn.LayerNorm, eps=1e-6),
                          drop_rate=0.,
                          attn_drop_rate=0.,
                          drop_path_rate=0.1,
                          num_frames=model_params["num_frames"],
                          attention_type=model_params["attention_type"])

checkpoint = torch.load(os.path.join(args["checkpoint_path"], "{}.pth".format(checkpoint_name)),
                        map_location=torch.device(device))
model.load_state_dict(checkpoint['model_state_dict'])
if torch.cuda.is_available():
    model.cuda()


save_dir = os.path.join(args["checkpoint_path"], checkpoint_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if dataset_name == "oxford":
    dataset = Oxford(data_path=oxford_data_path,
                     gt_path=oxford_gt_path,
                     window_size=args["window_size"],
                     overlap=args["overlap"],
                     transform=preprocess)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    with tqdm(test_loader, unit="batch") as batches:
        pred_poses = torch.zeros((1, args["window_size"] - 1, 6), device=device)
        batches.set_description("Oxford sequence")
        for images, gt in batches:
            if torch.cuda.is_available():
                images = images.cuda()
            with torch.no_grad():
                model.eval()
                model.training = False
                pred_pose = model(images.float())
                pred_pose = torch.reshape(pred_pose, (args["window_size"] - 1, 6)).to(device)
                pred_pose = pred_pose.unsqueeze(dim=0)
                pred_poses = torch.concat((pred_poses, pred_pose), dim=0)

    pred_poses = pred_poses[1:, :, :].cpu().detach().numpy()
    np.save(os.path.join(save_dir, "pred_poses_oxford.npy"), pred_poses)

else:  # kitti
    for sequence in kitti_sequences:
        dataset = KITTI(transform=preprocess, sequences=[sequence],
                        window_size=args["window_size"], overlap=args["overlap"])
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        with tqdm(test_loader, unit="batch") as batches:
            pred_poses = torch.zeros((1, args["window_size"] - 1, 6), device=device)
            batches.set_description(f"Sequence {sequence}")
            for images, gt in batches:
                if torch.cuda.is_available():
                    images, gt = images.cuda(), gt.cuda()
                with torch.no_grad():
                    model.eval()
                    model.training = False
                    pred_pose = model(images.float())
                    pred_pose = torch.reshape(pred_pose, (args["window_size"] - 1, 6)).to(device)
                    pred_pose = pred_pose.unsqueeze(dim=0)
                    pred_poses = torch.concat((pred_poses, pred_pose), dim=0)

        pred_poses = pred_poses[1:, :, :].cpu().detach().numpy()
        np.save(os.path.join(save_dir, "pred_poses_{}.npy".format(sequence)), pred_poses)
