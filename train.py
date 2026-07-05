import os
import torch
import torch.optim as optim
from tqdm import tqdm
from datasets.kitti import KITTI
from datasets.oxford import Oxford
from build_model import build_model
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
import pickle
import json
import argparse


torch.manual_seed(2023)


def val_epoch(model, val_loader, criterion, args):
    epoch_loss = 0
    with tqdm(val_loader, unit="batch") as tepoch:
        for images, gt in tepoch:
            tepoch.set_description(f"Validating ")
            # for batch_idx, (images, odom) in enumerate(train_loader):
            if torch.cuda.is_available():
                images, gt = images.cuda(), gt.cuda()

            # predict pose
            estimated_pose = model(images.float())

            # compute loss
            loss = compute_loss(estimated_pose, gt, criterion, args)

            epoch_loss += loss.item()
            tepoch.set_postfix(val_loss=loss.item())

    return epoch_loss / len(val_loader)


def train_epoch(model, train_loader, criterion, optimizer, epoch, tensorboard_writer, args):
    epoch_loss = 0
    iter = (epoch - 1) * len(train_loader) + 1

    with tqdm(train_loader, unit="batch") as tepoch:
        for images, gt in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            # for batch_idx, (images, odom) in enumerate(train_loader):
            if torch.cuda.is_available():
                images, gt = images.cuda(), gt.cuda()

            # predict pose
            estimated_pose = model(images.float())

            # compute loss
            loss = compute_loss(estimated_pose, gt, criterion, args)

            # compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            tepoch.set_postfix(loss=loss.item())

            # log tensorboard
            tensorboard_writer.add_scalar('training_loss', loss.item(), iter)

            iter += 1
    return epoch_loss / len(train_loader)  
  

def train(model, train_loader, val_loader, criterion, optimizer, tensorboard_writer, args):
    checkpoint_path = args["checkpoint_path"]
    epochs = args["epoch"]
    best_val = args["best_val"]
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(args["epoch_init"], epochs):
        # training for one epoch
        model.train()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch, tensorboard_writer, args)

        # validate model
        if val_loader:
            with torch.no_grad():
                model.eval()
                val_loss = val_epoch(model, val_loader, criterion, args)

            print(f"Epoch: {epoch} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f} \n")

            # save best mode
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                "best_val": best_val,
            }
            if val_loss < best_val:
                print(f"Saving new best model -- loss decreased from {best_val:.6f} to {val_loss:.6f} \n")
                best_val = val_loss
                state["best_val"] = best_val
                torch.save(state, os.path.join(checkpoint_path, "checkpoint_best.pth"))

            # log validation loss in TensorBoard
            tensorboard_writer.add_scalar("val_loss", val_loss, epoch)

        # save checkpoint every 20 epochs
        if not epoch%20:
            torch.save(state, os.path.join(checkpoint_path, "checkpoint_e{}.pth".format(epoch))) 
        # save last checkpoint
        torch.save(state, os.path.join(checkpoint_path, "checkpoint_last.pth"))  

        # log loss in TensorBoard
        tensorboard_writer.add_scalar("train_loss", train_loss, epoch)
    return


def get_optimizer(params, args):
    method = args["optimizer"]

    # initialize the optimizer
    if method == "Adam":
        optimizer = optim.Adam(params, lr=args["lr"])
    elif method == "SGD":
        optimizer = optim.SGD(params, lr=args["lr"],
                              momentum=args["momentum"],
                              weight_decay=args["weight_decay"])
    elif method == "RAdam":
        optimizer = optim.RAdam(params, lr=args["lr"])
    elif method == "Adagrad":
        optimizer = optim.Adagrad(params, lr=args["lr"],
                                  weight_decay=args["weight_decay"])

    # load checkpoint
    if args["checkpoint"] is not None:
        checkpoint = torch.load(os.path.join(args["checkpoint_path"], args["checkpoint"]))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return optimizer


def compute_loss(y_hat, y, criterion, args):
    if args["weighted_loss"] == None:
        loss = criterion(y_hat, y.float())
    else:
        y = torch.reshape(y, (y.shape[0], args["window_size"]-1, 6))
        gt_angles = y[:, :, :3].flatten()
        gt_translation = y[:, :, 3:].flatten()

        # predict pose
        y_hat = torch.reshape(y_hat, (y_hat.shape[0], args["window_size"]-1, 6))
        estimated_angles = y_hat[:, :, :3].flatten()
        estimated_translation = y_hat[:, :, 3:].flatten()

        # compute custom loss
        k = args["weighted_loss"]
        loss_angles = k * criterion(estimated_angles, gt_angles.float())
        loss_translation = criterion(estimated_translation, gt_translation.float())
        loss =  loss_angles + loss_translation   
    return loss


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train TSformer model")
    parser.add_argument("--config", type=str, default="configs/config_oxford.json",
                        help="Path to config file (default: configs/config_oxford.json)")
    cmd_args = parser.parse_args()

    # Load config from file
    with open(cmd_args.config, 'r') as f:
        config = json.load(f)

    dataset_name = config["dataset_name"]
    
    # Load source checkpoint args
    source_ckpt_path = config["source_ckpt_path"]
    source_ckpt_name = config["source_ckpt_name"]
    
    with open(os.path.join(source_ckpt_path, "args.pkl"), "rb") as f:
        source_args = pickle.load(f)

    # Initialize args from config
    args = {
        "bsize": config["bsize"],
        "val_split": config["val_split"],
        "window_size": source_args["window_size"],
        "overlap": source_args["overlap"],
        "optimizer": config["optimizer"],
        "lr": config["lr"],
        "momentum": config["momentum"],
        "weight_decay": config["weight_decay"],
        "epoch": config["epoch"],
        "weighted_loss": source_args.get("weighted_loss", None),
        "pretrained_ViT": config["pretrained_ViT"],
        "checkpoint_path": config["checkpoint_path"],
        "checkpoint": config.get("checkpoint", None),
        "finetune_from": os.path.join(source_ckpt_path, source_ckpt_name + ".pth"),
        "dataset_name": dataset_name,
        "oxford_data_path": config.get("oxford_data_path", "oxford/sequence"),
        "oxford_gt_path": config.get("oxford_gt_path", "oxford/ground_truth.txt"),
    }

    # Get model params from source checkpoint
    model_params = source_args["model_params"].copy()
    model_params["num_frames"] = args["window_size"]
    model_params["num_classes"] = 6 * (args["window_size"] - 1)
    args["model_params"] = model_params

    # Create checkpoints folder
    if not os.path.exists(args["checkpoint_path"]):
        os.makedirs(args["checkpoint_path"])

    with open(os.path.join(args["checkpoint_path"], 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)
    with open(os.path.join(args["checkpoint_path"], 'args.txt'), 'w') as f:
        f.write(json.dumps(args))

    # TensorBoard writer
    TensorBoardWriter = SummaryWriter(log_dir=args["checkpoint_path"])

    # Preprocessing operation
    preprocess = transforms.Compose([
        transforms.Resize((model_params["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.34721234, 0.36705238, 0.36066107],
            std=[0.30737526, 0.31515116, 0.32020183]),
    ])

    # Train and val dataloader
    print("Using CUDA: ", torch.cuda.is_available())
    print("Loading data...")
    if dataset_name == "oxford":
        dataset = Oxford(data_path=args["oxford_data_path"],
                         gt_path=args["oxford_gt_path"],
                         window_size=args["window_size"],
                         overlap=args["overlap"],
                         transform=preprocess)
    else:
        dataset = KITTI(window_size=args["window_size"], overlap=args["overlap"], transform=preprocess)
    nb_val = round(args["val_split"] * len(dataset))

    train_data, val_data = random_split(dataset, [len(dataset) - nb_val, nb_val])
    
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args["bsize"],
                                               shuffle=True,
                                               )
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=1,
                                             shuffle=False,
                                             )

    # Build and load model
    print("Building model...")
    model, args = build_model(args, model_params)

    # Loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = get_optimizer(model.parameters(), args)

    # Train network
    print(20*"--" +  " Training " + 20*"--")
    train(model, train_loader, val_loader, criterion, optimizer, TensorBoardWriter, args)
