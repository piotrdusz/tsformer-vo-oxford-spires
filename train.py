import os
import torch
import torch.optim as optim
from tqdm import tqdm
from datasets.kitti import KITTI
from datasets.oxford import Oxford
from build_model import build_model
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset, ConcatDataset
import pickle
import json
import argparse


torch.manual_seed(2023)


def chronological_split(dataset, val_split):
    """Split dataset by index order: first part train, last part validation."""
    dataset_size = len(dataset)
    if dataset_size < 2:
        raise ValueError("Dataset must contain at least 2 samples for train/val split.")

    nb_val = round(val_split * dataset_size)
    nb_val = max(1, min(dataset_size - 1, nb_val))
    nb_train = dataset_size - nb_val

    train_indices = range(0, nb_train)
    val_indices = range(nb_train, dataset_size)
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def build_oxford_dataset(data_path, gt_path, args, preprocess):
    return Oxford(data_path=data_path,
                  gt_path=gt_path,
                  window_size=args["window_size"],
                  overlap=args["overlap"],
                  transform=preprocess)


def build_oxford_datasets(data_paths, gt_paths, args, preprocess):
    if len(data_paths) != len(gt_paths):
        raise ValueError("Oxford data paths and gt paths must have the same length")

    datasets = []
    for data_path, gt_path in zip(data_paths, gt_paths):
        datasets.append(build_oxford_dataset(data_path, gt_path, args, preprocess))
    return datasets


def concat_if_needed(datasets):
    if len(datasets) == 0:
        raise ValueError("At least one dataset is required")
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def evaluate_test_once(model, test_loader, criterion, tensorboard_writer, args):
    if test_loader is None:
        return

    best_checkpoint_path = os.path.join(args["checkpoint_path"], "checkpoint_best.pth")
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best checkpoint for test evaluation: {best_checkpoint_path}")
    else:
        print("Best checkpoint not found. Evaluating test set with current model weights.")

    with torch.no_grad():
        model.eval()
        test_loss = val_epoch(model, test_loader, criterion, args)

    print(f"Final test_loss: {test_loss:.4f}")
    tensorboard_writer.add_scalar("test_loss", test_loss, args["epoch"])


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
  

def train(model,
          train_loader,
          val_loader,
          criterion,
          optimizer,
          tensorboard_writer,
          args,
          rotating_train_datasets=None):
    checkpoint_path = args["checkpoint_path"]
    epochs = args["epoch"]
    best_val = args["best_val"]
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    if rotating_train_datasets is not None and len(rotating_train_datasets) < 2:
        raise ValueError("Rotating validation requires at least 2 training sequences")

    for epoch in range(args["epoch_init"], epochs):
        if rotating_train_datasets is not None:
            val_seq_idx = epoch % len(rotating_train_datasets)
            val_data = rotating_train_datasets[val_seq_idx]
            train_parts = [
                dataset for idx, dataset in enumerate(rotating_train_datasets)
                if idx != val_seq_idx
            ]

            train_data = concat_if_needed(train_parts)
            train_loader = torch.utils.data.DataLoader(train_data,
                                                       batch_size=args["bsize"],
                                                       shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_data,
                                                     batch_size=1,
                                                     shuffle=False)
            print(f"Epoch {epoch}: rotating validation sequence index = {val_seq_idx}")
            tensorboard_writer.add_scalar("rotating_val_seq_idx", val_seq_idx, epoch)

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
        "oxford_data_paths": config.get("oxford_data_paths", None),
        "oxford_gt_paths": config.get("oxford_gt_paths", None),
        "oxford_train_data_paths": config.get("oxford_train_data_paths", None),
        "oxford_train_gt_paths": config.get("oxford_train_gt_paths", None),
        "oxford_test_data_path": config.get("oxford_test_data_path", None),
        "oxford_test_gt_path": config.get("oxford_test_gt_path", None),
        "oxford_test_data_paths": config.get("oxford_test_data_paths", None),
        "oxford_test_gt_paths": config.get("oxford_test_gt_paths", None),
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
    rotating_train_datasets = None
    test_loader = None
    if dataset_name == "oxford":
        oxford_train_data_paths = args.get("oxford_train_data_paths")
        oxford_train_gt_paths = args.get("oxford_train_gt_paths")
        oxford_data_paths = args.get("oxford_data_paths")
        oxford_gt_paths = args.get("oxford_gt_paths")

        # Preferred mode: rotate validation over full training sequences.
        if oxford_train_data_paths and oxford_train_gt_paths:
            rotating_train_datasets = build_oxford_datasets(oxford_train_data_paths,
                                                            oxford_train_gt_paths,
                                                            args,
                                                            preprocess)
            # Placeholders for signature compatibility; real loaders are built per epoch.
            train_loader = None
            val_loader = None

            oxford_test_data_paths = args.get("oxford_test_data_paths")
            oxford_test_gt_paths = args.get("oxford_test_gt_paths")
            if oxford_test_data_paths and oxford_test_gt_paths:
                test_datasets = build_oxford_datasets(oxford_test_data_paths,
                                                      oxford_test_gt_paths,
                                                      args,
                                                      preprocess)
                test_data = concat_if_needed(test_datasets)
                test_loader = torch.utils.data.DataLoader(test_data,
                                                          batch_size=1,
                                                          shuffle=False)
            elif args.get("oxford_test_data_path") and args.get("oxford_test_gt_path"):
                test_data = build_oxford_dataset(args["oxford_test_data_path"],
                                                args["oxford_test_gt_path"],
                                                args,
                                                preprocess)
                test_loader = torch.utils.data.DataLoader(test_data,
                                                          batch_size=1,
                                                          shuffle=False)
            else:
                print("No dedicated Oxford test path configured. Skipping final test evaluation.")

        # Backward-compatible mode: split each sequence chronologically.
        elif oxford_data_paths and oxford_gt_paths:
            datasets = build_oxford_datasets(oxford_data_paths,
                                             oxford_gt_paths,
                                             args,
                                             preprocess)
            train_datasets = []
            val_datasets = []
            for seq_dataset in datasets:
                seq_train, seq_val = chronological_split(seq_dataset, args["val_split"])
                train_datasets.append(seq_train)
                val_datasets.append(seq_val)

            train_data = concat_if_needed(train_datasets)
            val_data = concat_if_needed(val_datasets)

            train_loader = torch.utils.data.DataLoader(train_data,
                                                       batch_size=args["bsize"],
                                                       shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_data,
                                                     batch_size=1,
                                                     shuffle=False)
        else:
            dataset = build_oxford_dataset(args["oxford_data_path"],
                                           args["oxford_gt_path"],
                                           args,
                                           preprocess)
            train_data, val_data = chronological_split(dataset, args["val_split"])
            train_loader = torch.utils.data.DataLoader(train_data,
                                                       batch_size=args["bsize"],
                                                       shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_data,
                                                     batch_size=1,
                                                     shuffle=False)
    else:
        dataset = KITTI(window_size=args["window_size"], overlap=args["overlap"], transform=preprocess)
        train_data, val_data = chronological_split(dataset, args["val_split"])
        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=args["bsize"],
                                                   shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data,
                                                 batch_size=1,
                                                 shuffle=False)

    # Build and load model
    print("Building model...")
    model, args = build_model(args, model_params)

    # Loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = get_optimizer(model.parameters(), args)

    # Train network
    print(20*"--" +  " Training " + 20*"--")
    train(model,
          train_loader,
          val_loader,
          criterion,
          optimizer,
          TensorBoardWriter,
          args,
          rotating_train_datasets=rotating_train_datasets)

    # Evaluate the held-out test sequence once after training.
    if dataset_name == "oxford":
        evaluate_test_once(model, test_loader, criterion, TensorBoardWriter, args)
