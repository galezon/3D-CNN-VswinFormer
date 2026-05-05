import logging
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import sys
from pathlib import Path
from config import BEST_MODEL_PATH, TRAIN_LOSS_PATH, VAL_LOSS_PATH, TRAIN_ACC_PATH, VAL_ACC_PATH, CHECKPOINT_PATH
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

logger = logging.getLogger(__name__)

# TODO: automatic saves for different versions (also where it was ran, setup etc), run it already for ctf data as well

def train(
    model,
    device,
    train_loader,
    val_loader,
    train_num,
    val_num,
    output_dir,
    epochs=550,
    resume_from=None
):

    # Create output directory structure
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct paths from base dir + config relative paths
    best_model_path = output_dir / BEST_MODEL_PATH
    checkpoint_path = output_dir / CHECKPOINT_PATH
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    train_loss_path = output_dir / TRAIN_LOSS_PATH
    val_loss_path = output_dir / VAL_LOSS_PATH
    train_acc_path = output_dir / TRAIN_ACC_PATH
    val_acc_path = output_dir / VAL_ACC_PATH
    train_acc_path.parent.mkdir(parents=True, exist_ok=True)
    
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.000005, weight_decay=1e-1)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-7, last_epoch=-1
    )
    best_acc = 0.0
    best_epoch = 0
    train_loss = []
    val_loss = []
    tra_acc = []
    val_acc = []

    train_steps = len(train_loader)
    val_steps = len(val_loader)
    start_epoch = 0
    
    # load from checkpoint if exists
    if resume_from is not None:
        ckpt =  load_checkpoint(
            resume_from, device, model, optimizer, scheduler,
            train_loss_path, val_loss_path, train_acc_path, val_acc_path
            )
        model = ckpt["model"]
        optimizer = ckpt["optimizer"]
        scheduler = ckpt["scheduler"]
        start_epoch = ckpt["start_epoch"]
        best_acc = ckpt["best_acc"]
        best_epoch = ckpt["best_epoch"]
        train_loss = ckpt["train_loss"]
        val_loss = ckpt["val_loss"]
        tra_acc = ckpt["tra_acc"]
        val_acc = ckpt["val_acc"]


    for epoch in range(start_epoch, epochs):
        # train
        model.train()
        
        running_loss = 0.0
        train_acc = 0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            # Modify input
            images = images.permute(0, 1, 4, 3, 2)
            optimizer.zero_grad()
            pred = model(images.to(device))
            loss = loss_function(pred, labels.to(device))
            loss.backward()
            optimizer.step()
            pred_train = torch.argmax(pred, dim=1)
            train_acc += torch.eq(pred_train, labels.to(device)).sum().item()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, running_loss / (step + 1))
        
        scheduler.step()
        logger.info(f"Epoch {epoch + 1} LR: {optimizer.state_dict()['param_groups'][0]['lr']}")
        train_loss.append(running_loss / len(train_loader))

        # validate
        model.eval()
        acc_loss = 0.0
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_step, val_data in enumerate(val_bar):
                val_images, val_labels = val_data
                val_images = val_images.permute(0, 1, 4, 3, 2)
                pred = model(val_images.to(device))
                loss = loss_function(pred, val_labels.to(device))
                acc_loss += loss.item()
                predict_y = torch.argmax(pred, dim=1)
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        
        train_accurate = train_acc / train_num
        val_accurate = acc / val_num
        tra_acc.append(train_accurate)
        val_acc.append(val_accurate)
        val_loss.append(acc_loss / len(val_loader))
        logger.info(f"[epoch {epoch + 1}] train_loss: {running_loss / train_steps:.3f}  train_accuracy: {train_accurate:.3f}  val_loss: {acc_loss / val_steps:.3f}  val_accuracy: {val_accurate:.3f}")

        if val_accurate > best_acc:
            best_acc = val_accurate
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)

        # save current step
        if (epoch+1) % 10 == 0:
            save_checkpoint(epoch, model, optimizer, scheduler, best_acc, best_epoch, checkpoint_path)
            logger.info(f"Epoch {epoch + 1} checkpoint saved")

            # Save metrics
            append_value(train_loss_path, train_loss[-1])
            append_value(val_loss_path, val_loss[-1])
            append_value(train_acc_path, tra_acc[-1])
            append_value(val_acc_path, val_acc[-1])
            logger.info(f"Epoch {epoch+1} logged metrics")

    logger.info('Finished Training')


def load_checkpoint(
        resume_from,
        device, 
        model, 
        optimizer, 
        scheduler,
        train_loss_path,
        val_loss_path,
        train_acc_path,
        val_acc_path
        ):
    checkpoint = torch.load(resume_from, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_acc = checkpoint["best_acc"]
    best_epoch = checkpoint["best_epoch"]

    train_loss = load_values(train_loss_path)
    val_loss = load_values(val_loss_path)
    tra_acc = load_values(train_acc_path)
    val_acc = load_values(val_acc_path)

    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "start_epoch": start_epoch,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "tra_acc": tra_acc,
        "val_acc": val_acc,
    }
    
def save_checkpoint(epoch, model, optimizer, scheduler, best_acc, best_epoch, checkpoint_path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_acc": best_acc,
        "best_epoch": best_epoch,
    }, checkpoint_path)

def append_value(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(f"{value}\n")

def load_values(path):
    if not path.exists():
        return []
    with open(path, "r") as f:
        return [float(line.strip()) for line in f if line.strip()]

def get_plots(output_dir, title="Training Results"):
    """
    Load and plot training history from saved arrays.
    
    Args:
        output_dir: Output directory where results were saved
        title: Title for the plot
    
    Returns:
        fig: Matplotlib figure object
    """
    output_dir = Path(output_dir)
    
    train_loss = load_values(output_dir / TRAIN_LOSS_PATH)
    val_loss = load_values(output_dir / VAL_LOSS_PATH)
    tra_acc = load_values(output_dir / TRAIN_ACC_PATH)
    val_acc = load_values(output_dir / VAL_ACC_PATH)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(len(train_loss))
    
    ax1.plot(epochs, train_loss, 'g', label='Training Loss')
    ax1.plot(epochs, val_loss, 'b', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid()
    
    ax2.plot(epochs, tra_acc, label='Training Accuracy')
    ax2.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid()
    
    fig.suptitle(title)
    
    return fig