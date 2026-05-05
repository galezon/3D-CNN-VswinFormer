from pathlib import Path
import os

RANDOM_STATE = 0

# (hyper)-parameters
BATCH_SIZE = 2  # previously 2  # default value
NO_WORKERS = min([os.cpu_count(), BATCH_SIZE if BATCH_SIZE > 1 else 0, 8])  # number of workers  # default value

# classes
class_indict = {'0':'AD',
                 '1':'CN'}

# Data paths
DATASET_ROOT = Path("to/be/adjusted")  # is it even relevant after re-writing the Path read function

# Pre-training weights path
PRETRAIN_WEIGHTS_PATH = Path("/home/VIB.LOCAL/lunkyadikurniawan.sucipto/projects/3D-CNN-VswinFormer/checkpoints/pretrained_weight/swin_tiny_patch244_window877_kinetics400_1k.pth")

# these probably should be relative to some final relative path shouldn't it
# Output paths
BEST_MODEL_PATH = "checkpoints/best_model.pth"  # Model with best validation accuracy
CHECKPOINT_PATH = "checkpoints/latest.pt"
TRAIN_LOSS_PATH = "files/train_loss.txt"
VAL_LOSS_PATH = "files/val_loss.txt"
TRAIN_ACC_PATH = "files/train_acc.txt"
VAL_ACC_PATH = "files/val_acc.txt"
