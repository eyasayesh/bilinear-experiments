# config.py
BATCH_SIZE = 2048
LR = 1e-3
WEIGHT_DECAY = 1e-3
INPUT_NOISE = 0.3
EPOCHS = 30
DATASET = "mnist"
SAVE_PATH = "checkpoints/bilinear_mnist.pth"
WANDB_PROJECT = "bilinear_classifier_mnist"