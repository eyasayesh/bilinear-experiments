import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.vanilla_classifier import VanillaClassifier
from datasets import get_mnist_dataloaders
from utils.train_utils import train_one_epoch, evaluate
from configs import vanilla_classifier_config as config
import os
import wandb

def main():

    wandb.init(project=config.WANDB_PROJECT, 
               config={
                     "learning_rate": config.LR,
                     "weight_decay": config.WEIGHT_DECAY,
                     "batch_size": config.BATCH_SIZE,
                     "epochs": config.EPOCHS,
                     "architecture": "VanillaClassifier",
                     "dataset": config.DATASET
                })
    # Load data
    train_loader, test_loader = get_mnist_dataloaders(batch_size=config.BATCH_SIZE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, loss function, optimizer
    model = VanillaClassifier(n_embed=512,n_h=512,n_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)


    # Training loop
    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion,device)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "learning_rate": scheduler.get_last_lr()[0] if scheduler else config.LR
        })

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy*100:.2f}%")
        scheduler.step()

    # Save the trained model
    os.makedirs(os.path.dirname(config.SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), config.SAVE_PATH)
    print(f"Model saved to {config.SAVE_PATH}")

if __name__ == "__main__":
    main()
