import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.bilinear_adder import BilinearAdder
from datasets import get_mod_add_dataloaders
from utils.train_utils import train_one_epoch, evaluate
from configs import bilinear_adder_config as config
import os
import wandb

def main():

    wandb.init(project=config.WANDB_PROJECT, 
               config={
                     "learning_rate": config.LR,
                     "weight_decay": config.WEIGHT_DECAY,
                     "batch_size": config.BATCH_SIZE,
                     "epochs": config.EPOCHS,
                     "architecture": "BilinearAdder",
                     "dataset": config.DATASET,
                     "d_hidden": config.d_hidden,
                     "P": config.P,
                     "checkpoint_path": config.SAVE_PATH
                })
    # Load data
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_mod_add_dataloaders(modulus = config.P,
                                                        batch_size=config.BATCH_SIZE,
                                                        device=device)

    # Initialize model, loss function, optimizer
    model = BilinearAdder(d_input=2*config.P,
                          d_hidden=config.d_hidden,
                          P=config.P).to(device)
    
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
