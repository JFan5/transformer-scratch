
import os
from model import Transformer
import model
import torch
from torch.utils.data import DataLoader
from data import load_data
from  masks import create_padding_mask, create_causal_mask
import wandb
# Training function


class TransformerLRScheduler:
    """Warmup + decay learning rate scheduler from the original paper"""
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


def train(model, dataloader, optimizer, criterion, device, scheduler=None, accumulation_steps=1):
    model.train()
    total_loss = 0
    scaler = torch.amp.GradScaler('cuda')
    optimizer.zero_grad()

    for i, batch in enumerate(dataloader):
        input_seq, target_seq = batch
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)

        tgt_input = target_seq[:, :-1]
        tgt_output = target_seq[:, 1:]

        src_mask = create_padding_mask(input_seq)
        tgt_mask = create_padding_mask(tgt_input) & create_causal_mask(tgt_input)

        with torch.amp.autocast('cuda'):
            output = model(input_seq, tgt_input, src_mask, tgt_mask)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * accumulation_steps

    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    for batch in dataloader:
        input_seq, target_seq = batch
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)

        tgt_input = target_seq[:, :-1]
        tgt_output = target_seq[:, 1:]

        src_mask = create_padding_mask(input_seq)
        tgt_mask = create_padding_mask(tgt_input) & create_causal_mask(tgt_input)

        with torch.amp.autocast('cuda'):
            output = model(input_seq, tgt_input, src_mask, tgt_mask)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

if __name__ == "__main__":
    config = {
        "d_model": 256,
        "num_heads": 8,
        "d_ff": 1024,
        "num_layers": 3,
        "lr": 0.001,
        "epochs": 100,
        "patience": 10,
        "batch_size": 128,
        "accumulation_steps": 1,
        "max_len": 256,
        "max_samples": 200000,
    }

    wandb.init(project="transformer", config=config)

    print("Loading data...")
    train_dataloader, en_vocab_size, de_vocab_size = load_data(batch_size=config["batch_size"], max_len=config["max_len"], max_samples=config["max_samples"], split="train")
    val_dataloader, _, _ = load_data(batch_size=config["batch_size"], max_len=config["max_len"], max_samples=config["max_samples"] // 10, split="validation")

    model=Transformer(d_model=config["d_model"], num_heads=config["num_heads"], d_ff=config["d_ff"], num_layers=config["num_layers"], src_vocab_size=en_vocab_size, tgt_vocab_size=de_vocab_size)

    # Resume from checkpoint if exists
    checkpoint_path = "best_model.pt"
    if os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = TransformerLRScheduler(optimizer, config["d_model"], warmup_steps=4000)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print("Start training...")
    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(config["epochs"]):
        train_loss = train(model, train_dataloader, optimizer, criterion, device, scheduler, config["accumulation_steps"])
        val_loss = evaluate(model, val_dataloader, criterion, device)
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": current_lr})
        print(f"Epoch {epoch+1}/{config['epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print("  -> Best model saved.")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{config['patience']}")
            if patience_counter >= config["patience"]:
                print("Early stopping triggered.")
                break

    wandb.finish()

