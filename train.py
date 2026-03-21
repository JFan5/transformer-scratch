
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


def train(model, dataloader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss = 0
    for batch in dataloader:
        #
        input_seq, target_seq = batch
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)

        optimizer.zero_grad()

        # Shifted targets: feed ground truth as input for teacher forcing / parallel training
        # To predict token i, the model uses all tokens before position i
        tgt_input = target_seq[:, :-1]   # [<start>, tok1, tok2, tok3]
        tgt_output = target_seq[:, 1:]   # [tok1, tok2, tok3, <end>]

        # Create masks (must be before model forward pass)
        src_mask = create_padding_mask(input_seq)
        # tgt_mask combines padding mask and causal mask, shape: (batch, 1, seq_len, seq_len)
        tgt_mask = create_padding_mask(tgt_input) & create_causal_mask(tgt_input)

        # Forward pass

        output = model(input_seq, tgt_input, src_mask, tgt_mask)

        # Compute loss
        loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
        total_loss += loss.item()

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        # Update learning rate after each batch
        if scheduler is not None:
            scheduler.step()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

if __name__ == "__main__":
    config = {
        "d_model": 768,
        "num_heads": 8,
        "d_ff": 2048,
        "num_layers": 6,
        "lr": 0.001,
        "epochs": 30,
        "patience": 5,
    }

    wandb.init(project="transformer", config=config)

    print("Loading data...")
    dataloader, en_vocab_size, de_vocab_size=load_data()
    model=Transformer(d_model=config["d_model"], num_heads=config["num_heads"], d_ff=config["d_ff"], num_layers=config["num_layers"], src_vocab_size=en_vocab_size, tgt_vocab_size=de_vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = TransformerLRScheduler(optimizer, config["d_model"], warmup_steps=4000)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print("Start training...")
    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(config["epochs"]):
        avg_loss = train(model, dataloader, optimizer, criterion, device, scheduler)
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"epoch": epoch, "train_loss": avg_loss, "lr": current_lr})
        print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
            print("  -> Best model saved.")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{config['patience']}")
            if patience_counter >= config["patience"]:
                print("Early stopping triggered.")
                break

    wandb.finish()

