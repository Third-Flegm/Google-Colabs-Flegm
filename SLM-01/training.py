import math
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import tiktoken

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Training on:", device)

# --- Upgraded Hyperparameters ---
FILE_NAME = "books_set_en.txt"
CONTEXT_LENGTH = 256
EMBED_DIM = 256
N_HEAD = 8
N_LAYERS = 6
FF_DIM = 1024  # Standard 4x EMBED_DIM ratio
DROPOUT = 0.1
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 3e-4
WARMUP_STEPS = 200  # Prevents early gradient instability

# 1. Load Sub-word Tokenizer
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

# 2. Load and tokenize the dataset
print(f"Reading and tokenizing {FILE_NAME}...")
with open(FILE_NAME, "r", encoding="utf-8") as f:
    text = f.read()

tokens = enc.encode(text)
data = torch.tensor(tokens, dtype=torch.long)
print(f"Total tokens: {len(data):,} | Fixed Vocab Size: {vocab_size:,}")

# 3. Create training chunks
stride = CONTEXT_LENGTH  # Non-overlapping chunks = 2x faster epochs, less overfitting
chunks = data.unfold(0, CONTEXT_LENGTH + 1, stride)
X = chunks[:, :-1]
Y = chunks[:, 1:]

dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
print(f"Training batches per epoch: {len(loader):,}")


# 4. Define Upgraded TinySLM
class TinySLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_len, n_head, n_layers, ff_dim, dropout):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(context_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_head,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(embed_dim),
            enable_nested_tensor=False,
        )
        self.fc_out = nn.Linear(embed_dim, vocab_size, bias=False)
        self.fc_out.weight = self.token_emb.weight  # Weight tying

        # Fix: Proper GPT-style weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device)
        x = self.token_emb(x) + self.pos_emb(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.fc_out(x)


# 5. Initialize Model, Optimizer, Scheduler & Scaler
model = TinySLM(
    vocab_size, EMBED_DIM, CONTEXT_LENGTH, N_HEAD, N_LAYERS, FF_DIM, DROPOUT
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
total_steps = len(loader) * EPOCHS


def lr_lambda(step):
    if step < WARMUP_STEPS:
        return float(step + 1) / float(max(1, WARMUP_STEPS))
    progress = float(step - WARMUP_STEPS) / float(max(1, total_steps - WARMUP_STEPS))
    return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
loss_fn = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

param_count = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {param_count:,}")

# 6. Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    running_loss = 0.0

    for step, (batch_x, batch_y) in enumerate(loader):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits = model(batch_x)
            loss = loss_fn(logits.view(-1, vocab_size), batch_y.view(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_val = loss.item()
        total_loss += loss_val
        running_loss += loss_val

        if (step + 1) % 250 == 0:
            avg_recent = running_loss / 250
            print(
                f"  Epoch {epoch + 1} | Step {step + 1}/{len(loader)} | "
                f"Avg Loss (250): {avg_recent:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}"
            )
            running_loss = 0.0

    avg_loss = total_loss / len(loader)
    print(f"==> Epoch {epoch + 1}/{EPOCHS} Complete | Avg Loss: {avg_loss:.4f}")

torch.save(
    {
        "model_state": model.state_dict(),
        "tokenizer": "gpt2",
        "context_length": CONTEXT_LENGTH,
        "embed_dim": EMBED_DIM,
        "n_head": N_HEAD,
        "n_layers": N_LAYERS,
        "ff_dim": FF_DIM,
    },
    "brain-en.pt",
)
print("Saved checkpoint to brain-en.pt")
