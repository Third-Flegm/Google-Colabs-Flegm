import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

device = "cuda" if torch.cuda.is_available() else "cpu"

class TinySLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_len, n_head, n_layers, ff_dim, dropout=0.0):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(context_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_head,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",        # Matches training
            batch_first=True,
            norm_first=True           # Matches training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(embed_dim),  # Matches training
            enable_nested_tensor=False
        )
        self.fc_out = nn.Linear(embed_dim, vocab_size, bias=False)  # Matches training
        self.fc_out.weight = self.token_emb.weight

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device)
        x = self.token_emb(x) + self.pos_emb(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.fc_out(x)

# 1. Load checkpoint and tokenizer
checkpoint = torch.load("brain-en.pt", map_location=device, weights_only=True)
enc = tiktoken.get_encoding(checkpoint["tokenizer"])
context_length = checkpoint["context_length"]

model = TinySLM(
    enc.n_vocab,
    checkpoint["embed_dim"],
    context_length,
    checkpoint["n_head"],
    checkpoint["n_layers"],
    checkpoint["ff_dim"],
    dropout=0.0
).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# 2. Start from an English prompt (matching books_set_en.txt)
prompt = input("Enter starting prompt (e.g., 'Once upon a time' or 'The meaning of life'): ")
if not prompt.strip():
    prompt = "Once upon a time"

token_ids = enc.encode(prompt)

TEMPERATURE = 0.75
TOP_K = 40
REPETITION_PENALTY = 1.15

with torch.no_grad():
    for _ in range(300):
        context = torch.tensor([token_ids[-context_length:]], dtype=torch.long, device=device)
        logits = model(context)[0, -1, :]

        # Repetition penalty on the last 30 tokens
        for prev_id in set(token_ids[-30:]):
            if logits[prev_id] > 0:
                logits[prev_id] /= REPETITION_PENALTY
            else:
                logits[prev_id] *= REPETITION_PENALTY

        logits = logits / TEMPERATURE

        # Top-K sampling
        top_vals, top_idx = torch.topk(logits, TOP_K)
        probs = F.softmax(top_vals, dim=-1)
        next_token = top_idx[torch.multinomial(probs, num_samples=1)].item()

        token_ids.append(next_token)

generated_text = enc.decode(token_ids)
print("\n--- GENERATED TEXT ---\n")
print(generated_text)

with open("output.txt", "w", encoding="utf-8") as f:
    f.write(generated_text)
