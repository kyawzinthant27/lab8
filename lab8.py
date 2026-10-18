import copy
import math
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

EPOCHS = 40
BATCH_SIZE = 64
SEQ_LENGTH = 100
torch.manual_seed(1234)
torch.set_num_threads(4)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# Task 1
path = Path(__file__).with_name("wonderful_visit.txt")
text = path.read_text(encoding="utf-8-sig")
start = text.index("THE NIGHT OF THE STRANGE BIRD.\n\nI.")
end = text.index("\nTHE END.", start) + len("\nTHE END.")
text = text[start:end].strip()
a, b = int(len(text) * 0.8), int(len(text) * 0.9)
train, val, test = text[:a], text[a:b], text[b:]
chars = ["<UNK>"] + sorted(set(train))
char_to_id = {c: i for i, c in enumerate(chars)}
print("\n[Task 1] The Wonderful Visit by H. G. Wells")
print("Train / validation / test characters:", len(train), len(val), len(test))
print("Vocabulary size:", len(chars))

def make_loader(text, shuffle=False):
    ids = torch.tensor([char_to_id.get(c, 0) for c in text])
    n = (len(ids) - 1) // SEQ_LENGTH * SEQ_LENGTH
    x = ids[:n].reshape(-1, SEQ_LENGTH)
    y = ids[1:n + 1].reshape(-1, SEQ_LENGTH)
    return DataLoader(TensorDataset(x, y), BATCH_SIZE, shuffle=shuffle)

train_loader = make_loader(train, shuffle=True)
val_loader = make_loader(val)
test_loader = make_loader(test)

# Task 2
class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(chars), 32)
        self.gru = nn.GRU(32, 128, batch_first=True)
        self.output = nn.Linear(128, len(chars))

    def forward(self, x, hidden=None):
        x, hidden = self.gru(self.embedding(x), hidden)
        return self.output(x), hidden

model = LanguageModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
loss_fn = nn.CrossEntropyLoss()
print("\n[Task 2]", model)
print(f"Seed=1234, epochs={EPOCHS}, batch={BATCH_SIZE}, sequence={SEQ_LENGTH}")
print("Adam lr=0.002, gradient clip=1.0")

@torch.no_grad()
def evaluate(loader):
    model.eval()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = loss_fn(logits.reshape(-1, len(chars)), y.reshape(-1))
        total += loss.item() * y.numel()
        count += y.numel()
    return total / count

# Task 3
print("\n[Task 3] Training")
best_loss = float("inf")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total, count = 0.0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = loss_fn(logits.reshape(-1, len(chars)), y.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * y.numel()
        count += y.numel()
    val_loss = evaluate(val_loader)
    print(f"Epoch {epoch:02d}: train={total / count:.4f}, validation={val_loss:.4f}")
    if val_loss < best_loss:
        best_loss, best_epoch = val_loss, epoch
        best_weights = copy.deepcopy(model.state_dict())

# Task 4
model.load_state_dict(best_weights)
test_loss = evaluate(test_loader)
print("\n[Task 4] Best epoch:", best_epoch)
print(f"Test loss: {test_loss:.4f} nats per character")
print(f"Character-level perplexity: {math.exp(test_loss):.4f}")

# Task 5
print("\n[Task 5] Generated text (temperature=0.8)")
torch.manual_seed(1235)
with torch.no_grad():
    for prompt in ["The Vicar ", "The Angel ", "The village "]:
        x = torch.tensor([[char_to_id[c] for c in prompt]], device=device)
        hidden = None
        generated = prompt
        for _ in range(400):
            logits, hidden = model(x, hidden)
            scores = logits[:, -1, :] / 0.8
            scores[:, 0] = -float("inf")
            x = torch.multinomial(torch.softmax(scores, dim=-1), 1)
            generated += chars[x.item()]
        print(f"\nPrompt: {prompt!r}\n{generated}")
print("\nFinished.")
