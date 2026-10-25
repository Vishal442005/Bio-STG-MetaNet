import sys, os
sys.path.insert(0, os.path.abspath("."))
import torch
from src.data.uci_loader import load_uci_begleiter_dataset
from src.models import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

X, y, subs = load_uci_begleiter_dataset()
x_batch = torch.tensor(X[:32]).to(device)
y_batch = torch.tensor(y[:32]).to(device)

print("Instantiating conformer...")
model = get_model("conformer", num_channels=61, time_steps=256).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
crit = torch.nn.CrossEntropyLoss()

print("Running 5 epochs of training on batch...")
for ep in range(5):
    opt.zero_grad()
    out = model(x_batch)
    loss = crit(out, y_batch)
    loss.backward()
    opt.step()
    print(f"Epoch {ep+1}: Loss = {loss.item():.4f}")

print("Conformer batch test SUCCESS!")
