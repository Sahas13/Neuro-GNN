import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam

from model import SeizureGNN
from eeg_graph_utils import eeg_to_graph

# Load dataset
df = pd.read_csv("../dataset/eeg_sample.csv")

model = SeizureGNN()
optimizer = Adam(model.parameters(), lr=0.001)

print("🔄 Training started...")

for epoch in range(50):
    total_loss = 0

    for _, row in df.iterrows():
        eeg = row[1:-1].values
        label = torch.tensor([int(row["label"])])

        graph = eeg_to_graph(eeg)
        output = model(graph)

        loss = F.cross_entropy(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/50 | Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "seizure_gnn.pth")
print("✅ Model saved as seizure_gnn.pth")
