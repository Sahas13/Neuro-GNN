import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class SeizureGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 32)
        self.conv2 = GCNConv(32, 64)
        self.fc = torch.nn.Linear(64, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        batch = torch.zeros(x.size(0), dtype=torch.long)
        x = global_mean_pool(x, batch)

        return self.fc(x)
