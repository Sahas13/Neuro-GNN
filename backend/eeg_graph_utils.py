import torch
from torch_geometric.data import Data

def eeg_to_graph(eeg_values):
    """
    Converts EEG channel values into a fully-connected graph
    """
    x = torch.tensor(eeg_values, dtype=torch.float).view(-1, 1)

    num_nodes = x.shape[0]
    edge_index = []

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)
