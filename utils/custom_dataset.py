import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, nodes, edge_attributes, actions, rewards, log_probs):
        self.nodes = nodes
        self.edge_attributes = edge_attributes
        self.actions = actions
        self.rewards = rewards
        self.log_probs = log_probs

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):
        node = self.nodes[idx]
        edge_attr = self.edge_attributes[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        log_prob = self.log_probs[idx]

        return node, edge_attr, action, reward, log_prob