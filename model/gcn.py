import torch
import torch.nn as nn

# GCN 模型定义
class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(GCN, self).__init__()
        self.gc1 = nn.Linear(in_features, hidden_features)
        self.gc2 = nn.Linear(hidden_features, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # dropout

    def forward(self, adj, features):
        h = self.relu(torch.mm(adj, self.gc1(features)))
        h = self.dropout(h)
        output = torch.mm(adj, self.gc2(h))
        return output