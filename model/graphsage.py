import torch
import torch.nn as nn
import numpy as np


class GraphSAGE(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, aggr='mean', dropout=0.5, use_sampling=False,
                 num_samples=10):
        """
        GraphSAGE 模型定义（重新设计）

        参数:
            in_features (int): 输入特征维度
            hidden_features (int): 隐藏层特征维度
            out_features (int): 输出层特征维度（类别数）
            aggr (str): 邻居聚合方式，可选 'mean', 'sum'，默认 'mean'
            dropout (float): Dropout 比例，默认 0.5
            use_sampling (bool): 是否启用邻居采样，默认 False（全图聚合）
            num_samples (int): 每节点采样的邻居数，默认 10（仅在 use_sampling=True 时生效）
        """
        super(GraphSAGE, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.use_sampling = use_sampling
        self.num_samples = num_samples

        # 第一层：邻居聚合后变换
        self.conv1 = nn.Linear(in_features * 2, hidden_features)  # 拼接自身和邻居特征
        # 第二层：邻居聚合后变换
        self.conv2 = nn.Linear(hidden_features * 2, out_features)  # 输出层

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.aggr = aggr.lower()

    def aggregate(self, adj, features, nids=None):
        """
        聚合邻居特征

        参数:
            adj (torch.Tensor): 邻接矩阵 [N, N]（假设已归一化）
            features (torch.Tensor): 节点特征矩阵 [N, in_features]
            nids (torch.Tensor, optional): 需要聚合的节点 ID，默认 None（全图模式）

        返回:
            torch.Tensor: 聚合后的邻居特征 [N 或 len(nids), in_features]
        """
        if self.use_sampling and nids is not None:
            # 采样模式
            sampled_nids = self.sample_neighbors(adj, nids, self.num_samples, features.device)
            sampled_features = features[sampled_nids]  # [len(nids) * num_samples, in_features]
            # 重塑为 [len(nids), num_samples, in_features]
            sampled_features = sampled_features.view(len(nids), self.num_samples, features.shape[1])
            # 聚合
            if self.aggr == 'mean':
                return sampled_features.mean(dim=1)  # [len(nids), in_features]
            elif self.aggr == 'sum':
                return sampled_features.sum(dim=1)  # [len(nids), in_features]
            else:
                raise ValueError("Aggregation method must be 'mean' or 'sum'")
        else:
            # 全图模式
            neigh_features = torch.mm(adj, features)  # [N, in_features]
            return neigh_features


    def sample_neighbors(self, adj, nids, num_samples, device):
        """
        采样邻居

        参数:
            adj (torch.Tensor): 邻接矩阵 [N, N]
            nids (torch.Tensor): 需要采样的节点 ID
            num_samples (int): 每节点采样的邻居数
            device (torch.device): 张量的设备

        返回:
            torch.Tensor: 采样的邻居 ID
        """
        adj = adj.cpu().numpy()
        sampled_nids = []
        for nid in nids.cpu().numpy():
            neighbors = np.where(adj[nid] > 0)[0]
            if len(neighbors) > num_samples:
                sampled = np.random.choice(neighbors, num_samples, replace=False)
            else:
                sampled = neighbors
                sampled = np.pad(sampled, (0, num_samples - len(neighbors)), mode='edge')
            sampled_nids.append(sampled)
        return torch.tensor(np.array(sampled_nids), dtype=torch.long, device=device).flatten()

    def forward(self, adj, features):
        """
        前向传播

        参数:
            adj (torch.Tensor): 归一化的邻接矩阵 [N, N]
            features (torch.Tensor): 节点特征矩阵 [N, in_features]

        返回:
            torch.Tensor: 输出预测 [N, out_features]
        """
        # 指定所有节点用于采样（如果启用）
        nids = torch.arange(features.shape[0], device=features.device) if self.use_sampling else None

        # 第一层
        neigh_h = self.aggregate(adj, features, nids)  # [N, in_features]
        h = torch.cat([features, neigh_h], dim=1)  # [N, in_features * 2]
        h = self.relu(self.conv1(h))  # [N, hidden_features]
        h = self.dropout(h)

        # 第二层
        neigh_h = self.aggregate(adj, h, nids)  # [N, hidden_features]
        h = torch.cat([h, neigh_h], dim=1)  # [N, hidden_features * 2]
        h = self.conv2(h)  # [N, out_features]

        return h

