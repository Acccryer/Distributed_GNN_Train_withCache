import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected
import numpy as np


def load_cora_data(data_path="/home/sinon/dataset/Planetoid"):
    """
    使用 PyTorch Geometric 加载 Cora 数据集。

    参数:
        data_path (str): 数据存储目录，默认为 "/home/sinon/dataset/Planetoid/Cora"

    返回:
        adj (torch.FloatTensor): 归一化的邻接矩阵
        features (torch.FloatTensor): 节点特征矩阵
        labels (torch.LongTensor): 节点标签
        train_mask (torch.BoolTensor): 训练集掩码
        val_mask (torch.BoolTensor): 验证集掩码
        test_mask (torch.BoolTensor): 测试集掩码
    """
    dataset = Planetoid(root=data_path, name="Cora")
    data = dataset[0]  # Planetoid 返回一个列表，Cora 只有一个图

    features = data.x  # 特征矩阵 [num_nodes, num_features]
    labels = data.y  # 标签 [num_nodes]
    edge_index = data.edge_index  # 边索引 [2, num_edges]

    edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)

    rows, cols = edge_index[0], edge_index[1]
    adj = torch.zeros((data.num_nodes, data.num_nodes), dtype=torch.float32)
    adj[rows, cols] = 1.0
    adj = adj + torch.eye(data.num_nodes)  # 添加自环
    adj = adj / adj.sum(dim=1, keepdim=True)  # 归一化 (D^-1 A)

    #planetoid掩码
    train_mask = data.train_mask  # 默认 140 个训练节点
    val_mask = data.val_mask  # 默认 500 个验证节点
    test_mask = data.test_mask  # 默认 1000 个测试节点


    print(f"Loaded Cora data: {data.num_nodes} nodes, {data.num_edges} edges, "
          f"{data.num_features} features per node, {dataset.num_classes} classes")
    print(f"Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, Test: {test_mask.sum().item()}")

    return adj, features, labels, train_mask, val_mask, test_mask



if __name__ == "__main__":
    adj, features, labels, train_mask, val_mask, test_mask = load_cora_data()