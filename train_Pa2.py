import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import os
import time
import json
from load_data.load_pubmed_data import load_pubmed_data
from load_data.load_cora_data import load_cora_data
from load_data.load_citeseer_data import load_citeseer_data

from model.gcn import GCN
from model.graphsage import GraphSAGE

from eval.evaluate import evaluate

# 初始化分布式环境
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# 广播tensor到所有gpu
def broadcast_tensor(tensor, src, world_size, device):
    if src == 0:
        if tensor is None:
            raise ValueError("Tensor cannot be None on rank 0")
        tensor = tensor.to(device)
        shape = torch.tensor(tensor.shape, dtype=torch.long, device=device)
        dist.broadcast(shape, src=src)
        dist.broadcast(tensor, src=src)
    else:
        shape = torch.zeros(2, dtype=torch.long, device=device)
        dist.broadcast(shape, src=src)
        tensor = torch.zeros(shape.tolist(), dtype=torch.float32, device=device)
        dist.broadcast(tensor, src=src)
    return tensor


class GraphCacheServer:
    def __init__(self, features, adj, gpuid, max_cache_size_mb=10):
        """
        初始化参数：
        features: 所有节点特征
        adj: 邻接矩阵
        gpuid: GPU设备ID
        max_cache_size_mb: 最大缓存大小（MB），默认为10MB
        """
        self.features = features
        self.adj = adj
        self.gpuid = gpuid
        self.device = torch.device(f'cuda:{gpuid}')
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024  # 转换为字节

        # 计算每个特征的字节大小
        self.feature_size = features.element_size() * features.shape[1]
        self.max_capacity = self.max_cache_size_bytes // self.feature_size

        # 性能统计
        self.hit_count = 0
        self.total_count = 0
        self.fetch_time_total = 0
        self.fetch_calls = 0

        # 初始化缓存
        self._initialize_cache()

    def _initialize_cache(self):
        """初始化固定大小的缓存，按出度排序"""
        start_time = time.time()
        torch.cuda.set_device(self.device)

        # 确保容量不超过节点总数
        self.capacity = min(self.max_capacity, self.features.shape[0])

        # 初始化缓存结构
        self.cached_nids = torch.zeros(self.capacity, dtype=torch.long, device=self.device)
        self.cached_features = torch.zeros(self.capacity, self.features.shape[1],
                                           dtype=self.features.dtype, device=self.device)
        self.timestamp = torch.zeros(self.capacity, dtype=torch.long, device=self.device)
        self.current_time = 0

        # 计算每个节点及其邻居的度数和
        adj_raw = (self.adj > 0).float()
        node_degrees = adj_raw.sum(dim=1)  # 计算每个节点的度数（即出度）
        neighbor_degrees = adj_raw @ node_degrees  # 计算邻居的度数和（通过邻接矩阵与节点度数相乘）

        # 计算每个节点的出度和邻居的出度之和
        total_degrees = 0.33*node_degrees + 0.67*neighbor_degrees
        # 根据度数和进行排序，选出度数和最高的节点
        top_indices = np.argpartition(total_degrees.cpu().numpy(), -self.capacity)[-self.capacity:]

        # 填充缓存
        self.cached_nids[:] = torch.tensor(top_indices, device=self.device)
        self.cached_features[:] = self.features[top_indices].to(self.device)
        self.timestamp[:] = torch.arange(self.capacity, device=self.device)

        # 创建查找表
        self.cache_lookup = torch.full((self.features.shape[0],), -1,
                                       device=self.device, dtype=torch.long)
        self.cache_lookup[self.cached_nids] = torch.arange(self.capacity, device=self.device)

        self.cache_init_time = time.time() - start_time
        actual_size_mb = (self.feature_size * self.capacity) / (1024 * 1024)
        print(f"GPU {self.gpuid}: Initialized cache with {self.capacity} nodes, "
              f"Size: {actual_size_mb:.2f}MB, Init time: {self.cache_init_time:.4f}s")

    def fetch_features(self, nids):
        """获取节点特征，不更新缓存"""
        start_time = time.time()
        nids = nids.to(self.device)

        # 检查缓存命中
        cache_indices = self.cache_lookup[nids]
        mask = (cache_indices >= 0)

        # 更新统计
        self.hit_count += mask.sum().item()
        self.total_count += nids.size(0)

        # 初始化输出特征张量
        features = torch.zeros((nids.size(0), self.features.shape[1]),
                               device=self.device)

        # 处理缓存命中
        if mask.any():
            hit_indices = cache_indices[mask]
            features[mask] = self.cached_features[hit_indices]

        # 处理缓存未命中
        if (~mask).any():
            miss_nids = nids[~mask]
            miss_features = self.features[miss_nids.cpu()].to(self.device)
            features[~mask] = miss_features

        fetch_time = time.time() - start_time
        self.fetch_time_total += fetch_time
        self.fetch_calls += 1
        return features

    def get_cache_stats(self):
        """返回缓存统计信息"""
        hit_rate = self.hit_count / self.total_count if self.total_count > 0 else 0
        avg_fetch_time = self.fetch_time_total / self.fetch_calls if self.fetch_calls > 0 else 0
        cache_size_mb = (self.feature_size * self.capacity) / (1024 * 1024)
        return {
            "cache_init_time": self.cache_init_time,
            "hit_rate": hit_rate,
            "avg_fetch_time": avg_fetch_time,
            "total_fetch_time": self.fetch_time_total,
            "fetch_calls": self.fetch_calls,
            "cache_size_mb": cache_size_mb,
            "capacity": self.capacity,
            "hit_count": self.hit_count,
            "total_count": self.total_count
        }


# 分布式训练函数
def distributed_train(rank, world_size, adj, features, labels, train_mask, val_mask, test_mask, num_epochs=200):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    if rank == 0:
        adj = adj.to(device)
        features = features.to(device)
        labels = labels.to(device)
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)
    else:
        adj, features, labels, train_mask, val_mask, test_mask = None, None, None, None, None, None

    adj = broadcast_tensor(adj, 0, world_size, device)
    features = broadcast_tensor(features, 0, world_size, device)
    labels = broadcast_tensor(labels, 0, world_size, device)
    train_mask = broadcast_tensor(train_mask, 0, world_size, device)
    val_mask = broadcast_tensor(val_mask, 0, world_size, device)
    test_mask = broadcast_tensor(test_mask, 0, world_size, device)

    cache_server = GraphCacheServer(features, adj, rank, max_cache_size_mb=10) # 最大缓存大小，这里设置为10MB

    num_classes = labels.max().item() + 1
    model = GCN(in_features=features.shape[1], hidden_features=16, out_features=num_classes).to(device)  # 隐藏层大小16
    model = DDP(model, device_ids=[rank])
    dist.barrier()

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # 学习率0.01
    criterion = nn.CrossEntropyLoss()

    total_train_time = 0

    test_acc = 0
    cache_stats = 0
    results = {
        "train_loss": [],  # 在epoch循环中填充
        "train_acc": [],  # 在epoch循环中填充
        "val_acc": [],  # 在epoch循环中填充
        "test_acc": test_acc,
        "cache_stats": cache_stats
    }

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        start_time = time.time()
        nids = torch.arange(adj.shape[0], device=device)
        cached_features = cache_server.fetch_features(nids)
        output = model(adj, cached_features)
        loss = criterion(output[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()
        train_time = time.time() - start_time
        total_train_time += train_time

        if (epoch + 1) % 10 == 0:
            train_acc = evaluate(model, adj, features, labels, train_mask, cache_server, device)
            val_acc = evaluate(model, adj, features, labels, val_mask, cache_server, device)
            if rank == 0:
                results["train_loss"].append(loss.item())
                results["train_acc"].append(train_acc)
                results["val_acc"].append(val_acc)

                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, "
                      f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
                      f"Time: {train_time:.4f}s")

    if rank == 0:
        test_acc = evaluate(model, adj, features, labels, test_mask, cache_server, device)
        cache_stats = cache_server.get_cache_stats()

        results["test_acc"] = test_acc
        results["cache_stats"] = cache_stats

        print(f"Final Test Accuracy: {test_acc:.4f}")
        print(f"Cache Stats - Init Time: {cache_stats['cache_init_time']:.4f}s, "
              f"Hit Rate: {cache_stats['hit_rate']:.4f}, "
              f"Avg Fetch Time: {cache_stats['avg_fetch_time']:.4f}s, "
              f"Total Fetch Time: {cache_stats['total_fetch_time']:.4f}s, "
              f"Fetch Calls: {cache_stats['fetch_calls']}")
        print(f"Total Training Time: {total_train_time:.4f}s")

        with open("weighted_results.json", "w") as f:
            json.dump(results, f)

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size < 1:
        print("No GPU available, exiting.")
        exit()

    adj, features, labels, train_mask, val_mask, test_mask = load_cora_data()
    # adj, features, labels, train_mask, val_mask, test_mask = load_citeseer_data()
    # adj, features, labels, train_mask, val_mask, test_mask = load_pubmed_data()

    mp.spawn(distributed_train, args=(world_size, adj, features, labels, train_mask, val_mask, test_mask),
             nprocs=world_size, join=True)