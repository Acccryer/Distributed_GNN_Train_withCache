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

class NoCacheGraphServer:
    def __init__(self, features, adj, gpuid):
        self.features = features  # features在CPU上
        self.adj = adj
        self.gpuid = gpuid
        self.device = torch.device(f'cuda:{gpuid}')
        self.fetch_time_total = 0
        self.fetch_calls = 0

    def fetch_features(self, nids):
        start_time = time.time()
        nids = nids.to(self.device)
        features = self.features[nids.cpu()].to(self.device)  # 从CPU获取并传输到GPU
        fetch_time = time.time() - start_time
        self.fetch_time_total += fetch_time
        self.fetch_calls += 1
        return features

    def get_cache_stats(self):
        avg_fetch_time = self.fetch_time_total / self.fetch_calls if self.fetch_calls > 0 else 0
        return {
            "cache_init_time": 0,
            "hit_rate": 0,
            "avg_fetch_time": avg_fetch_time,
            "total_fetch_time": self.fetch_time_total,
            "fetch_calls": self.fetch_calls,
            "cache_size_mb": 0,
            "capacity": 0,
            "hit_count": 0,
            "total_count": 0
        }

# 分布式训练函数
def distributed_train(rank, world_size, adj, features, labels, train_mask, val_mask, test_mask, num_epochs=200):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    if rank == 0:
        adj = adj.to(device)
        features = features.to('cpu')  # features在CPU上
        labels = labels.to(device)
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)
    else:
        adj, labels, train_mask, val_mask, test_mask = None, None, None, None, None

    adj = broadcast_tensor(adj, 0, world_size, device)
    labels = broadcast_tensor(labels, 0, world_size, device)
    train_mask = broadcast_tensor(train_mask, 0, world_size, device)
    val_mask = broadcast_tensor(val_mask, 0, world_size, device)
    test_mask = broadcast_tensor(test_mask, 0, world_size, device)

    cache_server = NoCacheGraphServer(features, adj, rank)

    num_classes = labels.max().item() + 1
    model = GCN(in_features=features.shape[1], hidden_features=16, out_features=num_classes).to(device)
    model = DDP(model, device_ids=[rank])
    dist.barrier()

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
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
        fetched_features = cache_server.fetch_features(nids)
        output = model(adj, fetched_features)
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

    dist.barrier()

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

        with open("nocache_results.json", "w") as f:
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