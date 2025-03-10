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
from flask import Flask, request, jsonify
from flask_cors import CORS

from load_data.load_pubmed_data import load_pubmed_data
from load_data.load_cora_data import load_cora_data
from load_data.load_citeseer_data import load_citeseer_data
from model.gcn import GCN
from eval.evaluate import evaluate

app = Flask(__name__)
CORS(app)  # 启用 CORS


# 初始化分布式环境
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


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
        self.features = features
        self.adj = adj
        self.gpuid = gpuid
        self.device = torch.device(f'cuda:{gpuid}')
        self.fetch_time_total = 0
        self.fetch_calls = 0

    def fetch_features(self, nids):
        start_time = time.time()
        nids = nids.to(self.device)
        features = self.features[nids.cpu()].to(self.device)
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


class OutDegreeGraphCacheServer:
    def __init__(self, features, adj, gpuid, max_cache_size_mb=10):
        self.features = features
        self.adj = adj
        self.gpuid = gpuid
        self.device = torch.device(f'cuda:{gpuid}')
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        self.feature_size = features.element_size() * features.shape[1]
        self.max_capacity = self.max_cache_size_bytes // self.feature_size
        self.hit_count = 0
        self.total_count = 0
        self.fetch_time_total = 0
        self.fetch_calls = 0
        self._initialize_cache()

    def _initialize_cache(self):
        start_time = time.time()
        torch.cuda.set_device(self.device)
        self.capacity = min(self.max_capacity, self.features.shape[0])
        self.cached_nids = torch.zeros(self.capacity, dtype=torch.long, device=self.device)
        self.cached_features = torch.zeros(self.capacity, self.features.shape[1], dtype=self.features.dtype,
                                           device=self.device)
        adj_raw = (self.adj > 0).float()
        degrees = adj_raw.sum(dim=1).cpu().numpy()
        top_indices = np.argpartition(degrees, -self.capacity)[-self.capacity:]
        self.cached_nids[:] = torch.tensor(top_indices, device=self.device)
        self.cached_features[:] = self.features[top_indices].to(self.device)
        self.cache_lookup = torch.full((self.features.shape[0],), -1, device=self.device, dtype=torch.long)
        self.cache_lookup[self.cached_nids] = torch.arange(self.capacity, device=self.device)
        self.cache_init_time = time.time() - start_time

    def fetch_features(self, nids):
        start_time = time.time()
        nids = nids.to(self.device)
        cache_indices = self.cache_lookup[nids]
        mask = (cache_indices >= 0)
        self.hit_count += mask.sum().item()
        self.total_count += nids.size(0)
        features = torch.zeros((nids.size(0), self.features.shape[1]), device=self.device)
        if mask.any():
            hit_indices = cache_indices[mask]
            features[mask] = self.cached_features[hit_indices]
        if (~mask).any():
            miss_nids = nids[~mask]
            features[~mask] = self.features[miss_nids.cpu()].to(self.device)
        fetch_time = time.time() - start_time
        self.fetch_time_total += fetch_time
        self.fetch_calls += 1
        return features

    def get_cache_stats(self):
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


class WeightedGraphCacheServer:
    def __init__(self, features, adj, gpuid, max_cache_size_mb=10):
        self.features = features
        self.adj = adj
        self.gpuid = gpuid
        self.device = torch.device(f'cuda:{gpuid}')
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        self.feature_size = features.element_size() * features.shape[1]
        self.max_capacity = self.max_cache_size_bytes // self.feature_size
        self.hit_count = 0
        self.total_count = 0
        self.fetch_time_total = 0
        self.fetch_calls = 0
        self._initialize_cache()

    def _initialize_cache(self):
        start_time = time.time()
        torch.cuda.set_device(self.device)
        self.capacity = min(self.max_capacity, self.features.shape[0])
        self.cached_nids = torch.zeros(self.capacity, dtype=torch.long, device=self.device)
        self.cached_features = torch.zeros(self.capacity, self.features.shape[1], dtype=self.features.dtype,
                                           device=self.device)
        adj_raw = (self.adj > 0).float()
        node_degrees = adj_raw.sum(dim=1)
        neighbor_degrees = adj_raw @ node_degrees
        total_degrees = 0.33 * node_degrees + 0.67 * neighbor_degrees
        top_indices = np.argpartition(total_degrees.cpu().numpy(), -self.capacity)[-self.capacity:]
        self.cached_nids[:] = torch.tensor(top_indices, device=self.device)
        self.cached_features[:] = self.features[top_indices].to(self.device)
        self.cache_lookup = torch.full((self.features.shape[0],), -1, device=self.device, dtype=torch.long)
        self.cache_lookup[self.cached_nids] = torch.arange(self.capacity, device=self.device)
        self.cache_init_time = time.time() - start_time

    def fetch_features(self, nids):
        start_time = time.time()
        nids = nids.to(self.device)
        cache_indices = self.cache_lookup[nids]
        mask = (cache_indices >= 0)
        self.hit_count += mask.sum().item()
        self.total_count += nids.size(0)
        features = torch.zeros((nids.size(0), self.features.shape[1]), device=self.device)
        if mask.any():
            hit_indices = cache_indices[mask]
            features[mask] = self.cached_features[hit_indices]
        if (~mask).any():
            miss_nids = nids[~mask]
            features[~mask] = self.features[miss_nids.cpu()].to(self.device)
        fetch_time = time.time() - start_time
        self.fetch_time_total += fetch_time
        self.fetch_calls += 1
        return features

    def get_cache_stats(self):
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


def distributed_train(rank, world_size, adj, features, labels, train_mask, val_mask, test_mask, cache_type,
                      num_epochs=200):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    if rank == 0:
        adj = adj.to(device)
        features = features.to('cpu' if cache_type == 'nocache' else device)
        labels = labels.to(device)
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)
    else:
        adj, labels, train_mask, val_mask, test_mask = None, None, None, None, None
        features = features if cache_type == 'nocache' else None

    adj = broadcast_tensor(adj, 0, world_size, device)
    labels = broadcast_tensor(labels, 0, world_size, device)
    train_mask = broadcast_tensor(train_mask, 0, world_size, device)
    val_mask = broadcast_tensor(val_mask, 0, world_size, device)
    test_mask = broadcast_tensor(test_mask, 0, world_size, device)
    if cache_type != 'nocache':
        features = broadcast_tensor(features, 0, world_size, device)

    if cache_type == 'nocache':
        cache_server = NoCacheGraphServer(features, adj, rank)
    elif cache_type == 'outdegree':
        cache_server = OutDegreeGraphCacheServer(features, adj, rank)
    else:  # weighted
        cache_server = WeightedGraphCacheServer(features, adj, rank)

    num_classes = labels.max().item() + 1
    model = GCN(in_features=features.shape[1], hidden_features=16, out_features=num_classes).to(device)
    model = DDP(model, device_ids=[rank])
    dist.barrier()

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    total_train_time = 0
    results = {
        "train_loss": [],
        "train_acc": [],
        "val_acc": [],
        "test_acc": 0,
        "cache_stats": {}
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
                results["train_loss"].append(float(loss.item()))
                results["train_acc"].append(float(train_acc))
                results["val_acc"].append(float(val_acc))

    dist.barrier()
    if rank == 0:
        test_acc = evaluate(model, adj, features, labels, test_mask, cache_server, device)
        cache_stats = cache_server.get_cache_stats()
        results["test_acc"] = float(test_acc)
        results["cache_stats"] = cache_stats

    cleanup()
    if rank == 0:
        return results


@app.route('/train', methods=['POST'])
def train():
    data = request.json
    dataset = data.get('dataset')
    cache_type = data.get('cache_type', 'nocache')

    if dataset == 'cora':
        adj, features, labels, train_mask, val_mask, test_mask = load_cora_data()
    elif dataset == 'citeseer':
        adj, features, labels, train_mask, val_mask, test_mask = load_citeseer_data()
    elif dataset == 'pubmed':
        adj, features, labels, train_mask, val_mask, test_mask = load_pubmed_data()
    else:
        return jsonify({"error": "Invalid dataset"}), 400

    world_size = torch.cuda.device_count()
    if world_size < 1:
        return jsonify({"error": "No GPU available"}), 500

    # 使用 mp.spawn 运行分布式训练
    mp.spawn(distributed_train, args=(world_size, adj, features, labels, train_mask, val_mask, test_mask, cache_type),
             nprocs=world_size, join=True)

    # 读取主进程的结果（假设保存在文件或通过其他方式传递）
    with open(f"{cache_type}_results.json", "r") as f:  # 假设结果保存到文件
        results = json.load(f)
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)