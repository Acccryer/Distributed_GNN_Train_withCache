import torch
import torch.distributed as dist


def evaluate(model, adj, features, labels, mask, cache_server, device):
    model.eval()
    with torch.no_grad():
        nids = torch.arange(adj.shape[0], device=device)
        cached_features = cache_server.fetch_features(nids)
        output = model(adj, cached_features)
        preds = output[mask].max(1)[1]
        correct = preds.eq(labels[mask]).sum().item()
        total = mask.sum().item()
        correct_tensor = torch.tensor(correct, device=device)
        total_tensor = torch.tensor(total, device=device)
        dist.all_reduce(correct_tensor)
        dist.all_reduce(total_tensor)
        accuracy = correct_tensor.item() / total_tensor.item() if total_tensor.item() > 0 else 0
    return accuracy