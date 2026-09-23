"""KINE-Bench v0.1 tasks. Every probe runs on a frozen model, single-GPU (or CPU).

Tasks:
  KINE-TEMP-1  temporal-order probe  — can pooled features tell original from frame-shuffled?
  KINE-MOT-1   motion-magnitude probe — do features encode how much is moving?
  KINE-FUT-1   future-prediction fidelity — does the predictor's "imagination" of masked
               future frames match what the target encoder actually sees?
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def _pooled(encoder, videos, device):
    feats = encoder(videos.to(device))  # (B, N, D)
    return feats.mean(dim=1)            # (B, D)


def _split_indices(n, seed=0, groups=None):
    """Split complete source-video groups so augmented views cannot cross sets."""
    g = torch.Generator().manual_seed(seed)
    if groups is None:
        perm = torch.randperm(n, generator=g)
        n_test = max(2, min(int(n * 0.3), n - 2))
        return perm[n_test:], perm[:n_test]
    groups = torch.as_tensor(groups)
    if len(groups) != n:
        raise ValueError("groups must have one entry per probe sample")
    unique = torch.unique(groups)
    if len(unique) < 2:
        raise ValueError("grouped probe requires at least two source videos")
    perm = unique[torch.randperm(len(unique), generator=g)]
    n_test = max(1, min(int(len(unique) * 0.3), len(unique) - 1))
    test_mask = torch.isin(groups, perm[:n_test])
    return torch.where(~test_mask)[0], torch.where(test_mask)[0]


def _train_probe(X, y, kind="cls", epochs=400, lr=3e-3, seed=0, groups=None):
    train_idx, test_idx = _split_indices(X.shape[0], seed=seed, groups=groups)
    probe = nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    Xt, yt = X[train_idx], y[train_idx]
    for _ in range(epochs):
        logits = probe(Xt).squeeze(-1)
        if kind == "cls":
            loss = F.binary_cross_entropy_with_logits(logits, yt)
        else:
            loss = F.mse_loss(logits, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return probe, test_idx


def temporal_order(model, clips, device, seed=0):
    """KINE-TEMP-1: linear probe separates original clips from temporally shuffled ones."""
    torch.manual_seed(seed)
    random.seed(seed)
    T = clips[0].shape[1]
    if T < 2:
        raise ValueError("temporal-order probe requires at least two frames per clip")
    shuf = []
    for c in clips:
        p = torch.randperm(T)
        while torch.equal(p, torch.arange(T)):
            p = torch.randperm(T)
        shuf.append(c[:, p])
    orig = torch.stack(clips)
    shuf = torch.stack(shuf)
    X = torch.cat([
        _pooled(model.target, orig, device),
        _pooled(model.target, shuf, device),
    ], dim=0).cpu().float()
    y = torch.cat([torch.ones(len(clips)), torch.zeros(len(clips))])
    # Positive and shuffled views of one clip must stay in the same split.
    groups = torch.arange(len(clips)).repeat(2)
    probe, test_idx = _train_probe(X, y, kind="cls", seed=seed, groups=groups)
    with torch.no_grad():
        pred = (torch.sigmoid(probe(X[test_idx]).squeeze(-1)) > 0.5).float()
        acc = (pred == y[test_idx]).float().mean().item()
    return {"accuracy": round(acc, 4), "baseline": 0.5}


def motion_magnitude(model, clips, device, seed=0):
    """KINE-MOT-1: linear regression from features to clip motion energy (Pearson r)."""
    torch.manual_seed(seed)
    videos = torch.stack(clips)
    X = _pooled(model.target, videos, device).cpu().float()
    gt = torch.stack([(c[:, 1:] - c[:, :-1]).abs().mean() for c in clips])
    gt = (gt - gt.mean()) / (gt.std() + 1e-6)
    probe, test_idx = _train_probe(X, gt, kind="reg", epochs=600, seed=seed)
    with torch.no_grad():
        pred = probe(X[test_idx]).squeeze(-1)
        r = torch.corrcoef(torch.stack([pred, gt[test_idx]]))[0, 1].item()
    return {"pearson_r": round(float(r), 4), "baseline": 0.0}


def future_prediction(model, clips, device, seed=0):
    """KINE-FUT-1: mask the temporal second half, let the predictor imagine it,
    score cosine similarity against the target encoder's real future tokens."""
    torch.manual_seed(seed)
    gt_n, gh, gw = model.grid
    n = gt_n * gh * gw
    pos = torch.arange(n, device=device)
    t_of = pos // (gh * gw)
    mask_flat = t_of >= gt_n // 2
    B = len(clips)
    mask_idx = pos[mask_flat].unsqueeze(0).expand(B, -1)
    vis_idx = pos[~mask_flat].unsqueeze(0).expand(B, -1)
    videos = torch.stack(clips).to(device)
    with torch.no_grad():
        full = F.normalize(model.target(videos), dim=-1)
        target = torch.gather(full, 1, mask_idx.unsqueeze(-1).expand(-1, -1, full.shape[-1]))
        visible = model.encoder(videos, visible_idx=vis_idx)
        pred = model.predictor(visible, vis_idx, mask_idx)
        cos = F.cosine_similarity(pred, target, dim=-1).mean().item()
        flat = full.reshape(-1, full.shape[-1])
        perm = torch.randperm(flat.shape[0], device=device)
        base = F.cosine_similarity(flat, flat[perm], dim=-1).mean().item()
    return {"cosine": round(float(cos), 4), "random_baseline": round(float(base), 4)}


ALL_TASKS = [
    ("KINE-TEMP-1", temporal_order),
    ("KINE-MOT-1", motion_magnitude),
    ("KINE-FUT-1", future_prediction),
]
