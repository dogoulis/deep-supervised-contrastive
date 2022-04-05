import torch


def BarlowTwinsLoss(class_batch, lambda_param=5e-3):
    N, _ = class_batch.size()
    # cross-correlation matrix
    c = torch.mm(class_batch[: N // 2].T, class_batch[N // 2 :]) / (N // 2)
    # invariance term
    on_diag = torch.diagonal(c).add(-1).pow(2).sum()
    # redundancy reduction term
    off_diag = off_diagonal(c).pow(2).sum()
    loss = on_diag + lambda_param * off_diag
    return loss

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
