import random

from src.pytorch_balanced_sampler.sampler import SamplerFactory


def get_batch_sampler(dataset, batch_size, shuffle):
    class_idxs = [
        [i for i, l in enumerate(dataset.labels) if l == 0],
        [i for i, l in enumerate(dataset.labels) if l == 1],
    ]
    if shuffle:
        random.shuffle(class_idxs[0])
        random.shuffle(class_idxs[1])
    n_batches = len(dataset.labels) // batch_size
    batch_sampler = SamplerFactory().get(
        class_idxs=class_idxs,
        batch_size=batch_size,
        n_batches=n_batches,
        alpha=1,
        kind="fixed",
    )
    return batch_sampler
