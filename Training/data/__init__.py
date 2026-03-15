import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from .custom_transforms import *
from .datasets import RealFakeDataset, custom_collate_fn


def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1.0 / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights)
    )
    return sampler


def create_dataloader(opt, preprocess=None, return_dataset=False):

    shuffle = True if opt.isTrain else False

    # 每个样本在 collate 后会展开为 6 张图（real×1 + real_resized×2 + fake×1 + fake_resized×2），
    # 因此 DataLoader 的 batch_size 除以 6，使最终送入模型的 tensor 行数等于 opt.batch_size。
    batch_size = max(1, opt.batch_size // 6)
    dataset = RealFakeDataset(opt)

    sampler = None
    print(len(dataset))
    if return_dataset:
        return dataset

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=opt.num_threads,
        pin_memory=True,
        drop_last=opt.isTrain,
        collate_fn=custom_collate_fn,
    )
    return data_loader
