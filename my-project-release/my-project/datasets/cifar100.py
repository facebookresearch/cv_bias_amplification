#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Callable, Dict, Optional, Union

from classy_vision.dataset import ClassyDataset, register_dataset
from classy_vision.dataset.core.random_image_datasets import (
    RandomImageBinaryClassDataset,
    SampleType,
)
from classy_vision.dataset.transforms import ClassyTransform, build_transforms

from torchvision.datasets import CIFAR100
import torch.utils.data
import torch
import torchvision

from classy_vision.dataset import ClassyDataset, build_dataset, register_dataset
from classy_vision.dataset.transforms import build_transforms

    
@register_dataset("cifar100_train")
class CIFAR100Train(ClassyDataset):
    def __init__(self, batchsize_per_replica, shuffle, transform, num_samples):
        dataset = torchvision.datasets.CIFAR100(root='./', download=True, train='train')
        super().__init__(dataset, batchsize_per_replica, shuffle, transform, num_samples)
        
    def __getitem__(self, idx: int):
        assert idx >= 0 and idx < len(
            self.dataset
        ), "Provided idx ({}) is outside of dataset range".format(idx)
        sample = list(self.dataset[idx])
        sample.append(idx)
        sample = tuple(sample)
        if self.transform is None:
            return sample
        return self.transform(sample)
        
    @classmethod
    def from_config(cls, config):
        transform = build_transforms(config["transforms"])
        return cls(
            batchsize_per_replica=config["batchsize_per_replica"],
            shuffle=config["shuffle"],
            transform=transform,
            num_samples=None,
        )

@register_dataset("cifar100_test")
class MyClassyDatasetCIFAR100Test(ClassyDataset):
    def __init__(self, batchsize_per_replica, shuffle, transform, num_samples):
        dataset = torchvision.datasets.CIFAR100(root='./', download=True, train=False)
        super().__init__(dataset, batchsize_per_replica, shuffle, transform, num_samples)
        
    def __getitem__(self, idx: int):
        assert idx >= 0 and idx < len(
            self.dataset
        ), "Provided idx ({}) is outside of dataset range".format(idx)
        sample = list(self.dataset[idx])
        sample.append(idx)
        sample = tuple(sample)
        if self.transform is None:
            return sample
        return self.transform(sample)
        
    @classmethod
    def from_config(cls, config):
        transform = build_transforms(config["transforms"])
        return cls(
            batchsize_per_replica=config["batchsize_per_replica"],
            shuffle=config["shuffle"],
            transform=transform,
            num_samples=None,
        )

