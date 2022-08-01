# Copyright (c) Meta Platforms, Inc. and affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torchvision.datasets import FashionMNIST
import torch.utils.data
import torch
from torchvision import datasets, transforms
import classy_vision.generic.util as util

import torchvision
import math
import numpy as np
import json

from classy_vision.dataset import ClassyDataset, build_dataset, register_dataset
from classy_vision.dataset.transforms import build_transforms, ClassyTransform, register_transform

from PIL import Image


# Handle dataset so that we only get a subset of images (`task_classes`).
# Perform overlay transform (`attr_classes`) for a specific proportion of images (`epsilon`) with a specific strength (`eta`).
@register_dataset("cifar100_random_sample_train")
class CIFAR100RandomSampleTrain(ClassyDataset):
    def __init__(self, 
                 batchsize_per_replica, 
                 shuffle, transform, 
                 num_samples,
                 dataset_size,
                 p,
                 seed,
                 class_mapping):
        
        
        # Grab original dataset
        dataset = torchvision.datasets.CIFAR100(root='./', download=True, train='train')
    
         
        # Instantiate storage for task images
        self.valid_cifar_idx_tasks = [] # will become a list of original idxs for the task-class subset.
        self.mapped_classes = [] # will become a list of mapped classes for the task-class subset.
        valid_cifar_idx_tasks_breakdown = {i:[] for i in range(0,len(class_mapping))} # key=task-class, value(to be)=original idx
        
        # Store indices for task images
        for i in range(len(dataset)):
            valid_cifar_idx_tasks_breakdown[dataset[i][1]].append(i)   
        
        # Shuffle task images for selecting subset
        with util.torch_seed(seed):
            with util.numpy_seed(seed):
                for key, _ in valid_cifar_idx_tasks_breakdown.items():
                    np.random.shuffle(valid_cifar_idx_tasks_breakdown[key])
        
        class_size = int(np.rint(500*dataset_size))
        # Collect task images and class mappings for CIFAR100 subset
        for key, _ in valid_cifar_idx_tasks_breakdown.items():
            self.valid_cifar_idx_tasks.extend(valid_cifar_idx_tasks_breakdown[key][0:class_size])
            self.mapped_classes.extend([class_mapping[key]]*class_size)
        
        # Assign attribute based on task-class probability
        attr_breakdown = {} # key(to be)=task-class, value(to be)=attr-class
        with util.torch_seed(seed+1):
            with util.numpy_seed(seed+1):
                for key, _ in valid_cifar_idx_tasks_breakdown.items():
                    hold = [1] * (int)(np.round(class_size * p[class_mapping[key]], 0)) + [0] * (int)(np.round(class_size * (1.0-p[class_mapping[key]]), 0))
                    np.random.shuffle(hold)
                    attr_breakdown[key] = hold
        
        # Assign overlay image based on attribute-class assignment
        self.valid_attrs = [None]*class_size*100 # will become a list of attr-classes, aligned with corresponding idxs in task_idx_list 
        for key, _ in valid_cifar_idx_tasks_breakdown.items():
            for cifar_task_idx, attr in zip(valid_cifar_idx_tasks_breakdown[key], attr_breakdown[key]):
                # this assumes that the dataset ordering does not change between iterations.
                self.valid_attrs[self.valid_cifar_idx_tasks.index(cifar_task_idx)] = 'b' if attr else 'a'
        
        # Confirm there are the right number of samples 
        assert num_samples == len(self.valid_cifar_idx_tasks)
        assert num_samples == len(self.mapped_classes)
        assert num_samples == len(self.valid_attrs)
        
        super().__init__(dataset, batchsize_per_replica, shuffle, transform, num_samples)
        
    def __getitem__(self, idx: int):
        assert idx >= 0 and idx < len(
            self.valid_cifar_idx_tasks
        ), "Provided idx ({}) is outside of dataset range".format(idx)
        sample = list(self.dataset[self.valid_cifar_idx_tasks[idx]])
        img = sample[0]
        mapped_label = self.mapped_classes[idx]
        attribute = self.valid_attrs[idx]
        
        # perform overlay transform
        assert isinstance(img, Image.Image), "img should be PIL Image. Got {}".format(
            type(img)
        )

        img = np.asarray(img)
        img_dtype = img.dtype
        
        img = 255 - img if attribute == 'b' else img
        
        sample[0] = Image.fromarray(img.astype(img_dtype))
        sample.append(idx)
        sample.append(mapped_label)
        sample.append(attribute)
        sample = tuple(sample)         # TODO: Update future transforms with this new ordering.
        if self.transform is None:
            return sample
        return self.transform(sample) 

    def __len__(self):
        return len(self.valid_cifar_idx_tasks)
        
    @classmethod
    def from_config(cls, config):
        transform = build_transforms(config["transforms"])
        return cls(
            batchsize_per_replica=config["batchsize_per_replica"],
            shuffle=config["shuffle"],
            transform=transform,
            num_samples=config["num_samples"],
            dataset_size=config["dataset_size"],
            p=config["p"],
            seed=config["seed"],
            class_mapping=config["class_mapping"]
        )

