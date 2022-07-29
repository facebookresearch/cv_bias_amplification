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

from torchvision.datasets import FashionMNIST
import torch.utils.data
import torch
import torchvision

from classy_vision.dataset import ClassyDataset, build_dataset, register_dataset
from classy_vision.dataset.transforms import build_transforms

import numpy as np
import classy_vision.generic.util as util

from PIL import Image


# Handle dataset so that we only get a subset of images (`task_classes`).
# Perform overlay transform (`attr_classes`) for a specific proportion of images (`epsilon`) with a specific strength (`eta`).
@register_dataset("cifar10_train_overlay")
class CIFAR10TrainOverlay(ClassyDataset):
    def __init__(self, 
                 batchsize_per_replica, 
                 shuffle, transform, 
                 num_samples, 
                 task_classes, 
                 attr_classes, 
                 eta,
                 epsilon,
                 seed):
        
        # Set up necessary variables
        assert len(task_classes) == 2 # assume class size = 2 for now
        assert len(attr_classes) == 2 # assume class size = 2 for now
        p = [np.round(0.5 + (epsilon * 0.01), 2), np.round(0.5 - (epsilon * 0.01), 2)]
        self.eta = eta
        
        # Grab original dataset
        dataset = torchvision.datasets.CIFAR10(root='./', download=True, train='train')
    
        # Instantiate storage for task and attribute images
        self.valid_cifar_idx_tasks = [] # will become a list of original idxs for the task-class subset.
        valid_cifar_idx_tasks_breakdown = {i:[] for i in task_classes} # key=task-class, value(to be)=original idx
        valid_cifar_idx_attrs_breakdown = {i:[] for i in attr_classes} # key=attr-class, value(to be)=original idx
        
        # Store indices for task and attribute images
        for i in range(len(dataset)):
            if dataset[i][1] in task_classes:
                self.valid_cifar_idx_tasks.append(i)
                valid_cifar_idx_tasks_breakdown[dataset[i][1]].append(i)
            if dataset[i][1] in attr_classes:
                valid_cifar_idx_attrs_breakdown[dataset[i][1]].append(i)        
                
        # Shuffle attribute images for random pairing
        with util.torch_seed(seed):
            with util.numpy_seed(seed):
                for key, _ in valid_cifar_idx_attrs_breakdown.items():
                    np.random.shuffle(valid_cifar_idx_attrs_breakdown[key])
        
        # Assign attribute-class based on task-class probability
        attr_breakdown = {} # key(to be)=task-class, value(to be)=attr-class
        for t, t_i in zip(task_classes, range(len(task_classes))):
            hold = [attr_classes[0]] * (int)(np.round(len(valid_cifar_idx_tasks_breakdown[t]) * p[t_i], 0)) + [attr_classes[1]] * (int)(np.round(len(valid_cifar_idx_tasks_breakdown[t]) * (1.0-p[t_i]), 0))
            with util.torch_seed(seed+1):
                with util.numpy_seed(seed+1):
                    np.random.shuffle(hold)
            attr_breakdown[t] = hold
        
        # Assign overlay image based on attribute-class assignment
        self.valid_cifar_idx_attrs= [None]*num_samples # will become a list of original idxs for the attr-class subset, aligned with corresponding idxs in task_idx_list
        self.valid_attrs = [None]*num_samples # will become a list of attr-classes, aligned with corresponding idxs in task_idx_list 
        attr_pointers = {attr:0 for attr in attr_classes} # used to parse self.attr_idx_subset for exact assignment
        for key, _ in valid_cifar_idx_tasks_breakdown.items():
            for cifar_task_idx, attr in zip(valid_cifar_idx_tasks_breakdown[key], attr_breakdown[key]):
                # images at a given `idx` for both attr_idx_list and task_idx_list should be overlayed on each other.
                # we use the pointers to ensure that a unique attr-class image is used for each task-class image.
                # this assumes that the dataset ordering does not change between iterations.
                self.valid_cifar_idx_attrs[self.valid_cifar_idx_tasks.index(cifar_task_idx)] = valid_cifar_idx_attrs_breakdown[attr][attr_pointers[attr]]
                self.valid_attrs[self.valid_cifar_idx_tasks.index(cifar_task_idx)] = attr
                attr_pointers[attr] += 1
        
        # Confirm there are the right number of samples 
        assert num_samples == len(self.valid_cifar_idx_tasks)
        assert num_samples == len(self.valid_cifar_idx_attrs)
        
        super().__init__(dataset, batchsize_per_replica, shuffle, transform, num_samples)
        
    def __getitem__(self, idx: int):
        assert idx >= 0 and idx < len(
            self.valid_cifar_idx_tasks
        ), "Provided idx ({}) is outside of dataset range".format(idx)
        sample = list(self.dataset[self.valid_cifar_idx_tasks[idx]])
        overlay_sample = list(self.dataset[self.valid_cifar_idx_attrs[idx]])
        attribute = self.valid_attrs[idx]
        
        sample.append(idx)
        
        # perform overlay transform
        img = sample[0]
        overlay_img = overlay_sample[0]
        assert isinstance(img, Image.Image), "img should be PIL Image. Got {}".format(
            type(img)
        )
        
        sample[0] = Image.blend(img, overlay_img, self.eta*0.01)
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
            task_classes=config["task_classes"],
            attr_classes = config["attr_classes"],
            eta=config["eta"],
            epsilon=config["epsilon"],
            seed=config["seed"]
        )



# Handle dataset so that we only get a subset of images (`task_classes`).
# Perform overlay transform (`attr_classes`) for a specific proportion of images (`epsilon`) with a specific strength (`eta`).
@register_dataset("cifar10_test_overlay")
class CIFAR10TestOverlay(ClassyDataset):
    def __init__(self, 
                 batchsize_per_replica, 
                 shuffle, transform, 
                 num_samples, 
                 task_classes, 
                 attr_classes, 
                 eta,
                 epsilon,
                 seed):
        
        # Set up necessary variables
        assert len(task_classes) == 2 # assume class size = 2 for now
        assert len(attr_classes) == 2 # assume class size = 2 for now
        p = [np.round(0.5 + (epsilon * 0.01), 2), np.round(0.5 - (epsilon * 0.01), 2)]
        self.eta = eta
        
        # Grab original dataset
        dataset = torchvision.datasets.CIFAR10(root='./', download=True, train=False)
    
        # Instantiate storage for task and attribute images
        self.valid_cifar_idx_tasks = [] # will become a list of original idxs for the task-class subset.
        valid_cifar_idx_tasks_breakdown = {i:[] for i in task_classes} # key=task-class, value(to be)=original idx
        valid_cifar_idx_attrs_breakdown = {i:[] for i in attr_classes} # key=attr-class, value(to be)=original idx
        
        # Store indices for task and attribute images
        for i in range(len(dataset)):
            if dataset[i][1] in task_classes:
                self.valid_cifar_idx_tasks.append(i)
                valid_cifar_idx_tasks_breakdown[dataset[i][1]].append(i)
            if dataset[i][1] in attr_classes:
                valid_cifar_idx_attrs_breakdown[dataset[i][1]].append(i)        
                
        # Shuffle attribute images for random pairing
        with util.torch_seed(seed):
            with util.numpy_seed(seed):
                for key, _ in valid_cifar_idx_attrs_breakdown.items():
                    np.random.shuffle(valid_cifar_idx_attrs_breakdown[key])
        
        # Assign attribute-class based on task-class probability
        attr_breakdown = {} # key(to be)=task-class, value(to be)=attr-class
        for t, t_i in zip(task_classes, range(len(task_classes))):
            hold = [attr_classes[0]] * (int)(np.round(len(valid_cifar_idx_tasks_breakdown[t]) * p[t_i], 0)) + [attr_classes[1]] * (int)(np.round(len(valid_cifar_idx_tasks_breakdown[t]) * (1.0-p[t_i]), 0))
            with util.torch_seed(seed+1):
                with util.numpy_seed(seed+1):
                    np.random.shuffle(hold)
            attr_breakdown[t] = hold
        
        # Assign overlay image based on attribute-class assignment
        self.valid_cifar_idx_attrs= [None]*num_samples # will become a list of original idxs for the attr-class subset, aligned with corresponding idxs in task_idx_list
        self.valid_attrs = [None]*num_samples # will become a list of attr-classes, aligned with corresponding idxs in task_idx_list 
        attr_pointers = {attr:0 for attr in attr_classes} # used to parse self.attr_idx_subset for exact assignment
        for key, _ in valid_cifar_idx_tasks_breakdown.items():
            for cifar_task_idx, attr in zip(valid_cifar_idx_tasks_breakdown[key], attr_breakdown[key]):
                # images at a given `idx` for both attr_idx_list and task_idx_list should be overlayed on each other.
                # we use the pointers to ensure that a unique attr-class image is used for each task-class image.
                # this assumes that the dataset ordering does not change between iterations.
                self.valid_cifar_idx_attrs[self.valid_cifar_idx_tasks.index(cifar_task_idx)] = valid_cifar_idx_attrs_breakdown[attr][attr_pointers[attr]]
                self.valid_attrs[self.valid_cifar_idx_tasks.index(cifar_task_idx)] = attr
                attr_pointers[attr] += 1
        
        # Confirm there are the right number of samples 
        assert num_samples == len(self.valid_cifar_idx_tasks)
        assert num_samples == len(self.valid_cifar_idx_attrs)
        
        super().__init__(dataset, batchsize_per_replica, shuffle, transform, num_samples)
        
    def __getitem__(self, idx: int):
        assert idx >= 0 and idx < len(
            self.valid_cifar_idx_tasks
        ), "Provided idx ({}) is outside of dataset range".format(idx)
        sample = list(self.dataset[self.valid_cifar_idx_tasks[idx]])
        overlay_sample = list(self.dataset[self.valid_cifar_idx_attrs[idx]])
        attribute = self.valid_attrs[idx]
        
        sample.append(idx)
        
        # perform overlay transform
        img = sample[0]
        overlay_img = overlay_sample[0]
        assert isinstance(img, Image.Image), "img should be PIL Image. Got {}".format(
            type(img)
        )
        
        sample[0] = Image.blend(img, overlay_img, self.eta*0.01)
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
            task_classes=config["task_classes"],
            attr_classes = config["attr_classes"],
            eta=config["eta"],
            epsilon=config["epsilon"],
            seed=config["seed"]
        )