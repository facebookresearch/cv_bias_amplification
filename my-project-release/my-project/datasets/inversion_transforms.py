# Copyright (c) Meta Platforms, Inc. and affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from PIL import Image
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json
from classy_vision.dataset.transforms import build_transforms, ClassyTransform, register_transform
from classy_vision.dataset import build_dataset
import classy_vision.generic.util as util
from collections.abc import Iterable
import cv2
import random
import time
import logging

@register_transform("invert")
class Invert(ClassyTransform):
    """With probablity p_class, invert the image. 

    Args:
        p (dict <int: float>): Probabilities for each class.
        seed (int): Seed used for replication.
    """

    def __init__(self, p, seed):
        self.p = p
        self.seed = seed
        
    def __call__(self, sample):
        """
        Args:
            sample (tuple): Image to be altered and its class

        Returns:
            tuple: (Altered image, class)
        """
        
        img = sample[0]
        original_label = sample[1]
        sample_id = sample[2]
        mapped_label = sample[3] 
        
        assert isinstance(img, Image.Image), "img should be PIL Image. Got {}".format(
            type(img)
        )
        
        img = np.asarray(img)
        img_dtype = img.dtype
        
        if self.seed >= 0:
            with util.torch_seed(self.seed + sample_id):
                with util.numpy_seed(self.seed + sample_id):
                    attribute = 'b' if np.random.rand() < self.p[mapped_label] else 'a'
        else:
            attribute = 'b' if np.random.rand() < self.p[mapped_label] else 'a'
        
        img = 255 - img if attribute == 'b' else img

        img = img.astype(img_dtype)

        return (Image.fromarray(img), original_label, sample_id, mapped_label, attribute)

    def __repr__(self):
        return self.__class__.__name__


@register_transform("invert_exact")
class InvertExact(ClassyTransform):
    """Invert the image according to the provided inversion list.

    Args:
        invert (list <int>): Whether or not the image at index i should be inverted
    """

    def __init__(self, invert):
        self.invert = invert
        
    def __call__(self, sample):
        """
        Args:
            sample (tuple): Image to be altered and its class

        Returns:
            tuple: (Altered image, class)
        """
        
        img = sample[0]
        original_label = sample[1]
        sample_id = sample[2]
        mapped_label = sample[3] 
        
        assert isinstance(img, Image.Image), "img should be PIL Image. Got {}".format(
            type(img)
        )
        
        img = np.asarray(img)
        img_dtype = img.dtype
        
        attribute = 'b' if self.invert[sample_id] else 'a'
        
        img = 255 - img if attribute == 'b' else img
            
        img = img.astype(img_dtype)

        return (Image.fromarray(img), original_label, sample_id, mapped_label, attribute)

    def __repr__(self):
        return self.__class__.__name__

        
@register_transform("assign_class")
class AssignClass(ClassyTransform):
    """Re-assign each image class to a given class. 

    Args:
        classes (dict <int: int>): New class assignments, with current class:new class
    """

    def __init__(self, classes):
        self.classes = classes
        
    def __call__(self, sample):
        """
        Args:
            sample (tuple): Class to be altered and its image.

        Returns:
            tuple: (Altered image, class)
        """
        img = sample[0]
        label = sample[1]
        sample_id = sample[2]

        return (img, label, sample_id, self.classes[label])

    def __repr__(self):
        return self.__class__.__name__


@register_transform("swap_task_attr")
class SwapTaskAttr(ClassyTransform):
    """Switch the task and attribute.

    Converts the original attribute to a numeric form. 
    """
    def __call__(self, sample):
        img = sample[0]
        original_label = sample[1]
        sample_id = sample[2]
        mapped_label = sample[3] 
        attribute = sample[4]

        return (img, original_label, sample_id, ord(attribute)-97, mapped_label)

    def __repr__(self):
        return self.__class__.__name__


@register_transform("assign_class_str")
class AssignClassStr(ClassyTransform):
    """Re-assign the image to a given class. 

    Args:
        classes (dict <int: int>): New class assignments, with current class:new class
    """

    def __init__(self, classes):
        self.classes = classes
        
    def __call__(self, sample):
        """
        Args:
            sample (tuple): Class to be altered and its image.

        Returns:
            tuple: (Altered image, class)
        """
        img = sample[0]
        label = sample[1]
        sample_id = sample[2]
        attribute = sample[3]


        return (img, label, sample_id, attribute, self.classes[str(label)])

    def __repr__(self):
        return self.__class__.__name__


@register_transform("rand_assign_class_rand_invert")
class RandAssignClassRandInvert(ClassyTransform):
    """Helper function to make configs easier to write. Warning: Requires
    dataset to be registered before transform is called. Requires dataset
    to be cheap to do one pass over to create maps when transform is
    created

    Randomly assigns the original class targets to a new, smaller, set
    of class targets. The original class set will be evenly divided
    among the new classes Then inverts images with probability p based
    on the probability map provided.
    
    Args:
       num_new_classes (int): New set of classes
       invert_probs (array[float]): Inversion probability for each class
       dataset_name (string): Already registered dataset for retrieving class info
       exact (bool): Exact number of inversions (i.e. class_1: 0.5 => exactly half of of class_1 images will be inverted vs flipping a coin for each image)
       assignment_seed (optional int): This is the seed used for the random generation ... must be same if you want the class mapping to match for test set
       inversion_seed (optional int): This is the seed for actually inverting each image. If None, uses time.
    """
    def __init__(self, num_new_classes, invert_probs, dataset_config, exact=True, assignment_seed=0, inversion_seed=0):
        # assertions
        assert len(invert_probs) == num_new_classes, "inversion probabilities must match the number of final classes"
        assert assignment_seed is not None, "Assignment seed cannot be None otherwise it will be impossible to track the mapping"
        for i in range(0, num_new_classes):
            assert invert_probs[i] >= 0.0 and invert_probs[i] <= 1.0, "Valid probabilities must be provided"

        if inversion_seed is None:
            inversion_seed = int(time.time())
        # For most datasets, only the name is required, we set batchsize, shuffle, transforms, num_workers
        dataset_config["batchsize_per_replica"] = 1
        dataset_config["shuffle"] = False
        dataset_config["transforms"] = []
        dataset_config["num_workers"] = 0

        # Get target mapping directly from dataset
        dataset = build_dataset(dataset_config)
        index_to_class_mapping = {}
        target_counts = {}
        for i in range(0, len(dataset)):
            sample = dataset[i]
            index_to_class_mapping[i] = {"original_target": sample[1]}
            if sample[1] not in target_counts:
                target_counts[sample[1]] = 0
            target_counts[sample[1]] += 1

        target_list = list(target_counts.keys())
        target_list.sort()
        new_target_list = []
        quotient = len(target_list) // num_new_classes
        remainder = len(target_list) % num_new_classes
        # Create correct number of new class instances
        for i in range(0, num_new_classes):
            num = quotient
            if i < remainder:
                num += 1

            new_target_list += [i for j in range(0, num)]

        with util.numpy_seed(assignment_seed):
            np.random.shuffle(new_target_list)
        class_mapping = dict(zip(target_list, new_target_list))
        logging.info("Classy mapping: {}".format(str(class_mapping)))
        self.random_assign = AssignClass(class_mapping)

        # Now that we have our random assignment, need our exact list
        inversion_counts = {}
        for i in range(0, len(target_counts)):
            if class_mapping[i] not in inversion_counts:
                inversion_counts[class_mapping[i]] = 0
            inversion_counts[class_mapping[i]] += target_counts[i]

        target_to_inversion_lists = {}
        target_to_inversion_iterators = []
        for i in range(0, len(invert_probs)):
            prob = invert_probs[i]
            count = inversion_counts[i]
            target_to_inversion_lists[i] = [0] * round(count * (1 - prob)) + [1] * round(count * prob)
            with util.numpy_seed(inversion_seed):
                np.random.shuffle(target_to_inversion_lists[i])
            target_to_inversion_iterators.append(iter(target_to_inversion_lists[i]))
                
        inversions = [None] * len(dataset)
        for i in range(0, len(dataset)):
            it = target_to_inversion_iterators[class_mapping[index_to_class_mapping[i]["original_target"]]]
            inversions[i] = next(it)

        logging.info("Inversions: {}".format(str(inversions)))
        self.exact_invert = InvertExact(inversions)

    def __call__(self, sample):
        new_sample = self.random_assign(sample)
        new_sample = self.exact_invert(new_sample)
        return new_sample

    def __repr__(self):
        return self.__class__.__name__


@register_transform("PadToSize")
class PadToSize(ClassyTransform):
    """
    Pad the input PIL Image so that it has the specified size. The image is returned
    unchanged if at least one dimension of the original image is larger than the
    corresponding dimension in the requested size.

    Args:
        size (sequence): Output size (height, width)
        border_type (string): The type cv2 border type to use.
        pad_both_sides (bool): True: add padding to both sides to keep the image
        in the centre; False: add padding to the right and/or bottom.
    """

    def __init__(
        self,
        size,
        border_type="BORDER_CONSTANT",
        pad_both_sides=True,
    ):
        self.size = size
        self.pad_both_sides = pad_both_sides
        self.border_type = self._getBorderType(border_type)
        assert (
            isinstance(size, Iterable) and len(size) == 2
        ), "Got inappropriate size arg: {}. Expected a sequence (h, w)".format(
            type(size)
        )

    def _pad(self, img: Image.Image) -> Image.Image:
        padding = self._get_padding(img)
        assert len(padding) == 2
        padding_tlbr = self._get_padding_tlbr(padding)

        if (
            padding_tlbr[0] > 0
            or padding_tlbr[1] > 0
            or padding_tlbr[2] > 0
            or padding_tlbr[3] > 0
        ):
            padded = cv2.copyMakeBorder(
                cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR),
                padding_tlbr[0],
                padding_tlbr[2],
                padding_tlbr[1],
                padding_tlbr[3],
                self.border_type,
                value=[0, 0, 0],  # black
            )
            result_img = Image.fromarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))

        return result_img

    def _getBorderType(self, border_type: str) -> int:
        if border_type == "BORDER_CONSTANT":
            return cv2.BORDER_CONSTANT
        elif border_type == "BORDER_REFLECT":
            return cv2.BORDER_REFLECT
        elif border_type == "BORDER_REFLECT_101":
            return cv2.BORDER_REFLECT_101
        elif border_type == "BORDER_REPLICATE":
            return cv2.BORDER_REPLICATE
        elif border_type == "BORDER_WRAP":
            return cv2.BORDER_WRAP
        else:
            assert f'unsupported border type "{border_type}"'

    def _get_padding(self, img: Image.Image) -> Iterable:
        img_width, img_height = img.size
        return (self.size[0] - img_height, self.size[1] - img_width)

    def _get_padding_tlbr(self, padding: Iterable) -> Iterable:
        top_padding = padding[0] // 2 if self.pad_both_sides else 0
        left_padding = padding[1] // 2 if self.pad_both_sides else 0
        bottom_padding = padding[0] - top_padding
        right_padding = padding[1] - left_padding
        return [top_padding, left_padding, bottom_padding, right_padding]

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image with dimensions (h, w).
        """
        assert isinstance(img, Image.Image), "img should be PIL Image. Got {}".format(
            type(img)
        )
        img_width, img_height = img.size
        if img_height > self.size[0] or img_width > self.size[1]:
            return img
        else:
            return self._pad(img)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(size={0}, border_type={1}, pad_both_sides={2})".format(
                self.size,
                self.border_type,
                repr(self.pad_both_sides),
            )
        )
