#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional as F
from classy_vision.losses import ClassyLoss, register_loss


@register_loss("one_hot_binary_ce_loss")
class OneHotBinaryCELoss(ClassyLoss):
    def forward(self, input, target):
        labels = F.one_hot(target, num_classes=10).float()
        return F.binary_cross_entropy(input, labels)

    @classmethod
    def from_config(cls, config):
        # We don't need anything from the config
        return cls()