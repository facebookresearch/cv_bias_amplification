# Copyright (c) Meta Platforms, Inc. and affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import error
import torch
import torch.utils.data

import ast
import itertools
import json
import numpy as np
import pandas as pd
import random
import sys

from classy_vision.dataset import ClassyDataset, build_dataset, register_dataset
from classy_vision.generic.util import load_checkpoint
import classy_vision.generic.util as util
from classy_vision.models import ClassyModel
from PIL import Image

sys.path.append('../../..')
from datasets.inversion_transforms import AssignClass, Invert

root = './'

#### TODO: Change according to your directory structure.
CHECKPOINT_PATH = ""
####

EXPERIMENT_NAME = "cifar100_trainingsize"                       # used to store experiment configs and results
NBINS = 15                                      # used to designate the number of bins for overconfidence measures
NUM_CHECKPOINTS = None
NUM_SAMPLES_TEST = 10_000                   
TRANSFORMS ={                                   # used to designate which transforms to apply or change
             "invert_exact": {},                
             "assign_class": {}, 
             "normalize": {
                            "mean": [0.5071, 0.4867, 0.4408],
                            "std": [0.2675, 0.2565, 0.2761]
                           }
            }       
TEST_DATASET = "cifar100_test"
IDS_TEST = "./test_ids_cifar100.txt"               # used to designate file of sample ids for each test image
VISUALIZE = False                               # designate if plots should be shown
CLASS_ASSIGNMENTS = [
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0],
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0],
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0],
    [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    [1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1],
    [0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
    [0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1],
    [0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
    [0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
    [0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
    [1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
    [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
]

sm = torch.nn.Softmax()

def get_model_results(model, dataset, attributes, tasks):
    """Get model results needed for directional bias amplification measurements.

    Args:
        dataset (ClassyDataset): ClassyDataset, generally of test data
        attributes (List): list of distinct attributes
        tasks (List): list of distinct tasks
        
    Returns:
        results (List x List): nested list of size attributes x tasks with number of instances with that
            input attribute and predicted task
    """
    
    predictions = []
    attributes_in = []
    targets = []
    percents = []

    model.eval() 
    
    for k in dataset.iterator():
        attributes_in.append(k['attribute'])
        targets.append(k['target'])
        result = model(k['input']).data
        predictions.append(result.numpy().argmax(axis=1))
        percents.append(sm(result))
        
    flat_attributes = [item for sublist in attributes_in for item in sublist]
    flat_targets = [item for sublist in targets for item in sublist]
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_percents = [float(max(item)) for sublist in percents for item in sublist]

    results_in = np.zeros((len(attributes), len(tasks)))
    results_pred = np.zeros((len(attributes), len(tasks)))
    results_correct = np.zeros((len(attributes), len(tasks)))
    
    total = 0
    correct = 0
    for a, t, p in zip(flat_attributes, flat_targets, flat_predictions):
        results_in[attributes[a]][t] = results_in[attributes[a]][t] + 1
        results_pred[attributes[a]][p] = results_pred[attributes[a]][p] + 1
        results_correct[attributes[a]][t] += 1 if t == p else 0

        total += 1
        correct = correct + 1 if t == p else correct

    # TESTING return flat_attributes, flat_predictions, flat_targets
    return results_in, results_pred, correct * 1.0 / total, np.array(flat_predictions), np.array(flat_targets), np.array(flat_percents), results_correct / results_in


def format_results(results, attributes, tasks):
    """Format results for readability.

    Args:
        results (List): nested list of size attributes x tasks with number of instances with that
            attribute and task
        attributes (List): list of distinct attributes
        tasks (List): list of distinct tasks
        
    Returns:
        dict<(attribute, task), value> : dict with number of instances with that
            attribute and task
    """
    return {key: results[attributes[key[0]]][key[1]] for key in list(itertools.product(attributes, tasks))}


def calc_bias_amp_at(res, res_h, attributes, tasks):
    """Perform directional bias amplification a->t as defined in https://arxiv.org/pdf/2102.12594.pdf.

    1. Instantiate `results`, which will store the values defined in the paper's 
    summation expression.
    Looping over all attributes and tasks...
        2. Generate probabilities needed for y_at, delta_at calculations defined in the paper. 
            p_attr = P(Attribute_a = 1)
            p_attr_h = P(Attribute_a-hat = 1)
            p_task = P(Task_t = 1)
            p_task_h = P(Task_t-hat = 1)
            p_attr_task = P(Attribute_a = 1, Task_t = 1)
            p_attr_h_task_h = P(Attribute_a-hat = 1, Task_t = 1)
            p_task_h_cond_attr_h = P(Task_t-hat = 1 | Attribute_a-hat = 1)
            p_task_cond_attr = P(Task_t = 1 | Attribute_a = 1)
        3. Calculate y_at, delta_at, and expression inside of summation, and save to `results`.
    4. Perform summation and BiasAmp_at calculation.
    
    Args:
        res (List x List): nested list of size attributes x tasks with number of instances that have
            attribute and task, generally the training input or test input
        resh (List x List): nested list of size attributes x tasks with number of instances that have
            attribute and task, generally the test output
        attributes (List): list of distinct attributes
        tasks (List): list of distinct tasks
        
    Returns:
        results (dict<(attribute, task), value>): dict with value inside equation summation with the
            attribute and task
        bias_amp_at (float): directional bias amplification a->t metric
    """

    results = {key: 0 for key in list(itertools.product(attributes, tasks))}

    for key, value in results.items():
        attr = key[0]
        task = key[1]    
        
        p_attr = np.sum(res[attributes[attr]]) / np.sum(np.matrix(res))
        p_attr_h = np.sum(res_h[attributes[attr]]) / np.sum(np.matrix(res_h))
        p_task = np.sum(res, axis=0)[task] / np.sum(np.matrix(res))
        p_task_h = np.sum(res_h, axis=0)[task] / np.sum(np.matrix(res_h))

        p_attr_task = res[attributes[attr]][task] / np.sum(np.matrix(res))
        p_attr_h_task_h = res_h[attributes[attr]][task] / np.sum(np.matrix(res_h))
        p_task_h_cond_attr_h = p_attr_h_task_h / p_attr_h
        p_task_cond_attr = p_attr_task / p_attr

        y_at = p_attr_task > (p_attr * p_task)
        delta_at = p_task_h_cond_attr_h - p_task_cond_attr
        print(str(key)+".... y_at: " + str(y_at) + ", delta_at: " + str(delta_at))
        
        results[key] = (y_at * delta_at) + ((1 - y_at) * (-1 * delta_at))

    bias_amp_at = (1 / (len(attributes) * len(tasks))) * sum(results.values())

    return results, bias_amp_at

def get_binned_metrics(percents, predictions, targets, nbins):
    acc_bins = []
    conf_bins = []
    count_bins = []
    assert 0 not in percents

    for i in range(0, nbins):
        filter = np.where((percents > (i)/nbins) & (percents <= (i+1)/nbins))
        perc = percents[filter]
        pred = predictions[filter]
        targ = targets[filter]

        acc = sum(pred==targ)/len(pred) if len(pred) != 0 else np.nan
        conf = sum(perc)/len(perc) if len(perc) != 0 else np.nan
        
        acc_bins.append(acc)
        conf_bins.append(conf)
        count_bins.append(len(pred))
    
    return acc_bins, conf_bins, count_bins


def get_ece(acc_bins, conf_bins, count_bins, nbins):
    ece = 0
    for i in range(0, nbins):
        ece += (count_bins[i] / sum(count_bins)) * abs(acc_bins[i] - conf_bins[i]) if acc_bins[i] is not np.nan else 0
        
    return ece


def get_mce(acc_bins, conf_bins, nbins):
    mce = 0.0
    for i in range(0, nbins):
        mce = np.maximum(mce, abs(acc_bins[i] - conf_bins[i])) if acc_bins[i] is not np.nan else mce
        
    return mce


def get_p(epsilon, num_classes, inversion_config):
    p = []

    if inversion_config['type'] == "fixed":
        class_list = []
        if inversion_config['assign'] == "custom":
            class_list = inversion_config['custom_selection']
        elif inversion_config['assign'] == "shuffle":
            random.seed(inversion_config['seed'])
            class_list = random.sample(range(num_classes), inversion_config['num_inversion_classes'])
        for i in range(num_classes):
            if i in class_list:
                p.append(np.round(0.5 + (epsilon * 0.01), 2))
            else:
                p.append(np.round(0.5 - (epsilon * 0.01), 2))


    elif inversion_config['type'] == "evenly_spaced":
        for i in range(num_classes):
            min = np.round(0.5 - (epsilon * 0.01), 2)
            p.append(np.round(min + (i * (2 * epsilon * 0.01) / (num_classes - 1)), 2))

    return p

def run_measurements(name, json_template, class_assignment, epsilon, num_classes, p, train_seed, test_seed, checkpoint_dir, d_s, checkpoint):
    print("\nRunning \"" + checkpoint_dir + "\" .")

    transforms = []
    if "assign_class" in TRANSFORMS: 
        transforms.append({"name": "assign_class", "classes": class_assignment})
    if "invert" in TRANSFORMS:
        transforms.append({"name": "invert", "p": list(p), "seed": SEED_TEST})
    if "invert_exact" in TRANSFORMS:
        invert_original = []
        assert NUM_SAMPLES_TEST % len(p) == 0, "Dataset not evenly divisible by number of classes."
        class_size = NUM_SAMPLES_TEST / len(class_assignment) 
        with util.torch_seed(test_seed):
            with util.numpy_seed(test_seed):
                for c in class_assignment:
                    hold = [1] * (int)(np.round(class_size * p[c], 0)) + [0] * (int)(np.round(class_size * (1-p[c]), 0))
                    np.random.shuffle(hold)
                    invert_original.extend(hold)

        with open(IDS_TEST, 'r') as f:
            test_ids = json.load(f)

        invert_mapped = [0] * NUM_SAMPLES_TEST
        for k in range(0, NUM_SAMPLES_TEST):
            invert_mapped[k] = invert_original[int((test_ids[str(k)]['class'] * class_size) + test_ids[str(k)]['index'])]
        
        transforms.append({"name": "invert_exact", "invert": invert_mapped})

    if "swap_binary_task" in TRANSFORMS:
        transforms.append({"name": "swap_binary_task"})
    if "swap_task_attr" in TRANSFORMS:
        transforms.append({"name": "swap_task_attr"})

    key_transforms = [{"name": "ToTensor"}]
    if "normalize" in TRANSFORMS:
        key_transforms.append({
            "name": "Normalize",
            "mean": TRANSFORMS['normalize']['mean'],
            "std": TRANSFORMS['normalize']['std'],
        })

    transforms.extend([
                {"name": "tuple_to_map", "list_of_map_keys": ["input", "original_target", "sample_id", "target", "attribute"]},
                {
                    "name": "apply_transform_to_key", 
                        "transforms": key_transforms,
                        "key": "input"
                }
                ])

    test_dataset_config = {
        "name": TEST_DATASET,
        "batchsize_per_replica": 128,
        "shuffle": True,
        "transforms": transforms,
        "num_workers": 8
    }
    my_dataset_test = build_dataset(test_dataset_config)
    checkpoint_data = load_checkpoint(checkpoint_dir)
    model = ClassyModel.from_checkpoint(checkpoint_data)


    attributes = {"a": 0, "b": 1}
    tasks = [i for i in range(num_classes)]

    test_results_in, test_results_pred, acc, predictions, targets, percents, accuracy_breakdown = get_model_results(model, my_dataset_test, attributes, tasks)

    accuracy_breakdown = format_results(accuracy_breakdown, attributes, tasks)
    accuracy_breakdown = {str(k): v for k, v in accuracy_breakdown.items()}

    biasamp_breakdown, biasamp = calc_bias_amp_at(test_results_in, test_results_pred, attributes, tasks)
    biasamp_breakdown = {str(k): v for k, v in biasamp_breakdown.items()}
    acc_bins, conf_bins, count_bins = get_binned_metrics(percents, predictions, targets, NBINS)

    print(acc)
    print(format_results(test_results_in, attributes, tasks))
    print(format_results(test_results_pred, attributes, tasks))
    print(biasamp_breakdown)

    names.append(name)
    epsilons.append(epsilon)
    train_seeds.append(train_seed)
    test_seeds.append(test_seed)
    checkpoints.append(checkpoint)
    accuracies.append(acc)
    biasamps.append(biasamp)
    test_predictions.append(format_results(test_results_pred, attributes, tasks))
    acc_bins_list.append(acc_bins)
    conf_bins_list.append(conf_bins)
    count_bins_list.append(count_bins)
    eces.append(get_ece(acc_bins, conf_bins, count_bins, NBINS))
    mces.append(get_mce(acc_bins, conf_bins, NBINS))
    accuracy_breakdowns.append(accuracy_breakdown)
    biasamp_breakdowns.append(biasamp_breakdown)
    json_templates.append(json_template)
    dataset_sizes.append(d_s)

# Set up and run measurements

names = []
epsilons = []
train_seeds = []
test_seeds = []
checkpoints = []
accuracies = []
biasamps = []
test_predictions = []
acc_bins_list = []
count_bins_list = []
conf_bins_list = []
eces = []
mces = []
accuracy_breakdowns = []
biasamp_breakdowns = []
json_templates = []
dataset_sizes = []

try:
    original_results = pd.read_csv('../../' + EXPERIMENT_NAME + '/results_overconf.csv', index_col=False)
except:
    original_results = None


manifest_df = pd.read_csv('../../' + EXPERIMENT_NAME + '/scripts/manifest.txt')


for _, row in manifest_df.iterrows():
    json_template = row['json_template']
    class_assignment = CLASS_ASSIGNMENTS[row['class_assignment_mapping']]
    epsilon = row['epsilon']
    num_classes = row['n_classes']
    p = [float(x) for x in row['probabilities'][1:-1].split(' ')]
    d_s = row['dataset_size']
    train_seed = row['train_seed']
    test_seed = row['test_seed']
    counter = row['counter_id']
    name = row['name']


    checkpoint_dir = CHECKPOINT_PATH + name[:-5] + '/checkpoints/'
    run_measurements(   
                        name = name,
                        json_template = json_template,
                        class_assignment = class_assignment, 
                        epsilon = epsilon, 
                        num_classes = num_classes, 
                        p = p, 
                        train_seed = train_seed,
                        test_seed = test_seed, 
                        checkpoint_dir = checkpoint_dir, 
                        d_s = d_s,
                        checkpoint = None
                    )

    if counter % 10 == 0:
        data = {"name": names, 
                "epsilon": epsilons,
                "train_seed": train_seeds,
                "test_seed": test_seeds,
                "checkpoint_number": checkpoints,
                "acc@1": accuracies, 
                "biasamp": biasamps,
                "bins": NBINS,
                "acc_bins": acc_bins_list,
                "conf_bins": conf_bins_list,
                "count_bins": count_bins_list,
                "ece": eces,
                "mce": mces,
                "accuracy_breakdown": accuracy_breakdowns,
                "biasamp_breakdown": biasamp_breakdowns,
                "json_templates": json_templates,
                "dataset_size": dataset_sizes
                }
        df = pd.DataFrame.from_dict(data)

        if original_results is not None:
            combined_df = pd.concat([original_results, df])
            combined_df.to_csv('../../' + EXPERIMENT_NAME + '/results_overconf.csv', index=False)
        else:
            df.to_csv('../../' + EXPERIMENT_NAME + '/results_overconf.csv', index=False)

data = {"name": names, 
        "epsilon": epsilons,
        "train_seed": train_seeds,
        "test_seed": test_seeds,
        "checkpoint_number": checkpoints,
        "acc@1": accuracies, 
        "biasamp": biasamps,
        "bins": NBINS,
        "acc_bins": acc_bins_list,
        "conf_bins": conf_bins_list,
        "count_bins": count_bins_list,
        "ece": eces,
        "mce": mces,
        "accuracy_breakdown": accuracy_breakdowns,
        "biasamp_breakdown": biasamp_breakdowns,
        "json_templates": json_templates,
        "dataset_size": dataset_sizes
        }
df = pd.DataFrame.from_dict(data)

if original_results is not None:
    combined_df = pd.concat([original_results, df])
    combined_df.to_csv('../../' + EXPERIMENT_NAME + '/results_overconf.csv', index=False)
else:
    df.to_csv('../../' + EXPERIMENT_NAME + '/results_overconf.csv', index=False)