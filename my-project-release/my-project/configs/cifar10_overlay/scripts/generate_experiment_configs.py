# Copyright (c) Meta Platforms, Inc. and affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import numpy as np
import classy_vision.generic.util as util
import random
import pandas as pd
import os

CONFIG_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

CLASS_ASSIGNMENTS = [{'task': [5, 9], 'attr': [2, 4]},
                    {'task': [8, 7], 'attr': [3, 2]},
                    {'task': [7, 1], 'attr': [5, 2]},
                    {'task': [4, 0], 'attr': [9, 6]},
                    {'task': [2, 4], 'attr': [6, 3]},
                    {'task': [6, 8], 'attr': [7, 4]},
                    {'task': [8, 5], 'attr': [4, 1]},
                    {'task': [3, 4], 'attr': [5, 6]},
                    {'task': [1, 8], 'attr': [0, 2]},
                    {'task': [3, 5], 'attr': [2, 6]},
                    {'task': [5, 9], 'attr': [3, 4]},
                    {'task': [3, 7], 'attr': [8, 1]},
                    {'task': [0, 6], 'attr': [8, 1]},
                    {'task': [3, 1], 'attr': [0, 4]},
                    {'task': [6, 7], 'attr': [2, 5]},
                    {'task': [6, 9], 'attr': [2, 0]},
                    {'task': [5, 3], 'attr': [6, 7]},
                    {'task': [9, 2], 'attr': [1, 8]},
                    {'task': [3, 8], 'attr': [9, 2]},
                    {'task': [8, 0], 'attr': [4, 5]}]
COUNTER = 1000                                       # used to name the experiments and associated config
EPSILONS = range(20, 50, 10)             
ETAS = range(0, 101, 10)
EXPERIMENT_NAME = "cifar10_overlay"                           # used to store experiment configs and results
IDS_TEST = "./test_ids_cifar10.txt"                   # used to designate file of sample ids for each test image
IDS_TRAIN = "./train_ids_cifar10.txt"                 # used to designate the file of sample ids for each train image  
INVERSION_CONFIGS = [{                               # used to designate which classes should be inverted 
                        "type": "fixed",             # used to designate how inversion probabilities should be calculated
                        "assign": "custom",        # should we use a "custom" class assignment with "custom_selection", or "shuffle" randomly with a fixed "num_classes"
                        "custom_selection": [0],
                        "num_inversion_classes": None,
                        "seed": None
                    }
                    ]
JSON_BASE = None                                    # used to calculate base json when JSON type is custom
JSON_TYPE = "template"                              # used to determine which base json to build the config from
JSON_TEMPLATES = [                                  # used to designate template config
                    "config_template_cifar10_resnet110_gpu1_lrmultistep.json"
                 ]   
NUM_MODELS = 1                                      # number of models per epsilon-class assignment combination             
NUM_SAMPLES_TEST = 2_000
NUM_SAMPLES_TRAIN = 10_000
SEED_BASE = 0
SEED_TEST = 100_000                                 # seed, or "None" to indicate get_test_seed() should be used
TRANSFORMS = {                                      # used to designate which transforms to apply or change
                "assign_class_str": None
            }       



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


def get_train_seed(model_i):
    return ((model_i + SEED_BASE + 2) * 100_000)


def get_test_seed(model_i):
    return SEED_TEST if SEED_TEST else model_i


def get_base_json(json_type, template, counter=None):
    if json_type == "template":
        return template
    if json_type == "custom":
        return '../configs/' + str(EXPERIMENT_NAME) + '/config_' + str(EXPERIMENT_NAME) + "_bias_test_" + str(counter + JSON_BASE) + '.json'

config_paths = []
output_directories = []
names = []
epsilons = []
etas = []
train_seeds = []
test_seeds = []
json_templates = []
class_assignment_mappings = []
probabilities = []
n_classes= []
depths = []
counter_ids = []

for template in JSON_TEMPLATES:
    for class_assignment in CLASS_ASSIGNMENTS:
        for epsilon in EPSILONS: 
            for eta in ETAS:
                for inversion_config in INVERSION_CONFIGS:
                    num_classes = len(set(class_assignment))
                    p = get_p(epsilon, num_classes, inversion_config)

                    for model_i in range(0, NUM_MODELS):
                        with open(get_base_json(JSON_TYPE, template, COUNTER)) as f:
                            data = json.load(f)

                        train_seed = get_train_seed(CLASS_ASSIGNMENTS.index(class_assignment))
                        test_seed = get_test_seed(model_i)
                        for i in range(len(data['dataset']['train']['transforms'])):
                            if "assign_class_str" in TRANSFORMS.keys() and data['dataset']['train']['transforms'][i]['name'] == 'assign_class_str':
                                data['dataset']['train']['transforms'][i]['classes'] = {class_assignment["task"][0]: 0, class_assignment["task"][1]: 1}

                        for i in range(len(data['dataset']['test']['transforms'])):       
                            if "assign_class_str" in TRANSFORMS.keys() and data['dataset']['test']['transforms'][i]['name'] == 'assign_class_str':
                                data['dataset']['test']['transforms'][i]['classes'] = {class_assignment["task"][0]: 0, class_assignment["task"][1]: 1}

                        data['model']['num_classes'] = num_classes

                        data['dataset']['train']['num_samples'] = 10_000
                        data['dataset']['train']['task_classes'] = class_assignment["task"]
                        data['dataset']['train']['attr_classes'] = class_assignment["attr"]
                        data['dataset']['train']['eta'] = eta
                        data['dataset']['train']['epsilon'] = epsilon
                        data['dataset']['train']['seed'] = train_seed

                        data['dataset']['test']['num_samples'] = 2_000
                        data['dataset']['test']['task_classes'] = class_assignment["task"]
                        data['dataset']['test']['attr_classes'] = class_assignment["attr"]
                        data['dataset']['test']['eta'] = eta
                        data['dataset']['test']['epsilon'] = epsilon
                        data['dataset']['test']['seed'] = test_seed

                        filename = 'config_' + EXPERIMENT_NAME + '_bias_test_' + str(COUNTER) +'.json'
                        model_folder_path = CONFIG_PATH + '/models/'
                        if (not os.path.exists(CONFIG_PATH)): 
                            raise Exception(CONFIG_PATH + " not a valid config path")
                        if (not os.path.exists(model_folder_path)):
                            os.mkdir(model_folder_path)
                        config_path = model_folder_path + filename                 
                        with open(config_path, 'w') as out:
                            json.dump(data, out, indent=4)
                        COUNTER += 1


                        config_paths.append(config_path)
                        output_directories.append(EXPERIMENT_NAME)
                        names.append(filename)
                        epsilons.append(epsilon)
                        etas.append(eta)
                        train_seeds.append(train_seed)
                        test_seeds.append(test_seed)
                        json_templates.append(template)
                        class_assignment_mappings.append(CLASS_ASSIGNMENTS.index(class_assignment))
                        probabilities.append(str(p).replace(',', ''))
                        n_classes.append(num_classes)
                        counter_ids.append(COUNTER)


data = {            
        "config_path": config_paths,
        "output_dir": output_directories,
        "name": names,
        "epsilon": epsilons,
        "eta": etas,
        "train_seed": train_seeds,
        "test_seed": test_seeds,
        "json_template": json_templates,
        "class_assignment_mapping": class_assignment_mappings,
        "probabilities": probabilities,
        "n_classes": n_classes,
        "counter_id": counter_ids
        }
df = pd.DataFrame.from_dict(data)
df.to_csv('../../' + EXPERIMENT_NAME + '/scripts/manifest.txt', index=False)