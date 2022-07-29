# Run from within the /scripts folder.

import json
import numpy as np
import pandas as pd
import classy_vision.generic.util as util
import random
import os 

CONFIG_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

CLASS_ASSIGNMENTS = [
    [1, 1, 0, 1, 0, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 1, 1, 0, 1, 0, 0],
    [0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    [1, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 1],
    [1, 1, 0, 0, 0, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 1, 1, 0],
    [1, 1, 0, 0, 0, 1, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
    [0, 1, 0, 0, 1, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 1, 0, 1],
    [0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 1, 0, 0, 1, 1],
    [0, 0, 1, 0, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0, 1, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 1, 1, 1, 1, 0]
]
COUNTER = 1000                                       # used to name the experiments and associated config
EPSILONS = range(0, 55, 5)             
EXPERIMENT_NAME = "fashionmnist"                           # used to store experiment configs and results
IDS_TEST = "./test_ids.txt"                   # used to designate file of sample ids for each test image
IDS_TRAIN = "./train_ids.txt"                 # used to designate the file of sample ids for each train image  
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
                    "config_template.json"
                 ]   
NUM_MODELS = 1                                      # number of models per epsilon-class assignment combination             
NUM_SAMPLES_TEST = 10_000
NUM_SAMPLES_TRAIN = 60_000
SEED_BASE = 0
SEED_TEST = 100_000                                 # seed, or "None" to indicate get_test_seed() should be used
TRANSFORMS = {                                      # used to designate which transforms to apply or change
                "assign_class": None, 
                "invert_exact": None
            }       


def get_p(epsilon, num_classes, inversion_config):
    p = []

    if inversion_config['type'] == "fixed":
        class_list = []
        if inversion_config['assign'] == "custom":
            class_list = inversion_config['custom_selection']
        for i in range(num_classes):
            if i in class_list:
                p.append(np.round(0.5 + (epsilon * 0.01), 2))
            else:
                p.append(np.round(0.5 - (epsilon * 0.01), 2))


    return p


def get_train_seed(model_i):
    return ((model_i + SEED_BASE + 2) * 100_000)


def get_test_seed(model_i):
    return SEED_TEST if SEED_TEST else model_i


def get_base_json(json_type, template, counter=None):
    if json_type == "template":
        return template

config_paths = []
output_directories = []
names = []
epsilons = []
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
            for inversion_config in INVERSION_CONFIGS:
                num_classes = len(set(class_assignment))
                p = get_p(epsilon, num_classes, inversion_config)

                for model_i in range(0, NUM_MODELS):
                    with open(get_base_json(JSON_TYPE, template, COUNTER)) as f:
                        data = json.load(f)

                    invert_exact_index_train = -1
                    invert_exact_index_test = -1
                    for i in range(len(data['dataset']['train']['transforms'])):
                        if "invert_exact" in TRANSFORMS.keys() and data['dataset']['train']['transforms'][i]['name'] == 'invert_exact':
                            invert_original = []
                            assert NUM_SAMPLES_TRAIN % len(p) == 0, "Dataset not evenly divisible by number of classes."
                            class_size = NUM_SAMPLES_TRAIN / len(class_assignment) 
                            train_seed = get_train_seed(CLASS_ASSIGNMENTS.index(class_assignment))
                            with util.torch_seed(train_seed):
                                with util.numpy_seed(train_seed):
                                    for c in class_assignment:
                                        hold = [1] * (int)(np.round(class_size * p[c], 0)) + [0] * (int)(np.round(class_size * (1-p[c]), 0))
                                        np.random.shuffle(hold)
                                        invert_original.extend(hold)

                            with open(IDS_TRAIN, 'r') as f:
                                train_ids = json.load(f)

                            invert_mapped = [0] * NUM_SAMPLES_TRAIN
                            for k in range(0, NUM_SAMPLES_TRAIN):
                                invert_mapped[k] = invert_original[int((train_ids[str(k)]['class'] * class_size) + train_ids[str(k)]['index'])]

                            data['dataset']['train']['transforms'][i]['invert'] = invert_mapped

                        if data['dataset']['train']['transforms'][i]['name'] == 'invert_exact':
                            invert_exact_index_train = i
                        if "assign_class" in TRANSFORMS.keys() and data['dataset']['train']['transforms'][i]['name'] == 'assign_class':
                            data['dataset']['train']['transforms'][i]['classes'] = class_assignment

                    for i in range(len(data['dataset']['test']['transforms'])):
                        if "invert_exact" in TRANSFORMS.keys() and data['dataset']['test']['transforms'][i]['name'] == 'invert_exact':
                            invert_original = []
                            assert NUM_SAMPLES_TEST % len(p) == 0, "Dataset not evenly divisible by number of classes."
                            class_size = NUM_SAMPLES_TEST / len(class_assignment)
                            test_seed = get_test_seed(model_i) 
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

                            data['dataset']['test']['transforms'][i]['invert'] = invert_mapped

                        if data['dataset']['test']['transforms'][i]['name'] == 'invert_exact':
                            invert_exact_index_test = i
                        if "assign_class" in TRANSFORMS.keys() and data['dataset']['test']['transforms'][i]['name'] == 'assign_class':
                            data['dataset']['test']['transforms'][i]['classes'] = class_assignment

                    data['model']['num_classes'] = num_classes

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
