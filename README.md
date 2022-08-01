# cv_bias_amplification

This repository contains code needed to replicate experiments discussed in "[A Systematic Study of Bias Amplification](https://arxiv.org/pdf/2201.11706.pdf)".

## Begin by setting up your environment: 

```
module load anaconda3/2021.05
conda create --name conf_bias_amp python=3.7
conda activate conf_bias_amp
```
Ensure that `torch.cuda.is_available()` is `true`.

Cuda 11.1 isn't strictly necessary, but installing it allows us to get PyTorch 1.9+

```
module load cuda/11.1 
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c nvidia
pip install -r requirements.txt
```
Ready to run!

## Training models + running offline measurements

Scripts needed to kick-off and analyze each experiment discussed in the paper can be found in respective folders in `configs/`. Each experiment directory contains a `scripts/` directory which contains a sript  `generate_experiment_configs.py` that can be executed to create the model configs and `training_measurements*.py` scripts for running offline measurements of key metrics like bias amplification and overconfidence. The `description.txt` file contains a short explanation of the experiment and useful notes for its exectution. The experiment directories should contain an empty `models/` in which configs are stored following execution of `generate_experiment_configs.py`. 

Other directories in the repository (ex: `datasets\`, `losses\`, `models\`) contain infrastructure for actually executing the model training process.

As an example, the following steps can be used to generate the FashionMNIST experiment configs:
* `/my-project-release/my-project/configs/fashionmnist/scripts $ python generate_experiment_configs.py` 

After the training the models with the configs, you can generate results with:
* `/my-project-release/my-project/configs/fashionmnist/scripts $ python training_measurements.py`

Model results are now viewable in `/my-project-release/my-project/configs/fashionmnist/scripts/results_overconf.py`. 

# License
cv_bias_amplification is MIT-licensed, as found in the LICENSE file.
