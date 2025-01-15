# OcclusionFormer

This repository is an expansion upon the [AgentFormer](https://github.com/Khrylx/AgentFormer) model, introduced by [Yuan et al](https://arxiv.org/abs/2103.14023).
The focus of this work is to explore the ability of Transformer based prediction models to perform their predictions on agents whose past trajectories are occluded (and therefore, partly missing).

## Installation / Setup

### Environment

*We manage our environment through Anaconda, and we recommend that you do so too: the project relies on the [scikit-geometry](https://github.com/scikit-geometry/scikit-geometry) library, which is only directly accessible from the conda-forge channel (otherwise, it can be built from source, with [CGAL 5.0](https://www.cgal.org/) installed).
Though installation through other methods might be possible, only the following instructions have been verified to work properly.*

1. Create the environment:
   ```
   conda create -n <environment-name> python=3.8 pip
   ```
   Replace `<environment-name>` with your desired name for the environment.
2. Activate the environment:
   ```
   conda activate <environment-name>
   ```
3. Install [PyTorch 1.8.0](https://pytorch.org/get-started/previous-versions/#v180) with the appropriate CUDA version.
4. Install scikit-geometry:
   ```
   conda install -c conda-forge scikit-geometry
   ```
5. Install the remaining dependencies:
   ```
   pip install -r requirements.txt
   ```

### Occlusion Simulator

In order to study occlusions and their effect on trajectory prediction, we apply a simulator of occlusions on top of the Stanford Drone Dataset.

1. Download our [Occlusion Simulator](https://github.com/PFery4/occlusion-simulation) repository, and follow its setup instructions (use the same environment as the one you set up when going through the previous Environment section).

### Environment Variables

1. Add the root directory of this repository to the `PYTHONPATH` environment variable:
   ```
   export PYTHONPATH=$PWD
   ```
2. Add the root directory of the Occlusion Simulator repository to the `PYTHONPATH` environment variable:
   ```
   export PYTHONPATH="$PYTHONPATH:<path/to/occlusion-simulation>"
   ```
   where `<path/to/occlusion-simulation>` is the path to the Occlusion Simulator repository.

## Scripts

This repository contains many scripts that each fulfil a specific role.
Scripts are located in two places: the most important ones are in the repository's root directory, and scripts related to the analysis of model performance can be found inside the `performance_analysis` directory.
The following sections will briefly describe individual scripts' functionality, and how to use them in the most basic way.

*It is important to note that some optional script flags will not be discussed in this README, and we invite you to read the source code directly, in order to learn in what ways you can modify script's behaviour to your preference.*

### Stanford Drone Occlusion Dataset

We propose two separate dataset class implementations that can be used alongside our model (they can be found under `data/sdd_dataloader.py`):
   - `TorchDataGeneratorSDD`: preprocessing is done on the fly
   - `HDF5PresavedDatasetSDD`: preprocessed instances are extracted from an [HDF5 dataset](https://www.hdfgroup.org/solutions/hdf5/)

Presaving the dataset into a HDF5 file guarantees that random rotation of instances during training remains the same across epochs.
Datasets are configured through `.yml` files that can be found under `cfg/datasets/`.

### Obtaining HDF5 dataset files
If you wish to make use of `HDF5PresavedDatasetSDD`, you must first obtain a copy of a HDF5 dataset file for the particular configuration (and split) of your interest. You can save such a file using `save_hdf5_dataset.py`. This script can be run in the following way:
```
python save_hdf5_dataset.py --cfg cfg/datasets/DATASET_CONFIG_FILE.yml [--split SPLIT]
```
The script will generate a HDF5 file, which can be found at: `datasets/SDD/pre_saved_datasets/DATASET_ID/SPLIT/dataset_v2.h5`.
Here, `DATASET_ID` is an identifier derived from the provided configuration file `DATASET_CONFIG_FILE.yml`, and `SPLIT` is the dataset split.

[//]: # (TODO: explain legacy datasets, how they can be retrieved, and how they can be used)

### Verifying Dataset equivalence
The script `dataset_comparison.py` can be used to verify that both our dataset implementations produce identical data.
It can be run in the following way:
```
python dataset_comparison.py --cfg cfg/datasets/DATASET_CONFIG_FILE.yml [--split SPLIT] [--save_path REPORT_FILE.csv]
```
The script will run through two dataset instances (one `TorchDataGeneratorSDD` and one `HDF5PresavedDatasetSDD`), compare their produced data, and report the comparisons into a `.csv` file.

### Visualizing Dataset instances
Individual dataset instances can be visualized by means of the `visualize_dataset.py` script:
```
python visualize_dataset.py --cfg cfg/datasets/DATASET_CONFIG_FILE.yml [--split SPLIT] [--idx IDX_A IDX_B ...] --show
```
The script will generate image representations of agent's trajectories for the specified dataset instances `IDX_A`, `IDX_B`, etc.

### Prediction Models
The `model` directory contains code that is relevant to the implementation of our prediction models.
Models are configured through `.yml` files located under `cfg/models`.
Our work follows that of Yuan et al. who conduct their training process in two separate phases.
Models being trained in phase ***I*** and ***II*** are very different from one another architecture-wise.
Those differences can be directly seen in the model configuration files.

Notably, it is important to remark that phase ***II*** models are always related to a corresponding phase ***I*** model.
This is specified in phase ***II*** config files by two fields:
   - `pred_cfg` indicates the name of the phase ***I*** model being used
   - `pred_checkpoint_name` indicates the name of the checkpoint file containing the weights to initialize that phase ***I*** model

*When training / evaluating any model, always make sure that it is correctly configured*.

### Training

Training a model can be done by using the `train.py` script:

```
python train.py --cfg cfg/models/PATH-TO-MODEL_CONFIG_FILE.yml [--checkpoint_name CHECKPOINT_NAME]
```
All the parameters relevant to the models' training regime are found inside its config file.
Here, the option `CHECKPOINT_NAME` can be used to continue a previously interrupted training session from a specific point.
`CHECKPOINT_NAME` corresponds to the name of a model checkpoint file inside the model's directory, inside `results/MODEL_CONFIG_FILE/models/`.

### Evaluation

Evaluating models' performance is done in multiple steps.
First, the model is being run on a dataset split, and its predictions are saved:
```
python save_predictions.py --cfg cfg/models/PATH-TO-MODEL_CONFIG_FILE.yml --dataset_cfg cfg/datasets/DATASET_CONFIG_FILE.yml [--data_split SPLIT] [--checkpoint_name CHECKPOINT_NAME]
```
The script will store individual predictions as pickle files under `results/MODEL_CONFIG_FILE/results/DATASET_ID/CHECKPOINT_NAME/SPLIT/`.

Once those predictions have been saved, they can be used to evaluate the model's performance against different metrics by running:
```
python model_eval.py --cfg cfg/models/PATH-TO-MODEL_CONFIG_FILE.yml --dataset_cfg cfg/datasets/DATASET_CONFIG_FILE.yml [--data_split SPLIT] [--checkpoint_name CHECKPOINT_NAME]
```
Two files will be created under `results/MODEL_CONFIG_FILE/results/DATASET_ID/CHECKPOINT_NAME/SPLIT/`: 
- `prediction_scores.csv` contains an exhaustive performance report of every prediction made.
- `prediction_scores.yml` contains a performance summary of metrics aggregated over the entire test set.

Performance analysis scripts can be used to inspect the performance of our models in more detail.
Some of those scripts require some information about trajectories, which must first be generated in the following way:
```
python save_occlusion_trajectories_information.py --cfg cfg/datasets/occlusion_simulation_no_rand_rot.py --split test
```
This will save some information about trajectories (e.g. distance travelled by agents, occlusion pattern...) into the following file: `datasets/SDD/pre_saved_datasets/occlusion_simulation/test/trajectories_info.csv`
Additionally, some scripts require information about the performance of a regular Constant Velocity predictor as well. It is therefore important to save and evaluate the CV predictor:
```
python save_predictions.py --cfg cfg/models/untrained/CV_predictor.yml --dataset_cfg cfg/datasets/fully_observed_no_rand_rot.yml ;
python save_predictions.py --cfg cfg/models/untrained/CV_predictor.yml --dataset_cfg cfg/datasets/occlusion_simulation_no_rand_rot.yml ;
python save_predictions.py --cfg cfg/models/untrained/CV_predictor.yml --dataset_cfg cfg/datasets/occlusion_simulation_imputed_no_rand_rot.yml ;
python model_eval.py --cfg cfg/models/untrained/CV_predictor.yml --dataset_cfg cfg/datasets/fully_observed_no_rand_rot.yml ;
python model_eval.py --cfg cfg/models/untrained/CV_predictor.yml --dataset_cfg cfg/datasets/occlusion_simulation_no_rand_rot.yml ;
python model_eval.py --cfg cfg/models/untrained/CV_predictor.yml --dataset_cfg cfg/datasets/occlusion_simulation_imputed_no_rand_rot.yml ;
```

Once those prerequisite steps have been taken, the scripts inside the `performance_analysis/` directory can be run.
The three most important ones are the following:
- `performance_summary.py` produces a performance summary table using multiple models' `prediction_scores.csv` files.
- `boxplots.py` displays boxplots of performance metrics by grouping trajectories by their last observed timestep.
- `qualitative_example.py` visualizes model predictions qualitatively.

### Miscellaneous

[//]: # (parameter_count.py)
[//]: # (plot_loss_graph.py)
[//]: # (CHECKSUMS)
[//]: # (EXAMPLES)
[//]: # (OBTAINING OUR RESULTS)

[//]: # (# TODO: describe downloading from TU Delft repository)
[//]: # (TODO: INDICATE SOMEWHERE THAT WE DO NO RAND ROT IN TEST SPLITS ONLY)