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

1. Download our [Occlusion Simulator](https://github.com/PFery4/occlusion-simulation) repository, and follow its setup instructions (use the same environment as the one you just set up when going through the previous Environment section).

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

## Stanford Drone Occlusion Dataset

We propose two separate implementations of dataset classes that can be used by the model: one where preprocessing is done on the fly, and one that extracts preprocessed instances from a [HDF5 dataset](https://www.hdfgroup.org/solutions/hdf5/).
Presaving the dataset into a HDF5 file guarantees that random rotation of instances during training remains the same across epochs.
The implementation of our dataset classes can be found under `data/sdd_dataloader.py`. Datasets are configured through config `.yml` files that can be found under `cfg/datasets/`.

### Saving HDF5 dataset files
If you wish to make use of HDF5 based datasets, you must first obtain a copy of a HDF5 dataset file for the particular configuration (and split) of your interest. You can save such a file using `save_hdf5_dataset.py`. A simple call of this script can be done so:
```
save_hdf5_dataset.py --cfg cfg/datasets/<your-desired-config-file.yml> --split <split>
```
Additional flags can be passed to further modify the script's behaviour (those are documented directly inside the script).
The output of this script is a HDF5 file called `dataset_v2.h5`, which is stored by default under `datasets/SDD/pre_saved_datasets/<dataset_id>/<split>/`, where `<dataset_id>` is an identifier for the dataset, which is derived from the configuration file, and `<split>` is the dataset split ('train', 'val' or 'test').

[//]: # (dataset_comparison.py)
[//]: # (visualize_dataset.py)

## Training

[//]: # (train.py)

## Evaluation

[//]: # (save_predictions.py)
[//]: # (save_occlusion_trajectories_information.py)
[//]: # (occlusionformer_eval.py)

[//]: # (performance_analysis/ttest.py)
[//]: # (performance_analysis/performance_summary.py)
[//]: # (performance_analysis/qualitative_example.py)
[//]: # (performance_analysis/occlusion_score_histograms.py)
[//]: # (performance_analysis/boxplots.py)
[//]: # (performance_analysis/prediction_groups_statistics.py)

## Miscellaneous

[//]: # (parameter_count.py)
[//]: # (plot_loss_graph.py)
[//]: # (CHECKSUMS)

