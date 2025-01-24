# OcclusionFormer

This repository is an expansion upon the [AgentFormer](https://github.com/Khrylx/AgentFormer) model, introduced by [Yuan et al](https://arxiv.org/abs/2103.14023).
The focus of this work is to explore the ability of Transformer based prediction models to perform their predictions on agents whose past trajectories are occluded (and therefore, partly missing).

## Installation / Setup

The following sections describe the steps that must be taken in order before you can work with the code.

<details>
   <summary><b><u>Environment</u></b></summary>

The following instructions are only valid for the Linux operating system.

1. Create a conda environment from the [conda-environment.txt](conda-environment.txt) file:
   ```
   conda create --name <environment-name> --file occlusionformer-environment.txt
   ```
   Replace `<environment-name>` with your desired name for the environment.
2. Activate the environment:
   ```
   conda activate <environment-name>
   ```
3. Install the remaining dependencies, which are listed in the [requirements.txt](requirements.txt) file:
   ```
   pip install -r requirements.txt
   ```
- <details>
      <summary><i>For platforms other than Linux:</i></summary>
  
   Setting up from the [conda-environment.yml](conda-environment.yml) file should result in an environment that is equivalent to the Linux installation process.
   *Important Note*: though the specifications inside this file are equivalent to our previous installation instructions, we did *not* perform the following installation procedure on a non-Linux machine:
   ```
   conda env create -f conda-environment.yml
   ```

  </details>

</details>

<details>
   <summary><b><u>Occlusion Simulator</u></b></summary>

This project makes use of our simulator of occlusions, whose implementation can be found [here](https://github.com/PFery4/occlusion-simulation).

1. Download the [Occlusion Simulator repository](https://github.com/PFery4/occlusion-simulation) on your machine (going through that repository's setup instructions is *not* necessary if you successfully set up the environment by following the instructions in the previous section).
2. From the Occlusion Simulator repository's root directory, run the `src/data/save_coord_conv_file.py` script:
   ```commandline
   python src/data/save_coord_conv_file.py
   ```

</details>

<details>
   <summary><b><u>Environment Variables</u></b></summary>

1. Add the root directory of this repository to the `PYTHONPATH` environment variable:
   ```
   export PYTHONPATH=$PWD
   ```
2. Add the root directory of the Occlusion Simulator repository to the `PYTHONPATH` environment variable:
   ```
   export PYTHONPATH="$PYTHONPATH:<path/to/occlusion-simulation>"
   ```
   where `<path/to/occlusion-simulation>` is the path to the Occlusion Simulator repository.

</details>

## Reproducing our results

This section covers the steps necessary for reproducing the results we obtained in [our research](https://repository.tudelft.nl/record/uuid:a168eb7b-fc6b-475f-9280-934d1dbc54cd).

<details>
   <summary><b><u>Downloading Models & Legacy Datasets</u></b></summary>

   Our models and datasets are available in [TU Delft's archive](https://doi.org/10.4121/3dc88884-d8f4-42db-b643-e799fe7fb432).
-  <details>
      <summary><b>(A) Legacy HDF5 dataset files</b></summary>

   The datasets we use are split into 3 separate `.tar.gz` files:
   - `fully_observed.tar.gz`
   - `occlusion_simulation.tar.gz`
   - `occlusion_simulation_imputed.tar.gz`
   
   Before downloading and extracting them, starting from this repository's root directory, execute the following commands:
   ```commandline
   cd datasets/SDD/
   mkdir pre_saved_datasets
   cd pre_saved_datasets/
   ```
   Download the datasets into the `datasets/SDD/pre_saved_datasets/` directory, and extract them with:
   ```commandline
   tar -xvzf fully_observed.tar.gz
   tar -xvzf occlusion_simulation.tar.gz
   tar -xvzf occlusion_simulation_imputed.tar.gz
   ```
   **IMPORTANT NOTE:** If you intend to use our legacy dataset files, you should *systematically pass the `--legacy` option when calling any script that implements it.*
   </details>
-  <details>
      <summary><b>(B) Prediction Model files</b></summary>

   14 individual `.tar.gz` model files are available for download:
   - `agentformer_100.tar.gz`
   - `agentformer_101.tar.gz`
   - `agentformer_102.tar.gz`
   - `agentformer_103.tar.gz`
   - `agentformer_104.tar.gz`
   - `occlusionformer_FO_1.tar.gz`
   - `occlusionformer_FO_2.tar.gz`
   - `occlusionformer_FO_3.tar.gz`
   - `occlusionformer_FO_4.tar.gz`
   - `occlusionformer_FO_5.tar.gz`
   - `occlusionformer_OS.tar.gz`
   - `occlusionformer_DS.tar.gz`
   - `occlusionformer_DS_mapA.tar.gz`
   - `occlusionformer_DS_mapB.tar.gz`

   Each file contains their respective model's phase *I* and *II* checkpoint files, alongside with relevant metadata files.
   Before downloading and extracting them, starting from this repository's root directory, execute the following commands:
   ```commandline
   mkdir results
   cd results
   ```
   Download the datasets into the `results/` directory, and extract them with:
   ```commandline
   tar -xvzf agentformer_100.tar.gz
   tar -xvzf agentformer_101.tar.gz
   tar -xvzf agentformer_102.tar.gz
   tar -xvzf agentformer_103.tar.gz
   tar -xvzf agentformer_104.tar.gz
   tar -xvzf occlusionformer_FO_1.tar.gz
   tar -xvzf occlusionformer_FO_2.tar.gz
   tar -xvzf occlusionformer_FO_3.tar.gz
   tar -xvzf occlusionformer_FO_4.tar.gz
   tar -xvzf occlusionformer_FO_5.tar.gz
   tar -xvzf occlusionformer_OS.tar.gz
   tar -xvzf occlusionformer_DS.tar.gz
   tar -xvzf occlusionformer_DS_mapA.tar.gz
   tar -xvzf occlusionformer_DS_mapB.tar.gz
   ```
   </details>

</details>
<details>
   <summary><b><u>Evaluating our Models</u></b></summary>

   The following steps can be taken to evaluate our models (if you wish to know more about the scripts being run throughout these steps, feel free to consult the **Scripts** section of this README, which discusses their functionalities in more detail):
1. <details>
      <summary>Save the models' predictions against their relevant dataset types:</summary>
   
   ```
   python save_predictions.py --cfg cfg/models/AgentFormer/agentformer_100_II.yml --dataset_cfg cfg/datasets/fully_observed_no_rand_rot.yml --legacy
   python save_predictions.py --cfg cfg/models/AgentFormer/agentformer_101_II.yml --dataset_cfg cfg/datasets/fully_observed_no_rand_rot.yml --legacy
   python save_predictions.py --cfg cfg/models/AgentFormer/agentformer_102_II.yml --dataset_cfg cfg/datasets/fully_observed_no_rand_rot.yml --legacy
   python save_predictions.py --cfg cfg/models/AgentFormer/agentformer_103_II.yml --dataset_cfg cfg/datasets/fully_observed_no_rand_rot.yml --legacy
   python save_predictions.py --cfg cfg/models/AgentFormer/agentformer_104_II.yml --dataset_cfg cfg/datasets/fully_observed_no_rand_rot.yml --legacy

   python save_predictions.py --cfg cfg/models/OcclusionFormer/occlusionformer_FO_1_II.yml --dataset_cfg cfg/datasets/fully_observed_no_rand_rot.yml --legacy
   python save_predictions.py --cfg cfg/models/OcclusionFormer/occlusionformer_FO_2_II.yml --dataset_cfg cfg/datasets/fully_observed_no_rand_rot.yml --legacy
   python save_predictions.py --cfg cfg/models/OcclusionFormer/occlusionformer_FO_3_II.yml --dataset_cfg cfg/datasets/fully_observed_no_rand_rot.yml --legacy
   python save_predictions.py --cfg cfg/models/OcclusionFormer/occlusionformer_FO_4_II.yml --dataset_cfg cfg/datasets/fully_observed_no_rand_rot.yml --legacy
   python save_predictions.py --cfg cfg/models/OcclusionFormer/occlusionformer_FO_5_II.yml --dataset_cfg cfg/datasets/fully_observed_no_rand_rot.yml --legacy

   python save_predictions.py --cfg cfg/models/OcclusionFormer/occlusionformer_FO_1_II.yml --dataset_cfg cfg/datasets/occlusion_simulation_no_rand_rot.yml --legacy
   python save_predictions.py --cfg cfg/models/OcclusionFormer/occlusionformer_FO_2_II.yml --dataset_cfg cfg/datasets/occlusion_simulation_no_rand_rot.yml --legacy
   python save_predictions.py --cfg cfg/models/OcclusionFormer/occlusionformer_FO_3_II.yml --dataset_cfg cfg/datasets/occlusion_simulation_no_rand_rot.yml --legacy
   python save_predictions.py --cfg cfg/models/OcclusionFormer/occlusionformer_FO_4_II.yml --dataset_cfg cfg/datasets/occlusion_simulation_no_rand_rot.yml --legacy
   python save_predictions.py --cfg cfg/models/OcclusionFormer/occlusionformer_FO_5_II.yml --dataset_cfg cfg/datasets/occlusion_simulation_no_rand_rot.yml --legacy

   python save_predictions.py --cfg cfg/models/OcclusionFormer/occlusionformer_OS_II.yml --dataset_cfg cfg/datasets/occlusion_simulation_no_rand_rot.yml --legacy

   python save_predictions.py --cfg cfg/models/OcclusionFormer/occlusionformer_DS_II.yml --dataset_cfg cfg/datasets/occlusion_simulation_no_rand_rot.yml --legacy
   python save_predictions.py --cfg cfg/models/OcclusionFormer/occlusionformer_DS_mapA_II.yml --dataset_cfg cfg/datasets/occlusion_simulation_no_rand_rot.yml --legacy
   python save_predictions.py --cfg cfg/models/OcclusionFormer/occlusionformer_DS_mapB_II.yml --dataset_cfg cfg/datasets/occlusion_simulation_no_rand_rot.yml --legacy

   python save_predictions.py --cfg cfg/models/OcclusionFormer/occlusionformer_FO_1_II.yml --dataset_cfg cfg/datasets/occlusion_simulation_imputed_no_rand_rot.yml --legacy
   python save_predictions.py --cfg cfg/models/OcclusionFormer/occlusionformer_FO_2_II.yml --dataset_cfg cfg/datasets/occlusion_simulation_imputed_no_rand_rot.yml --legacy
   python save_predictions.py --cfg cfg/models/OcclusionFormer/occlusionformer_FO_3_II.yml --dataset_cfg cfg/datasets/occlusion_simulation_imputed_no_rand_rot.yml --legacy
   python save_predictions.py --cfg cfg/models/OcclusionFormer/occlusionformer_FO_4_II.yml --dataset_cfg cfg/datasets/occlusion_simulation_imputed_no_rand_rot.yml --legacy
   python save_predictions.py --cfg cfg/models/OcclusionFormer/occlusionformer_FO_5_II.yml --dataset_cfg cfg/datasets/occlusion_simulation_imputed_no_rand_rot.yml --legacy
   ```
   </details>

2. <details>
      <summary>Evaluate saved predictions against performance metrics:</summary>

   ```
   python model_eval.py --cfg cfg/models/AgentFormer/agentformer_100_II.yml --dataset_cfg cfg/datasets/fully_observed_no_rand_rot.yml --legacy
   python model_eval.py --cfg cfg/models/AgentFormer/agentformer_101_II.yml --dataset_cfg cfg/datasets/fully_observed_no_rand_rot.yml --legacy
   python model_eval.py --cfg cfg/models/AgentFormer/agentformer_102_II.yml --dataset_cfg cfg/datasets/fully_observed_no_rand_rot.yml --legacy
   python model_eval.py --cfg cfg/models/AgentFormer/agentformer_103_II.yml --dataset_cfg cfg/datasets/fully_observed_no_rand_rot.yml --legacy
   python model_eval.py --cfg cfg/models/AgentFormer/agentformer_104_II.yml --dataset_cfg cfg/datasets/fully_observed_no_rand_rot.yml --legacy

   python model_eval.py --cfg cfg/models/OcclusionFormer/occlusionformer_FO_1_II.yml --dataset_cfg cfg/datasets/fully_observed_no_rand_rot.yml --legacy
   python model_eval.py --cfg cfg/models/OcclusionFormer/occlusionformer_FO_2_II.yml --dataset_cfg cfg/datasets/fully_observed_no_rand_rot.yml --legacy
   python model_eval.py --cfg cfg/models/OcclusionFormer/occlusionformer_FO_3_II.yml --dataset_cfg cfg/datasets/fully_observed_no_rand_rot.yml --legacy
   python model_eval.py --cfg cfg/models/OcclusionFormer/occlusionformer_FO_4_II.yml --dataset_cfg cfg/datasets/fully_observed_no_rand_rot.yml --legacy
   python model_eval.py --cfg cfg/models/OcclusionFormer/occlusionformer_FO_5_II.yml --dataset_cfg cfg/datasets/fully_observed_no_rand_rot.yml --legacy

   python model_eval.py --cfg cfg/models/OcclusionFormer/occlusionformer_FO_1_II.yml --dataset_cfg cfg/datasets/occlusion_simulation_no_rand_rot.yml --legacy
   python model_eval.py --cfg cfg/models/OcclusionFormer/occlusionformer_FO_2_II.yml --dataset_cfg cfg/datasets/occlusion_simulation_no_rand_rot.yml --legacy
   python model_eval.py --cfg cfg/models/OcclusionFormer/occlusionformer_FO_3_II.yml --dataset_cfg cfg/datasets/occlusion_simulation_no_rand_rot.yml --legacy
   python model_eval.py --cfg cfg/models/OcclusionFormer/occlusionformer_FO_4_II.yml --dataset_cfg cfg/datasets/occlusion_simulation_no_rand_rot.yml --legacy
   python model_eval.py --cfg cfg/models/OcclusionFormer/occlusionformer_FO_5_II.yml --dataset_cfg cfg/datasets/occlusion_simulation_no_rand_rot.yml --legacy

   python model_eval.py --cfg cfg/models/OcclusionFormer/occlusionformer_OS_II.yml --dataset_cfg cfg/datasets/occlusion_simulation_no_rand_rot.yml --legacy

   python model_eval.py --cfg cfg/models/OcclusionFormer/occlusionformer_DS_II.yml --dataset_cfg cfg/datasets/occlusion_simulation_no_rand_rot.yml --legacy
   python model_eval.py --cfg cfg/models/OcclusionFormer/occlusionformer_DS_mapA_II.yml --dataset_cfg cfg/datasets/occlusion_simulation_no_rand_rot.yml --legacy
   python model_eval.py --cfg cfg/models/OcclusionFormer/occlusionformer_DS_mapB_II.yml --dataset_cfg cfg/datasets/occlusion_simulation_no_rand_rot.yml --legacy

   python model_eval.py --cfg cfg/models/OcclusionFormer/occlusionformer_FO_1_II.yml --dataset_cfg cfg/datasets/occlusion_simulation_imputed_no_rand_rot.yml --legacy
   python model_eval.py --cfg cfg/models/OcclusionFormer/occlusionformer_FO_2_II.yml --dataset_cfg cfg/datasets/occlusion_simulation_imputed_no_rand_rot.yml --legacy
   python model_eval.py --cfg cfg/models/OcclusionFormer/occlusionformer_FO_3_II.yml --dataset_cfg cfg/datasets/occlusion_simulation_imputed_no_rand_rot.yml --legacy
   python model_eval.py --cfg cfg/models/OcclusionFormer/occlusionformer_FO_4_II.yml --dataset_cfg cfg/datasets/occlusion_simulation_imputed_no_rand_rot.yml --legacy
   python model_eval.py --cfg cfg/models/OcclusionFormer/occlusionformer_FO_5_II.yml --dataset_cfg cfg/datasets/occlusion_simulation_imputed_no_rand_rot.yml --legacy
   ```
   </details>

3. <details>
      <summary>Save prerequisite trajectory information before delving into further analysis:</summary>

   ```
   python save_occlusion_trajectories_information.py --cfg cfg/datasets/occlusion_simulation_no_rand_rot.py --split test --legacy
   python save_predictions.py --cfg cfg/models/untrained/CV_predictor.yml --dataset_cfg cfg/datasets/fully_observed_no_rand_rot.yml --legacy
   python save_predictions.py --cfg cfg/models/untrained/CV_predictor.yml --dataset_cfg cfg/datasets/occlusion_simulation_no_rand_rot.yml --legacy
   python save_predictions.py --cfg cfg/models/untrained/CV_predictor.yml --dataset_cfg cfg/datasets/occlusion_simulation_imputed_no_rand_rot.yml --legacy
   python model_eval.py --cfg cfg/models/untrained/CV_predictor.yml --dataset_cfg cfg/datasets/fully_observed_no_rand_rot.yml --legacy
   python model_eval.py --cfg cfg/models/untrained/CV_predictor.yml --dataset_cfg cfg/datasets/occlusion_simulation_no_rand_rot.yml --legacy
   python model_eval.py --cfg cfg/models/untrained/CV_predictor.yml --dataset_cfg cfg/datasets/occlusion_simulation_imputed_no_rand_rot.yml --legacy
   ```
   </details>

4. <details>
      <summary>Produce performance summary tables and boxplot figures:</summary>

   To produce the performance tables:
   ```
   python performance_analysis/performance_summary.py --score_files performance_analysis/performance_scores_to_analyze.txt --filter fully_observed_ids --sort_by dataset_used experiment_name
   python performance_analysis/performance_summary.py --score_files performance_analysis/performance_scores_to_analyze.txt --filter occluded_ids --sort_by dataset_used experiment_name
   python performance_analysis/performance_summary.py --score_files performance_analysis/performance_scores_to_analyze.txt --filter difficult_occluded_ids --sort_by dataset_used experiment_name
   ```
   
   To produce boxplot figures
   ```
   python performance_analysis/boxplots.py --score_files ./results/MODEL_NAME/results/DATASET_ID/MODEL_CHECKPOINT/test/prediction_scores.csv
   ```
   When running this command, make sure to replace `MODEL_NAME`, `DATASET_ID` and `MODEL_CHECKPOINT` with valid names.
   </details>

</details>


## Scripts

This repository contains many scripts that each fulfil a specific role.
The following sections will describe individual scripts' functionality, and how to use them.

<details>
   <summary><b><u>Stanford Drone Occlusion Dataset</u></b></summary>

We propose two separate dataset class implementations that can be used alongside our model (they can be found under `data/sdd_dataloader.py`):
   - `TorchDataGeneratorSDD`: preprocessing is done on the fly
   - `HDF5PresavedDatasetSDD`: preprocessed instances are extracted from an [HDF5 dataset](https://www.hdfgroup.org/solutions/hdf5/)

We recommend that you use HDF5 datasets.
Presaving the dataset into a HDF5 file guarantees that random rotation of instances during training remains the same across epochs.
Datasets are configured through `.yml` files that can be found under `cfg/datasets/`.
</details>

<details>
   <summary><b><u>Saving HDF5 dataset files</u></b></summary>

If you do not wish to use the legacy hdf5 dataset files, you can run the `save_hdf5_dataset.py` script to pre-save your own copies of HDF5 dataset files from an instance of `TorchDataGeneratorSDD`.
This script can be run in the following way:
```
python save_hdf5_dataset.py --cfg cfg/datasets/DATASET_CONFIG_FILE.yml [--split SPLIT] [--start_idx START_INDEX] [--end_idx END_INDEX]
```
The script will generate an HDF5 file, which can be found at: `datasets/SDD/pre_saved_datasets/DATASET_ID/SPLIT/dataset_v2.h5`.
Here, `DATASET_ID` is an identifier derived from the provided configuration file `DATASET_CONFIG_FILE.yml`, and `SPLIT` is the dataset split.
If desired, the saving process will be done over the [`START_INDEX`-`END_INDEX`] range.

-  <details>
      <summary><i>Saving recipe for our legacy HDF5 dataset files</i></summary>
   
   The saving process for our legacy datasets is as follows:
   ```
   python save_hdf5_dataset.py cfg/datasets/fully_observed.yml --split train --start_idx 0 --end_idx 30000 ;
   python save_hdf5_dataset.py cfg/datasets/fully_observed.yml --split train --start_idx 30000 --end_idx 60000 ;
   python save_hdf5_dataset.py cfg/datasets/fully_observed.yml --split train --start_idx 60000 ;
   python save_hdf5_dataset.py cfg/datasets/fully_observed.yml --split val ;
   python save_hdf5_dataset.py cfg/datasets/fully_observed_no_rand_rot.yml --split test;
   python save_hdf5_dataset.py cfg/datasets/occlusion_simulation.yml --split train --start_idx 0 --end_idx 30000 ;
   python save_hdf5_dataset.py cfg/datasets/occlusion_simulation.yml --split train --start_idx 30000 --end_idx 60000 ;
   python save_hdf5_dataset.py cfg/datasets/occlusion_simulation.yml --split train --start_idx 60000 ;
   python save_hdf5_dataset.py cfg/datasets/occlusion_simulation.yml --split val ;
   python save_hdf5_dataset.py cfg/datasets/occlusion_simulation_no_rand_rot.yml --split test ;
   python save_hdf5_dataset.py cfg/datasets/occlusion_simulation_imputed_no_rand_rot.yml --split test ;
   ```
   Executing those commands should give you datasets that are *almost* identical to the legacy datasets we provide for download.
   Those will however not be *exactly* identical, as they will sometimes very slightly deviate.
   The reason for this deviation is that the initial datasets we saved were *not* in HDF5 format, and were very large.
   To facilitate accessibility to our work, we decided to overhaul our storage method.
   The *legacy* hdf5 dataset files perfectly replicate the data stored in our initial, inefficient storage approach.
   Some difference in floating point rounding between our initial storing method and our current one is most likely the cause for the slight deviation.
 
   </details>
</details>

<details>
   <summary><b><u>Prediction Models</u></b></summary>

The `model` directory contains code that is relevant to the implementation of our prediction models.
Models are configured through `.yml` files located under `cfg/models/`.
Our work follows that of Yuan et al. who conduct their training process in two separate phases.
Models being trained in phase ***I*** and ***II*** are very different from one another architecture-wise.
Those differences can be directly seen in the model configuration files.

Notably, it is important to remark that phase ***II*** models are always related to a corresponding phase ***I*** model.
This is specified in phase ***II*** config files by two fields:
   - `pred_cfg` indicates the name of the phase ***I*** model being used
   - `pred_checkpoint_name` indicates the name of the checkpoint file containing the weights to initialize that phase ***I*** model

Before you can train and/or evaluate any model, you must first create (or use an already existing) model configuration `.yml` file.
The configuration files of the models we produced throughout our research can be found under `cfg/models/`.

</details>

<details>
   <summary><b><u>Training Models</u></b></summary>

Training a model can be done with the `train.py` script:
```
python train.py --cfg cfg/models/PATH-TO-MODEL_CONFIG_FILE.yml [--checkpoint_name CHECKPOINT_NAME]
```
All the parameters relevant to the models' training regime are found inside its `.yml` config file.
Here, the option `CHECKPOINT_NAME` can be used to continue a previously interrupted training session from a specific point.
`CHECKPOINT_NAME` corresponds to the name of a model checkpoint file inside the model's directory, under `results/MODEL_CONFIG_FILE/models/`.
</details>

<details>
   <summary><b><u>Evaluation</u></b></summary>

Evaluating models' performance is done in multiple steps.

1. <details>
      <summary>Running the model on a dataset split, and saving its predictions:</summary>
   
   ```
   python save_predictions.py --cfg cfg/models/PATH-TO-MODEL_CONFIG_FILE.yml --dataset_cfg cfg/datasets/DATASET_CONFIG_FILE.yml [--data_split SPLIT] [--checkpoint_name CHECKPOINT_NAME]
   ```
   The script will store individual predictions as pickle files under `results/MODEL_CONFIG_FILE/results/DATASET_ID/CHECKPOINT_NAME/SPLIT/`.
   </details>
2. <details>
      <summary>Evaluating predictions against performance metrics:</summary>
   
   ```
   python model_eval.py --cfg cfg/models/PATH-TO-MODEL_CONFIG_FILE.yml --dataset_cfg cfg/datasets/DATASET_CONFIG_FILE.yml [--data_split SPLIT] [--checkpoint_name CHECKPOINT_NAME]
   ```
   Two files will be created under `results/MODEL_CONFIG_FILE/results/DATASET_ID/CHECKPOINT_NAME/SPLIT/`: 
   - `prediction_scores.csv` contains an extensive report of every performance metric measured over every prediction made.
   - `prediction_scores.yml` contains a performance summary of metrics aggregated over the entire test set.

   </details>
3. <details>
      <summary>Prerequisites for further performance analysis:</summary>

   Some of our performance analysis scripts require some information about trajectories, which must first be generated in the following way:
   ```
   python save_occlusion_trajectories_information.py --cfg cfg/datasets/occlusion_simulation_no_rand_rot.py --split test
   ```
   This will save some information about trajectories (e.g. distance travelled by agents, occlusion pattern...) into the following file: `datasets/SDD/pre_saved_datasets/occlusion_simulation/test/trajectories_info.csv`

   Additionally, some scripts require information about the performance of a regular Constant Velocity predictor as well. It is therefore important to save and evaluate the CV predictor:
   ```
   python save_predictions.py --cfg cfg/models/untrained/CV_predictor.yml --dataset_cfg cfg/datasets/fully_observed_no_rand_rot.yml
   python save_predictions.py --cfg cfg/models/untrained/CV_predictor.yml --dataset_cfg cfg/datasets/occlusion_simulation_no_rand_rot.yml
   python save_predictions.py --cfg cfg/models/untrained/CV_predictor.yml --dataset_cfg cfg/datasets/occlusion_simulation_imputed_no_rand_rot.yml
   python model_eval.py --cfg cfg/models/untrained/CV_predictor.yml --dataset_cfg cfg/datasets/fully_observed_no_rand_rot.yml
   python model_eval.py --cfg cfg/models/untrained/CV_predictor.yml --dataset_cfg cfg/datasets/occlusion_simulation_no_rand_rot.yml
   python model_eval.py --cfg cfg/models/untrained/CV_predictor.yml --dataset_cfg cfg/datasets/occlusion_simulation_imputed_no_rand_rot.yml
   ```
   </details>
4. <details>
      <summary>Running performance analysis scripts:</summary>
   
   The scripts inside the `performance_analysis/` directory allow for further analysis of the models' performance.

   `performance_summary.py` produces a performance summary table using multiple models' `prediction_scores.csv` files:
   ```
   python performance_analysis/performance_summary.py [--score_files [FILES.txt | FILE_1 FILE_2 ...]] [--filter FILTER]
   ```
   the `--score_files` argument is either a text file containing paths to multiple `prediction_scores.csv` files (such as [this file](performance_analysis/performance_scores_to_analyze.txt)), or a sequence of multiple `prediction_scores.csv` files.
   `--filter` is an option that allows the user to aggregate performance scores over certain subsets of the dataset.
   
   `boxplots.py` displays boxplots of performance metrics by grouping trajectories by their last observed timestep:
   ```
   python performance_analysis/boxplots.py [--score_files [FILES.txt | FILE_1 FILE_2 ...]]
   ```

   `qualitative_example.py` visualizes model predictions qualitatively:
   ```
   python performance_analysis/qualitative_example.py --cfg cfg/models/PATH-TO-MODEL_CONFIG_FILE.yml --dataset_cfg cfg/datasets/DATASET_CONFIG_FILE.yml --instance_num INSTANCE_INDEX [--ids ID_1 ID_2 ...]
   ```
   Here, `--cfg` and `--dataset_cfg` indicate the model and dataset being used.
   The argument `--instance_num` is used to select a particular dataset instance from the dataset.
   The option `--ids` can be used to filter agent identities present in the instance, if you wish to display the predictions and future ground truth trajectories for a subset of agents only.

   </details>
</details>

<details>
   <summary><b><u>Miscellaneous</u></b></summary>

<details>
   <summary><b>Loss graphs</b></summary>

The script `plot_loss_graph.py` can be used to display individual model's training and validation loss values:
```
python plot_loss_graph.py --cfg cfg/models/PATH-TO-MODEL_CONFIG_FILE.yml --split [train | val]
```

</details>

<details>
   <summary><b>Parameter counter</b></summary>

The script `parameter_count.py` prints a summary of a model's weights:
```
python parameter_count.py --cfg cfg/models/PATH-TO-MODEL_CONFIG_FILE.yml
```

</details>

<details>
   <summary><b>Visualizing Dataset instances</b></summary>

Individual dataset instances can be visualized with the `visualize_dataset.py` script:
```
python visualize_dataset.py --cfg cfg/datasets/DATASET_CONFIG_FILE.yml [--split SPLIT] [--idx IDX_A IDX_B ...] --show
```
The script will generate image representations of agent's trajectories for the specified dataset instances `IDX_A`, `IDX_B`, etc.
</details>

<details>
   <summary><b>Verifying Dataset equivalence</b></summary>

The script `dataset_comparison.py` can be used to verify that both our dataset implementations produce identical data.
It can be run in the following way:
```
python dataset_comparison.py --cfg cfg/datasets/DATASET_CONFIG_FILE.yml [--split SPLIT] [--start_idx START_INDEX] [--end_idx END_INDEX] [--save_path REPORT_FILE.csv]
```
The script will run through two dataset instances (one `TorchDataGeneratorSDD` and one `HDF5PresavedDatasetSDD`), compare their produced data, and report the comparisons into a `.csv` file.

</details>

</details>
