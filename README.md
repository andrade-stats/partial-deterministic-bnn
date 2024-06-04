# On the Effectiveness of Partially Deterministic Bayesian Neural Networks

Implementation and source code for all (partial deterministic) Bayesian Neural Networks and experiments as described in [On the Effectiveness of Partially Deterministic Bayesian Neural Networks], 2024 (under review).

## Requirements

- Python == 3.10
- arviz >= 0.17.1
- OptBNN (as provided here)

## Preparation

1. Create experiment environment using e.g. conda as follows
```bash
conda create -n sghmc python=3.10
conda activate sghmc
```

2. Install OptBNN (as provided here)
```bash
pip install .
```

3. Install arviz and other necessary packages
```bash
pip install -U arviz natsort tqdm seaborn scikit-learn scipy==1.12
```

4. Prepare Datasets
Split your dataset into train (train_X.csv, train_y.csv), 
validation (val_X.csv, val_y.csv) and test (test_X.csv, test_y.csv), saved in "./{data,gap_data}/DATA_NAME/FOLD_ID/" where
DATA_NAME is the name of the dataset, and FOLD_ID is the fold id (from 1 to 10).
For example:
```bash
python code/create_dataset_standard.py --dataset=yacht --test_size=0.1
```
Afterwards use "create_dataset_gap.py" to create the gap-splits


## Usage - Basic Example Workflow for MAP/SGHMC

This is an example that shows training and evaluation using the maximum a-posterior (MAP) or SGMHC method.

1. Run Training using MAP for the yacht dataset (train) of all folds in parallel:
```bash
python ./code/map.py --dataset yacht --id=-1
```

For using SGHMC use "sghmc.py" instead of "map.py" in the same way as above, i.e.
```bash
python ./code/sghmc.py --dataset yacht --id=-1
```

All result are saved in "./results/METHOD_NAME/DATA_NAME/cv/."

## Usage - Basic Example Workflow for Training Partially Determininstic BNNs

1. Train with MAP 
```bash
python ./code/map.py --dataset yacht --id=-1
```

2. Run sampling on partially deterministic BNN


(A) Train with Pivots Selected by Row  (pivots-per-row, here denoted as abs_fix)
```bash
python ./code/sghmc.py --dataset yacht --method abs_fix --id=-1
```

(B) Train with Row Subset (row-subset, here denoted as row_fix)
```bash
python ./code/sghmc.py --dataset yacht --method row_fix --id=-1
```

(C) Train with Maximal Elements (pivots-free, here denoted as sharma_fix)
```bash
python ./code/sghmc.py --dataset yacht --method sharma_fix --id=-1
```

(D) Train with Whole Deterministic Layers (layer-fix, here denoted as layer_fix)
```bash
python ./code/sghmc.py --dataset yacht --method layer_fix --id=-1
```

By default one hidden layer is only used. For two or more hidden layers use the "n_hidden" argument.
For classification use classification_map.py (instead of map.py), and classification_sghmc.py (instead of sghmc.py). 

## Show results 

Create plot showing average improvement of SGHMC variations over MAP (in terms of NLL using CV for all methods)
Use "plot_performance.py" for create comparison plots.
Use "show_all_results.py" for creating all tables.


## Remarks

Note that the code in optbnn is a slight modification of the original [source code](https://github.com/tranbahien/you-need-a-good-prior) of 
"All you need is a good functional prior for Bayesian Deep Learning", Tran, Ba-Hien, et al., JMLR, 2022 
which is redistributed here with permission from the authors.