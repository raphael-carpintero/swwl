# Sliced Wasserstein Weisfeiler Lehman Graph kernel

## Description
This repository contains the experiments of the paper:

Carpintero Perez, R.; Da Veiga, S.; Garnier, J.; Staber, S. (2024) Gaussian process regression with Sliced Wasserstein Weisfeiler-Lehman graph kernels.    

### Abstract
Supervised learning has recently garnered significant attention in the field of computational physics due to its ability to effectively extract complex patterns for tasks like solving partial differential equations, or predicting material properties. Traditionally, such datasets consist of inputs given as meshes with a large number of nodes representing the problem geometry (seen as graphs), and corresponding outputs obtained with a numerical solver. This means the supervised learning model must be able to handle large and sparse graphs with continuous node attributes. In this work, we focus on Gaussian process regression, for which we introduce the Sliced Wasserstein Weisfeiler-Lehman (SWWL) graph kernel. In contrast to existing graph kernels, the proposed SWWL kernel enjoys positive definiteness and a drastic complexity reduction, which makes it possible to process datasets that were previously impossible to handle. The new kernel is first validated on graph classification for molecular datasets, where the input graphs have a few tens of nodes. The efficiency of the SWWL kernel is then illustrated on graph regression in computational fluid dynamics and solid mechanics, where the input graphs are made up of tens of thousands of nodes.

### External resources
The implementation of fused Gromov Wasserstein (FGW) as well as the Wasserstein distance come from the POT library https://pythonot.github.io/ .
Propagation and graph hopper kernels are tested from the Grakel package https://pypi.org/project/GraKeL/ .

We leverage pytorch geometric implemenation of continuous Weisfeiler-Lehman iterations and dataloaders https://pytorch-geometric.readthedocs.io/en/latest/ .

Gaussian process regression uses a variant of the RobustGaSP R package https://github.com/cran/RobustGaSP .

## Requirements
All the packages used are listed in the `env_swwl.yml` file. If you use conda, you can install the environment by running

```bash
conda env create -f env_swwl.yml
```

and then activate the environment with

```bash
conda activate env_swwl.yml
```

Remark that if you plan to reproduce the experiments with the sgml kernel, you need to install extra packages, namely `tensorflow>=2.5`, `dppy>=0.3.2` and `python-igraph>=0.9.6` ( `pytorch` is not required to run sgml).

## Usage
Two tasks are possible in the `experiments` folder: 
- Support vector classification (SVC) of TU benchmark datasets made of small graphs,
- Gaussian process regression (GPR) of large meshes.

The list of commands used to reproduce all the results of the paper are given in the `experiments/scripts`.

### Installation

As a developer, you should install this repository in editable mode as follows:

```bash
pip install -e .
```

If this is not possible because you don't own the python/conda environment you're using, then simply add the repository to your PYTHONPATH:

```bash
export PYTHONPATH=$PYTHONPATH:/PATH_TO_THIS_REPOSITORY/
```

### Installation of the custom RGaSP package
The R code only requires the `devtools`, `reticulate`, `yaml` libraries and our custom `RGaSP` library modified from https://github.com/cran/RobustGaSP . We used R.4.2.3 during our experiments.

To install the library, use the following command in your R environment:
```
devtools::build(pkg="RobustGaSP")
install.packages("RobustGaSP_0.6.5.tar.gz", repos = NULL, type="source")
```

#### Configuration file

The files in the experiments all use the configuration file `experiments/config.yml` that allows you to modify global configuration parameters (optimization settings, backup options, file paths, etc.). These parameters may remain unchanged, but the paths to the datasets must be changed by the user.

Importantly, if the option 'matrices:save' and 'matrices:load_if_exists' are both True in the configuration file, distance and Gram matrices are computed a first time and they can be reloaded at any time. 

### Datasets
Classification datasets can be downloaded at https://chrsmrrs.github.io/datasets/docs/datasets/ and loaded using `experiments/datasets_classification.py`. If you want to use Grakel graph kernels (propagation ou graph hopper), place the .zip file in a specific folder (see the root_TUDatasets_zip option of `experiments/config.yml`). For other methods, we recommand to download the datasets in another folder using pytorch_geometric (see the root_TUDatasets option of `experiments/config.yml`) as the loading functions depend on the kernel.

Regression datasets are loaded thanks to `experiments/datasets_regression.py`. `Tensile2d`, `Rotor37` and `AirfRANS` datasets must be downloaded from https://plaid-lib.readthedocs.io/en/latest/source/data_challenges.html and then unzipped using (for instance) the following command in the right folder: 
```bash
tar –xvzf Tensile2d.tar.gz –C FOLDER
```
Note that the public version of `Rotor37` is different from the one we used in our experiments due to a different representation of finite elements. This could produce minor variations of the scores that we present in the paper. `Rotor37_CM `, `Tensile2d_CM` and  `AirfRANS_CM` are not available in this version (we only show the signatures of the loading functions).

### Classification of TU datasets
Use `experiments/run_classification.py` to compute distance/Gram matrices, train and test a classifier on TU datasets.

```bash
python run_classification.py --dataset=BZR --kernel=swwl --classifier=svc -H 3 -Q=20 -P=10 --seed 0 --verbose=1
```

See `scripts/script_classification.sh` for the complete list of commands.

### Gaussian process regression of meshes

In order to have a fair comparison with propagation kernel that cannot be used with the python wrapper of RGaSP, Gaussian process regression is done with our custom RGaSP package. This supposes 2 steps: 1- computing distance/Gram matrices. 2- regression in R.
You can always perform regression without resorting to R by using step 1A with the argument --out=x where w needs to be replaced by 2 for `Rotor37` and  `Rotor37_CM ` (isentropic efficiency),  0 for  `Tensile2d ` and  `Tensile2d_CM ` (maximum Von Mises), and 0 for  `AirfRANS ` and  `AirfRANS_CM ` (drag coefficient).

#### Step 1A: Computing distance/Gram matrices
Use `example/run_regression.py` to compute the distance matrices of SWWL or the Gram matrices of Propag.

```bash
python run_regression.py --dataset=Rotor37 --kernel=swwl --regressor=rgasp -H 3 -Q=500 -P=50 -T=1 --seeds 0 --verbose=1;
```

See `scripts/script_regression.sh` for the complete list of commands.

#### Step 1B: Computing distance matrices (special case of WWL) 
You need to run the 3 commands in sequence: `experiments/wwl_meshes_parallelized_step1.py`, `experiments/wwl_meshes_parallelized_step2.py`, `experiments/wwl_meshes_parallelized_step3.py`. The first step computes the continuous WL iterations. The second step computes the distance matrix line by line. You need to launch it separately for lines 0, ..., 99. Step 3 assembles the distance matrix.

```bash
python wwl_meshes_parallelized_step1.py --dataset Rotor37_CM -H=3 -T=1
python run_wl_no_parallelized2.py --dataset Rotor37_CM -H=3 -T=1 --line 0 # 1,2,...,99
python wwl_meshes_parallelized_step3.py --dataset Rotor37_CM -H=3 -T=1
```

See `scripts/script_regression_wwl.sh` for the complete list of commands.

#### Step 2: Robust Gaussian process regression in R
Once step 1A or 1B are finished and matrices are saved, just use `experiments/run_regression.R` to run the Gaussian process regression using RGaSP in R.

```bash
R CMD BATCH "--args Rotor37 swwl 3 1 50 500" run_regression.R ./scripts/logs/o/regression_R/resRotor37CM_swwl_JOBID.txt
```

See `scripts/script_regression_R.sh` for the complete list of commands.

### Figures

Figures and tables of the paper were made using the `experiments/create_score_tables_and_figures.py`.