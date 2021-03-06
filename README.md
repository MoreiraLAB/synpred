# SynPred
*Full Machine Learning Pipeline for the Synpred prediction and Standalone deployment*

**Prerequisites**

To properly run SynPred without interfering the user should setup an isolate conda environment. Please follow the specifications below.
1. `conda create --name synpred_env python=3.8.2` - There is no problem in changing the environment name, provided the user uses the same name onward, however, the user should use this Python version, as some packages (e.g. Tensorflow) do not have installation support with pip at the timing of writing this tutorial.
2. `conda activate synpred_env` - All the following steps should be performed in the environment. 

Then, the user needs to install several packages.

3. `conda install -c conda-forge r-base` - This package is only required if you are trying to deploy the SynPred pipeline from scratch, and not for the standalone version.
4. Only for the full SynPred deployment: open R (after step 3) on the command line/terminal:
	- `if (!requireNamespace("BiocManager", quietly = TRUE))
    	install.packages("BiocManager")`
    - `BiocManager::install("synergyfinder")`
    - `install.packages("devtools")` - This package might or might not be necessary depending on whether the previous installations were successful, however, the `pillar`package might issue an error down the line, and installing it with `devtools` solved it in this case.
    - `devtools::install_github("r-lib/pillar")`
    - `library(synergyfinder)` - To check if the package was properly installed.
3. `conda install -c conda-forge rdkit` - Install rdkit for molecular feature extraction with mordred. 
4. `pip install mordred` - Install mordred for feature extraction.
5. `pip install tensorflow==2.3.1` - Install this version of tensorflow.
6. `pip install scikit-learn` - Scikit-learn is required at many steps of the pipeline. 
7. `pip install xgboost`- The package to use the Extreme Gradient Boosting methods needs to be installed separately from scikit-learn.

**Changes**

After downloading/cloning this repository, there are still some changes to be made.
1. At `synpred_variables.py`, change the variable `DEFAULT_LOCATION` to the location where you will be running your files
2. If the folders were not automatically downloaded, create the folders, on the same location as the scripts, with the following names:
	- CCLE
	- datasets
	- evaluation_non_DL
	- evaluation_summary
	- molecules
	- redeployment_variables
	- results
	- saved_model
	- support
	- train_log

**Deploy the SynPred pipeline**

1. `synpred_variables.py` - most of the variables are stored in here (paths, table variables, etc).
2. `synpred_support_functions.py` - several functions that will be present in more than one script.
3. `syn_find_synergy.R` - Use R with `synergyfinder` package to attain the classification with the different metrics.
4. `synpred_edit_dataset.py` - start by editing the dataset to generate the full-agreement class and the properly scaled features.
5. `synpred_CCLE_filter.py` - run this to generate CCLE subsets.
	Only the files in "support/CCLE_log_file.csv" will be called.
	Input and output files at the "datasets" folder.
6. `synpred_join_features.py` - join the dataset's classes and IDs.
7. `synpred_generate_dataset_optimized.py` - run to generate dimensionality reduction (PCA) on the CCLE subsets.
	and keep only the cell lines present in the NCI-ALMANAC dataset.
8. `synpred_synpred_class.py` - Neural network with keras/tensorflow. To be called from the command line or script 6.
9. `synpred_gridsearch.py` - Run the gridsearch on the SEP_synpred_class.py. Outputs to "evaluation_summary" folder.
10. `synpred_synpred_class.py` - Same as 5, but for single run.
11. `synpred_analysis.py` - Gather the gridsearch results and identify the best for each metric.

**Standalone Deployment**
