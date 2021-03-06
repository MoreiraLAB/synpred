# SynPred
*Full Machine Learning Pipeline for the Synpred prediction and Standalone deployment*

**Prerequisites**
To properly run SynPred without interfering the user should setup an isolate conda environment. Please follow the specifications below.
1. `conda create --name synpred_env python=3.8.2` - There is no problem in changing the environment name, provided the user uses the same name onward, however, the user should use this Python version, as some packages (e.g. Tensorflow) do not have installation support with pip at the timing of writing this tutorial.
2. `conda activate synpred_env` - All the following steps should be performed in the environment. 

Then, the user needs to install several packages.

3. `conda install -c conda-forge rdkit` - Install rdkit for molecular feature extraction with mordred. 
4. `pip install mordred` - Install mordred for feature extraction.
5. `pip install tensorflow==2.3.1` - Install this version of tensorflow.
6. `pip install scikit-learn` - Scikit-learn is required at many steps of the pipeline. 
7. `pip install xgboost`- The package to use the Extreme Gradient Boosting methods needs to be installed separately from scikit-learn.


**Deploy the SynPred pipeline**

1. DEC_variables.py - most of the variables are stored in here (paths, table variables, etc).
2. DEC_support_functions.py - several functions that will be present in more than one script.
3. DEC_edit_dataset.py - start by editing the dataset to generate the full-agreement class and the properly scaled features.
4. DEC_CCLE_filter.py - run this to generate CCLE subsets.
	Only the files in "support/CCLE_log_file.csv" will be called.
	Input and output files at the "datasets" folder.
5. DEC_join_features.py - join the dataset's classes and IDs.
8. DEC_generate_dataset_optimized.py - run to generate dimensionality reduction (PCA) on the CCLE subsets.
	and keep only the cell lines present in the NCI-ALMANAC dataset.
9. SEP_synpred_class.py - Neural network with keras/tensorflow. To be called from the command line or script 6.
10. SEP_gridsearch.py - Run the gridsearch on the SEP_synpred_class.py. Outputs to "evaluation_summary" folder.
11. DEC_synpred_class.py - Same as 5, but for single run.
12. DEC_analysis.py - gather the gridsearch results and identify the best for each metric.

**Standalone Deployment**
