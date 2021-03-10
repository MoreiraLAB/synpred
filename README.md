# SynPred
*Full Machine Learning Pipeline for the Synpred prediction and Stand-alone deployment*

**Prerequisites**

To properly run SynPred without interfering the user should setup an isolate conda environment. Please follow the specifications below.
1. `conda create --name synpred_env python=3.8.2` - There is no problem in changing the environment name, provided the user uses the same name onward, however, the user should use this Python version, as some packages (e.g. Tensorflow) do not have installation support with pip at the timing of writing this tutorial.
2. `conda activate synpred_env` - All the following steps should be performed in the environment. 

Then, the user needs to install several packages:

3. `conda install -c conda-forge r-base=3.6.3` - Using R is only required if you are trying to deploy the SynPred pipeline from scratch, and not for the stand-alone version. If you are going to deploy the full pipeline, make sure to install this specific version of R, as others might have compatibility issues on the `synergyfinder` package.
4. Only for the full SynPred deployment: open R (after step 3) on the command line/terminal:
	- `if (!requireNamespace("BiocManager", quietly = TRUE))
    	install.packages("BiocManager")`
    - `BiocManager::install("synergyfinder")`
    - `library(synergyfinder)` - Check if the package was properly installed.
    - `q()` - Exit the R shell.
3. `conda install -c conda-forge rdkit` - Install rdkit for molecular feature extraction with mordred. 
4. `pip install mordred` - Install mordred for feature extraction.
5. `pip install pubchempy` - Install pubchempy to download SMILE from name.
6. `pip install tensorflow==2.3.1` - Install this version of tensorflow.
7. `pip install scikit-learn` - Scikit-learn is required at many steps of the pipeline. 
8. `pip install xgboost`- The package to use the Extreme Gradient Boosting methods needs to be installed separately from scikit-learn.

**Stand-alone Deployment**

For the stand-alone development, the user will require the files in the folder `CCLE_processed`, which contains the CCLE features as processed in the paper. Furthermore, the user requires the pretrained models. These are too large for GitHub, as such, the user must download them at `http://www.moreiralab.com/resources/synpred`, at the `Stand-alone deployment section` and store them in a folder in the same location as the scripts, with the name `standalone_models`. Finally, the user must have performed environment setup and needs to run the standa-lone inside the activated environment.
Regarding the scripts, the only change required should be in the `standalone_variables.py` script, in which the user should change the `HOME` variable into the folder path on his/her computer.

Finally, the user must create (if not already cloned) a folder with the name `standalone_results` at the same folder of the scripts. 

The way to run the SynPred stand-alone deployment is, after initializing the conda environment typing:

`python standalone_synpred.py your_table_name.csv`

The user can then check the features and results calculated at the `standalone_results` folder. The user is advised that running the script with different input tables will result in overwriting of the predictions. To check an example of the input file, consult the `standalone_example.csv`, in the folder of the scripts. The names of the columns should be the same, as well as the content. 

**Changes for pipeline replication**

After downloading/cloning this repository, there are still some changes to be made if you wish to replicate the full pipeline of SynPred.
1. At `synpred_variables.py`, change the variable `DEFAULT_LOCATION` to the location where you will be running your files
2. If the folders were not automatically downloaded, the user needs to create the folders, on the same location as the scripts, with the following names:
	- CCLE
	- datasets
	- evaluation_summary
	- molecules
	- redeployment_variables
	- resources
	- saved_model
	- support
	- train_log
3. Some of the files required are not available on this page because their are either too large, or were developed by a third party. Particularly, the CCLE subsets, can be downloaded at CCLE website, and should go on the CCLE folder. There are 7 files required, please check file `support/CCLE_log_file.csv` to see which files are required for the full SynPred deployment. This files should have the same names as indicated in column `File_name` of the log file and go into `CCLE` folder. Furthermore, the user is also required to have the `NCI_ALMANAC` dataset at the location and name `datasets/NCI_ALMANAC.csv`. Finally, the user should change the `datasets/example.csv` file to the file with the combinations, with the same format and columns as advised.

**Deploy the SynPred pipeline**

After performing the changes previously pointed and properly installing and setting up the environment, these scripts should simply run without requiring changes.
1. `synpred_variables.py` - Most of the variables are stored in here (paths, table variables, etc).
2. `synpred_support_functions.py` - Several functions that will be present in more than one script.
3. `synpred_find_synergy.R` - Use R with `synergyfinder` package to attain the classification with the different metrics.
4. `synpred_retrieve_smiles.py` - Use `pubchempy` to download the smile into the `molecules` folder.
5. `synpred_construct_split_dataset.py` - Split the dataset while joining the drug features.
6. `synpred_edit_dataset.py` - Start by editing the dataset to generate the full-agreement class and the properly scaled features.
7. `synpred_CCLE_filter.py` - Run this to generate CCLE subsets.
	Only the files in "support/CCLE_log_file.csv" will be called.
	Input and output files at the "datasets" folder.
8. `synpred_join_features.py` - Join the dataset's classes and IDs.
9. `synpred_generate_dataset_optimized.py` - Run to generate dimensionality reduction (PCA) on the CCLE subsets.
	and keep only the cell lines present in the NCI-ALMANAC dataset.
10. `synpred_keras.py` - Neural network with keras/tensorflow. To be called from the command line or script 11.
11. `synpred_gridsearch_keras.py` - Run the gridsearch on the `synpred_keras.py`. Outputs to "evaluation_summary" folder. This makes use of only 100% of the dataset.
12. `synpred_keras_final.py` - Neural network with keras/tensorflow after gridsearch. To be called from the command line or script 13.
13. `synpred_best_keras_final.py` - Run and save the best keras models.
14. `synpred_ML_gridsearch.py` - Run the gridsearch for the ML methods that do not involve Keras on 10% of the training set.
15. `synpred_ML.py` - Run the best ML methods that do not involve Keras on the full training set, with the best parameters from script 14. Save the best models. 
16. `synpred_ensemble_gridsearch.py` - Test several ensemble methods, namely, several Keras neural network.
17. `synpred_ensemble_final.py` - Run the best ensemble model with Keras.

**Please cite:**

António J. Preto, Pedro Matos-Filipe, Joana Mourão and Irina S. Moreira

*SynPred: Prediction of Drug Combination Effects in Cancer using Full-Agreement Synergy Metrics and Deep Learning*