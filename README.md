# SynPred
*Full Machine Learning Pipeline for the Synpred prediction and Standalone deployment*

**Deploy the SynPred pipeline**

Prerequisites:
1. conda environment: source activate black

Scripts:

0 - DEC_variables.py - most of the variables are stored in here (paths, table variables, etc)
0.5 - DEC_support_functions.py - several functions that will be present in more than one script
1 - DEC_edit_dataset.py - start by editing the dataset to generate the full-agreement class and the properly scaled features
2 - DEC_CCLE_filter.py - run this to generate CCLE subsets.
	Only the files in "support/CCLE_log_file.csv" will be called.
	Input and output files at the "datasets" folder.

3 - DEC_join_features.py - join the dataset's classes and IDs
3.1 - DEC_custom_molecules.py - generate custom molecules' features
3.2 - DEC_molecules_features.py - join the custom molecules' features with the rest of the dataset 
4 - DEC_generate_dataset_optimized.py - run to generate dimensionality reduction (PCA) on the CCLE subsets
	and keep only the cell lines present in the NCI-ALMANAC dataset
5 - SEP_synpred_class.py - Neural network with keras/tensorflow. To be called from the command line or script 6.
6 - SEP_gridsearch.py - Run the gridsearch on the SEP_synpred_class.py. Outputs to "evaluation_summary" folder.
7 - DEC_synpred_cass.py - Same as 5, but for single run.
8 - DEC_analysis.py - gather the gridsearch results and identify the best for each metric

**Standalone Deployment**
