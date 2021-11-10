###
#Author: A.J.Preto & P.Matos-Filipe
#Group: Data-Driven Molecular Design
#Group Leader: Irina S. Moreira
#Project: SynPred
###

library(synergyfinder)
library(tidyverse)
library(dplyr)

setwd(getwd())
csv_files = list.files(pattern = "*.csv")
for (current_file in csv_files){
  input_table <- read.csv(current_file, sep = ",", header = TRUE)
  reshape_table <- ReshapeData(input_table, data_type = "viability", seed = 42, impute_method = "cart")
  calculated_synergy <- CalculateSynergy(reshape_table, seed = 42)
  
  synergy_scores_combo <- calculated_synergy$synergy_scores
  output_name_combo <- paste("combo_calculated_class_",current_file, sep = "")
  synergy_scores_combo <- synergy_scores_combo[,c("block_id","conc1","conc2","ZIP_synergy","HSA_synergy","Bliss_synergy","Loewe_synergy")]
  colnames(synergy_scores_combo) <- c("block_id","conc1","conc2","ZIP","HSA","Bliss","Loewe")
  write.csv(synergy_scores_combo,output_name_combo, row.names = FALSE)
  
  synergy_scores_pairs <- calculated_synergy$drug_pairs
  synergy_scores_pairs <- synergy_scores_pairs[,c("block_id","drug1","drug2","ZIP_synergy","HSA_synergy","Bliss_synergy","Loewe_synergy")]
  colnames(synergy_scores_pairs) <- c("block_id","drug1","drug2","ZIP","HSA","Bliss","Loewe")
  
  output_name_pairs <- paste("pairs_calculated_class_",current_file, sep = "")
  write.csv(synergy_scores_pairs, output_name_pairs, row.names = FALSE)

}
