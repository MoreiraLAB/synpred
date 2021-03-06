###
#Author: Pedro Matos-Filipe
#Group: Data-Driven Molecular Design
#Group Leader: Irina S. Moreira
#Project: SynPred
###

library(synergyfinder)
library(progress)

  calc_synergy <- function(i, data_set, final_df, j) {
    ds_in_analysis <- subset(data_set, block_id == i)
    numbers <- ds_in_analysis[c(1:8)]
    #numbers$conc_r <- numbers$conc_r * 1000
    #numbers$conc_c <- numbers$conc_c * 1000
    numbers$conc_r_unit <- 'nM'
    numbers$conc_c_unit <- 'nM'
    
    dose.response.mat <- ReshapeData(numbers, data.type = "viability")
    means <- list()
    l <- 1
    
    for (kind in c('ZIP', 'Bliss', 'HSA', 'Loewe')) {
      synergy.score <- CalculateSynergy(dose.response.mat, method = kind)
      y <- synergy.score$scores[[1]]
      summary.score <- mean(y[c(2:4), c(2:4)])
      means[l] <- summary.score
      l <- l + 1
    }
    
    means[5] <- levels(ds_in_analysis$drug_row)[[ds_in_analysis$drug_row[[1]]]]
    means[6] <-  levels(ds_in_analysis$drug_col)[[ds_in_analysis$drug_col[[1]]]]
    means[7] <- levels(ds_in_analysis$cell)[[ds_in_analysis$cell[[1]]]]
    means <- unlist(means)
    final_df[[j]] <- means
    return(final_df)
  }
  
  summary_synergy <- function(data_set) {
    
    final_df <- list()
    total = length(unique(data_set$block_id))
    pb <- progress_bar$new(format = "[:bar] :current/:total (:percent)", total = total)
    
    j <- 1
    pb$tick(0)
    
    for (i in unique(data_set$block_id)) {
      pb$tick(1)
      tryCatch({
        final_df <- calc_synergy(i, data_set, final_df, j)
      }, error=function(e){})
      j <- j + 1
    }
    
    k <- as.data.frame(do.call("rbind", final_df))
    colnames(k) <- c('ZIP', 'Bliss', 'HSA', 'Loewe', 'Drug1', 'Drug2', 'Cell')
    return(k)
  }

input_file <- read.csv("datasets/192_combinations.csv", sep = ",")
processed_table <- summary_synergy(input_file)
write.csv(processed_table,"NCI_ALMANAC_synergy_example.csv", row.names = FALSE)