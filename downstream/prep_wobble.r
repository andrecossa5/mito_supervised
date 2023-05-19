# Code from Caleb Lareu

library(dplyr)
library(data.table)

##

# Utils
annotate_synonomous <- function(df){

  sdf <- df %>% mutate(fp = Consequence == "synonymous_variant")
  variant_causes_Wobble_to_WCF <-  (grepl("\\*",sdf$Variant.tRNA) & grepl("near",sdf$Reference.tRNA)) & sdf$fp
  variant_causes_WCF_to_Wobble <-  (grepl("\\*",sdf$Reference.tRNA) & grepl("near",sdf$Variant.tRNA)) & sdf$fp
  variant_causes_Wobble_to_Wobble <-  (grepl("near",sdf$Variant.tRNA) & grepl("near",sdf$Reference.tRNA)) & sdf$fp
  variant_causes_WCF_to_WCF <- (grepl("\\*",sdf$Variant.tRNA) & grepl("\\*",sdf$Reference.tRNA)) & sdf$fp
  
  # Sanity check
  summary(as.numeric(variant_causes_Wobble_to_WCF) + 
            as.numeric(variant_causes_WCF_to_Wobble) + 
            as.numeric(variant_causes_Wobble_to_Wobble) + as.numeric(variant_causes_WCF_to_WCF))
  vec_annotate <- case_when(
    variant_causes_Wobble_to_WCF ~ "Wobble_to_WCF", 
    variant_causes_WCF_to_Wobble ~ "WCF_to_Wobble", 
    variant_causes_Wobble_to_Wobble ~ "Wobble_to_Wobble", 
    variant_causes_WCF_to_WCF ~ "WCF_to_WCF", 
    TRUE ~ "other"
  )
}

# Read data
path_tables <- '/Users/IEO5505/Desktop/mito_bench/data/'
path_original <- paste0(path_tables, 'functional_variant_tRNA_anticodon_table.tsv')
df <- fread(path_original) %>% data.frame()

# Format
df$mutation <- paste0("m", df$Position, df$Reference, ">", df$Variant)
df$syn_annotation <- annotate_synonomous(df)
df <- df %>% select(-rowid)

# Write
write.csv(df, paste0(path_tables, 'formatted_table_wobble.csv'))


##
