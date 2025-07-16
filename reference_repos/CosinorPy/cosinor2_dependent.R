library(cosinor2)
library(openxlsx)
library(ggplot2)
library(magrittr)
library(stringr)
library(purrr)
library(matrixStats)

options(stringsAsFactors=FALSE)

# the input file can be produced with the file_parser.export_cosinor2 function
file_name = "test_data/dependent_data_cosinor2.xlsx"

sheetNames <- getSheetNames(file_name)
n_sheets <- length(sheetNames)

test <- c()
p <- c()
amplitude <- c()
LB_amplitude <- c()
UB_amplitude <- c()
acrophase <- c()
LB_acrophase <- c()
UB_acrophase <- c()


for (i in 1:n_sheets) {
  mydf <- read.xlsx(file_name, sheet = i, colNames = FALSE, rowNames = FALSE)
  
  
  
  ## row names are in the first column
  row.names(mydf) <- t(mydf[1])
  mydf <- mydf[-1]
  
  # time is in the first row
  df.time <- as.numeric(mydf[1,])
  
  
  # data without time
  mydf <- mydf[-1,]

  fit.cosinor <- population.cosinor.lm(data = mydf, time = df.time, period = 24)
  pval <- cosinor.detect(fit.cosinor)[4]
  
  amp <- fit.cosinor$coefficients$Amplitude
  acr <- fit.cosinor$coefficients$Acrophase
  
  CIs <- fit.cosinor$conf.ints
  amp_UB <- CIs[3]
  amp_LB <- CIs[4]
  acr_UB <- CIs[5]
  acr_LB <- CIs[6]
  
  test <- c(test, sheetNames[i])
  p <- c(p, pval)
  amplitude <- c(amplitude, amp)
  LB_amplitude <- c(LB_amplitude,amp_LB)
  UB_amplitude <- c(UB_amplitude,amp_UB)
  acrophase <- c(acrophase,acr)
  LB_acrophase <- c(LB_acrophase,acr_LB)
  UB_acrophase <- c(UB_acrophase,acr_UB)
  
}

res <- data.frame(test, p, amplitude, LB_amplitude, UB_amplitude, acrophase, LB_acrophase, UB_acrophase)
write.csv(res, "supp_table_9.csv", row.names = FALSE)





