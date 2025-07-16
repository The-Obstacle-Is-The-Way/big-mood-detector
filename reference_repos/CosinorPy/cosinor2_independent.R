library(cosinor2)
library(openxlsx)
library(ggplot2)
library(magrittr)
library(stringr)
library(purrr)

options(stringsAsFactors=FALSE)

file_name = "test_data/data.xlsx"

sheetNames <- getSheetNames(file_name)
n_sheets <- length(sheetNames)

test <- c()
amplitude <- c()
acrophase <- c()
acrophase_corrected <- c()
p <- c()
p_amplitude <- c()
p_acrophase <- c()

for (i in 1:n_sheets) {
  mydf <- read.xlsx(file_name, sheet = i, colNames = FALSE, rowNames = FALSE)
  
  df.time <- as.numeric(mydf[,1])
  df.data <- mydf[,-1]
  
  mydf <- data.frame(df.time, df.data)
  colnames(mydf) <- c('time', 'y')
  fit.cosinor <- cosinor.lm(y ~ time(time), period = 24, data = mydf)
  
  
  
  amp <- fit.cosinor$coefficients[2]
  acr <- fit.cosinor$coefficients[3]
  acr_corrected <- 2*pi+correct.acrophase(fit.cosinor)
  pval <- cosinor.detect(fit.cosinor)[4]
  s <- summary(fit.cosinor)$transformed.table
  pval_amp <- s$p.value[2]
  pval_acr <- s$p.value[3]
  
  test <- c(test, sheetNames[i])
  amplitude <- c(amplitude, amp)
  acrophase <- c(acrophase, acr)
  acrophase_corrected <- c(acrophase_corrected, acr_corrected)
  p <- c(p, pval)
  p_amplitude <- c(p_amplitude, pval_amp)
  p_acrophase <- c(p_acrophase, pval_acr)
  
}

res <- data.frame(test, p, amplitude, p_amplitude, acrophase, p_acrophase, acrophase_corrected)
write.csv(res, "supp_table_7.csv", row.names = FALSE)













