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
#n_sheets <- length(sheetNames)

i <- 1
mydf <- read.xlsx(file_name, sheet = i, colNames = FALSE, rowNames = FALSE)
row.names(mydf) <- t(mydf[1])
mydf <- mydf[-1]
df.time <- as.numeric(mydf[1,])
mydf <- mydf[-1,]
fit.cosinor1 <- population.cosinor.lm(data = mydf, time = df.time, period = 24)

i <- 2
mydf <- read.xlsx(file_name, sheet = i, colNames = FALSE, rowNames = FALSE)
row.names(mydf) <- t(mydf[1])
mydf <- mydf[-1]
df.time <- as.numeric(mydf[1,])
mydf <- mydf[-1,]
fit.cosinor2 <- population.cosinor.lm(data = mydf, time = df.time, period = 24)

i <- 3
mydf <- read.xlsx(file_name, sheet = i, colNames = FALSE, rowNames = FALSE)
row.names(mydf) <- t(mydf[1])
mydf <- mydf[-1]
df.time <- as.numeric(mydf[1,])
mydf <- mydf[-1,]
fit.cosinor3 <- population.cosinor.lm(data = mydf, time = df.time, period = 24)

i <- 4
mydf <- read.xlsx(file_name, sheet = i, colNames = FALSE, rowNames = FALSE)
row.names(mydf) <- t(mydf[1])
mydf <- mydf[-1]
df.time <- as.numeric(mydf[1,])
mydf <- mydf[-1,]
fit.cosinor4 <- population.cosinor.lm(data = mydf, time = df.time, period = 24)

fit12 <- cosinor.poptests(fit.cosinor1, fit.cosinor2)
fit34 <- cosinor.poptests(fit.cosinor3, fit.cosinor4)

test <- c('test1 vs. test2', 'test3 vs. test4')
amplitude1 <- c(fit12[2,5], fit34[2,5])
amplitude2 <- c(fit12[2,6], fit34[2,6])
p_d_amplitude <- c(fit12[2,4], fit34[2,4])
acrophase1 <- c(fit12[3,5], fit34[3,5])
acrophase2 <- c(fit12[3,6], fit34[3,6])
p_d_acrophase <- c(fit12[3,4], fit34[3,4])

res <- data.frame(test, amplitude1, amplitude2, p_d_amplitude, acrophase1, acrophase2, p_d_acrophase)
write.csv(res, "supp_table_10.csv", row.names = FALSE)


