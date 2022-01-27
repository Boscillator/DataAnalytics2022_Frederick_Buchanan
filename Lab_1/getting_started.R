install.packages("MASS")
library(MASS)
attach(Boston)

?Boston

head(Boston)
dim(Boston)
str(Boston)
nrow(Boston)
ncol(Boston)
summary(Boston)

summary(Boston$crim)
