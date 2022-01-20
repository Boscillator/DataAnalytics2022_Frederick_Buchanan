epi <- read.csv("2010EPI_data.csv", header = TRUE, skip=1, na.strings = c("NA","..","--"))

summary(epi$EPI)

boxplot(epi$EPI)
title("EPI")

fivenum(epi$EPI, na.rm=TRUE)

hist(epi$EPI)
