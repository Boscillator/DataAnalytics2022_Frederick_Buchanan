library(readr)

df <- read_csv("GPW3_GRUMP_SummaryInformation_2010 (1).csv")
View(df)

plot(density(na.omit(df$PopulationPerUnit)))

plot(ecdf(df$PopulationPerUnit), do.points=F)

qqnorm(df$PopulationPerUnit); qqline(df$PopulationPerUnit)

boxplot(df$PopulationPerUnit)

