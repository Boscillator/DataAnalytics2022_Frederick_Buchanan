wine_data <- read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", sep = ",")
head(wine_data)

names(wine_data) <- c("class", "alcoh", "malic", "ash", "alcalin", "magn", "pheno", "flavn", "nonflavn", "proan", "color", "hue", "od280", "proline")

nrow(wine_data)
head(wine_data)

heatmap(cor(wine_data), Rowv=NA, Colv=NA)

wine_data$class <- as.factor(wine_data$class)
pca <- prcomp(scale(wine_data[,-1]))
summary(pca)


