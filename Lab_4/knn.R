library(ggplot2)
par(mfrow=c(1, 1))

df <- data.frame(iris)
View(df)
clusts = kmeans(df[,0:3], 5)

cc <- as.factor(clusts$cluster)

ggplot(df, aes(x=Sepal.Length, y=Sepal.Width, shape=cc)) + 
  scale_color_brewer(palette="Dark2") +
  #theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank()) +
  geom_point()

table(df$Species, cc)
