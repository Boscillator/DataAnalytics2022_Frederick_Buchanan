data('iris')
head(iris)

irisdata1 <- iris[,1:4]
head(irisdata1)

principal_components <- princomp(irisdata1, cor = TRUE, score = TRUE)
summary(principal_components)

plot(principal_components, type='l')
biplot(principal_components)
