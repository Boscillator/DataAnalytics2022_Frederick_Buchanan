library(e1071)
data('iris')
mm <- svm(Species~., iris)
summary(mm)

pred <- fitted(mm)
table(pred, iris$Species)
