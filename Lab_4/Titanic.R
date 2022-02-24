data(Titanic)
View(Titanic)

df <- data.frame(Titanic)

ctrl <- rpart.control(minsplit = 5, cp=0)
mm <- rpart(Survived ~ Class + Age + Sex + Freq, data=df, control=ctrl)
plot(mm)
text(mm)

