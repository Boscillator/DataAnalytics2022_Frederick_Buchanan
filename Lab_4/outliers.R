
cars1 <- cars[1:30,]
head(cars1)

cars_outliers <- data.frame(speed=c(19,19,20,20,20), dist=c(190,186,210,220,218))
cars_outliers
head(cars_outliers)
cars2 <- rbind(cars1, cars_outliers)

par(mfrow=c(1, 2))
plot(cars2$speed, cars2$dist, xlim=c(0, 28), ylim=c(0, 230), main="With Outliers", xlab="speed", ylab="dist",
     pch=".", col="red", cex=2)
abline(lm(dist ~ speed, data=cars2), col="blue", lwd=1, lty=2)
# Plot of original data without outliers. Note the change in slope (angle) of best fit line.
plot(cars1$speed, cars1$dist, xlim=c(0, 28), ylim=c(0, 230), main="Outliers removed \n A much better fit!",
     xlab="speed", ylab="dist", pch=".", col="red", cex=2)
abline(lm(dist ~ speed, data=cars1), col="blue", lwd=1, lty=2)

