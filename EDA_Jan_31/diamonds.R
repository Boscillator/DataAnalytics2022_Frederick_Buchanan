library(ggplot2)
library(dplyr)
View(diamonds)

diamonds2 <- diamonds %>% filter(carat == 0)

attach(diamonds2)

hist(carat, breaks=10)
hist(carat, breaks=100)
hist(carat, breaks=1000)

hist(price)
hist(price, breaks=100)
hist(price, breaks=1000)

d <- density(price)
plot(d)


sum(carat == 0.99)
sum(carat == 1)

ggplot(diamonds) +
  geom_histogram(mapping = aes(x = price),
                 binwidth =500) + coord_cartesian(ylim = c(0, 10000))

plot(carat, price)
r <- lm(log(price) ~ carat, data = diamonds)
summary(r)
plot(r)

r <- lm(price ~ carat + factor(cut) + factor(color) + factor(clarity))
summary(r)
plot(r)

plot(factor(cut), price)
plot(factor(color), price)


