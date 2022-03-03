library(ggplot2)

mm <- lm(mpg ~ wt + cyl, data = mtcars)
summary(mm)

ggplot(mtcars, aes(x=wt, y=mpg, color=as.factor(cyl))) +
  geom_point()

cd <- cooks.distance(mm)
ggplot(mtcars, aes(x=wt, y=mpg, color=cd)) +
  geom_point()

sort(cd)

# ------------------------------------------------------------------
library(ISLR)
library(dplyr)
head(Hitters)
h = na.omit(Hitters)

mm <- lm(Salary ~ ., data=h)
summary(mm)

cd <- cooks.distance(mm)
ggplot(h, aes(x=Hits, y=Salary, color=cd)) +
  geom_point()

influ <- cd[(cd > 3*mean(cd))]
names(influ)

outliers <- h[names(influ),]
RO <- anti_join(h, outliers)

mmr <- lm(Salary ~ ., data=RO)
summary(mmr)

cdr = cooks.distance(mmr)
ggplot(RO, aes(x=Hits, y=Salary, color=cdr)) +
  geom_point()


