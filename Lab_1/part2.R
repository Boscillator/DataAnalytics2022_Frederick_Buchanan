epi <- read.csv("2010EPI_data.csv",
                header = TRUE,
                skip=1,
                na.strings = c("NA","..","--"))

attach(epi)

qqplot(AIR_H, WATER_H)

mm <- lm(ENVHEALTH ~ AIR_H)
summary(mm)
abline(mm)
plot(mm)

plot(ecdf(ENVHEALTH), do.points=FALSE, verticals=TRUE) 
hist(ENVHEALTH)
plot(density(na.omit(ENVHEALTH)))
shapiro.test(ENVHEALTH)
