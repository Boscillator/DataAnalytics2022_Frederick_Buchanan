epi <- read.csv("2010EPI_data.csv",
                header = TRUE,
                skip=1,
                na.strings = c("NA","..","--"))

View(epi)

attach(epi)

plot(density(na.omit(GDPCAP07)))

plot(ecdf(GDPCAP07), do.points=FALSE, verticals=TRUE)

GDPDesert <- na.omit(GDPCAP07[Desert == F])
GDPNotDesert <- na.omit(GDPCAP07[Desert == T])

boxplot(GDPDesert, GDPNotDesert)

r <- lm(EPI ~ GDPCAP07 + factor(EPI_regions))
summary(r)
