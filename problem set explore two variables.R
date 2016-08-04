setwd("C:/Users/Weiwen/OneDrive/Study/R")
library(ggplot2)
data("diamonds")

ggplot(aes(x = price, y = x), data = diamonds) + geom_point()

ggplot(aes(x = price, y = y), data = diamonds) + geom_point() + ylim(0,20)

ggplot(aes(x = price, y = z), data = diamonds) + geom_point() + ylim(0,10)

with(diamonds, cor.test(x = price, y =  x))

with(diamonds, cor.test(x = price, y =  y))

with(diamonds, cor.test(x = price, y =  z))

ggplot(aes(x = price, y = depth), data = diamonds) + geom_point(alpha = 0.01)

summary(diamonds$depth)

with(diamonds, cor.test(x = price, y =  depth))

ggplot(aes(x = price, y = carat), 
       data = subset(diamonds, diamonds$price < quantile(diamonds$price, 0.99) & diamonds$carat < quantile(diamonds$carat, 0.99))) + 
  geom_point()

ggplot(aes(x = price, y = x*y*z), data = diamonds) + geom_point(alpha = 0.01)

with(subset(diamonds, diamonds$x*diamonds$y*diamonds$z <= 800 & diamonds$x*diamonds$y*diamonds$z >0), cor.test(x = price, y =  x*y*z))

ggplot(aes(x = price, y = x*y*z), data = diamonds) + 
  geom_point(alpha = 0.01) + 
  ylim(0,800) + 
  geom_smooth(method = "lm")

library(dplyr)  

group_by_clarity <- group_by(diamonds, clarity)


diamondsByClarity <- summarise(group_by_clarity, mean_price = mean(price),
            median_price = median(price),
            min_price = min(price),
            max_price = max(price),
            n = n())

diamondsByClarity <- arrange(diamondsByClarity)


install.packages("magrittr")
library(magrittr)

install.packages("gridExtra")
library(gridExtra)

diamonds_by_clarity <- group_by(diamonds, clarity)
diamonds_mp_by_clarity <- summarise(diamonds_by_clarity, mean_price = mean(price))

diamonds_by_color <- group_by(diamonds, color)
diamonds_mp_by_color <- summarise(diamonds_by_color, mean_price = mean(price))

b1 <- ggplot(aes(x = clarity), data = diamonds_mp_by_clarity) + geom_bar(aes(weight = mean_price))
b2 <- ggplot(aes(x = color), data = diamonds_mp_by_color) + geom_bar(aes(weight = mean_price))

grid.arrange(b1, b2)
