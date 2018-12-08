## bicup2006.xls Demand Transportation
## Visit www.forecasting
library(plyr)
library(dplyr)
library(ggplot2)
library(lubridate)
## Reading the data into R
data <- read.clipboard(header = TRUE)
data_test <- read.clipboard(header = TRUE)
data <- cbind(data,trend)

#Adding trend to the database 
trend <- seq(1,1323,1)
trend <- data.frame(trend)
data <- cbind(data,trend)

## Plotting the graph 
ggplot(data,aes(x= trend,y = DEMAND))+
  geom_line()

## Extracting Week from the data
data$DATE <- dmy(data$DATE)
data$TIME <- hm(data$TIME)
data <- mutate(data,hour = hour(data$TIME))
data <- mutate(data,min = minute(data$TIME))
data <- mutate(data,day = day(data$DATE))
data <- mutate(data,wday = wday(data$DATE))


## Plotting the graph 
ggplot(data,aes(x= day,y = DEMAND))+
  geom_point()

ggplot(data,aes(x= wday,y = DEMAND))+
  geom_point()

ggplot(data,aes(x= hour,y = DEMAND))+
  geom_point()

ggplot(data,aes(x= min,y = DEMAND))+
  geom_point()

## Creating dummies for the data. 
## Assumptions:
#1. there is no Trend as observed in the data 

unique(data$hour)
data <- mutate(data,dummy6 = ifelse(data$hour == 6, 1, 0))
data <- mutate(data,dummy7 = ifelse(data$hour == 7, 1, 0))
data <- mutate(data,dummy8 = ifelse(data$hour == 8, 1, 0))
data <- mutate(data,dummy9 = ifelse(data$hour == 9, 1, 0))
data <- mutate(data,dummy10 = ifelse(data$hour == 10, 1, 0))
data <- mutate(data,dummy11 = ifelse(data$hour == 11, 1, 0))
data <- mutate(data,dummy12 = ifelse(data$hour == 12, 1, 0))
data <- mutate(data,dummy13 = ifelse(data$hour == 13, 1, 0))
data <- mutate(data,dummy14 = ifelse(data$hour == 14, 1, 0))
data <- mutate(data,dummy15 = ifelse(data$hour == 15, 1, 0))
data <- mutate(data,dummy16 = ifelse(data$hour == 16, 1, 0))
data <- mutate(data,dummy17 = ifelse(data$hour == 17, 1, 0))
data <- mutate(data,dummy18 = ifelse(data$hour == 18, 1, 0))
data <- mutate(data,dummy19 = ifelse(data$hour == 19, 1, 0))
data <- mutate(data,dummy20 = ifelse(data$hour == 20, 1, 0))
data <- mutate(data,dummy21 = ifelse(data$hour == 21, 1, 0))

unique(data$wday)
## Creating dummies for Weekday
data <- mutate(data, w1= ifelse(data$wday == 1,1,0))
data <- mutate(data, w2= ifelse(data$wday == 2,1,0))
data <- mutate(data, w3= ifelse(data$wday == 3,1,0))
data <- mutate(data, w4= ifelse(data$wday == 4,1,0))
data <- mutate(data, w5= ifelse(data$wday == 5,1,0))
data <- mutate(data, w6= ifelse(data$wday == 6,1,0))

##Creating dummies for .
data <- mutate(data, min0= ifelse(data$min == 0,1,0))
data <- mutate(data, min15= ifelse(data$min == 15,1,0))
data <- mutate(data, min30= ifelse(data$min == 30,1,0))


## Removing some of the variables from the data 
data_final <- data[,-c(1,2,4,5,6,7,8)]

## Dividing the data into training and validation test
data_training <- data_final[1:1200,]
data_validation <- data_final[1201:1323,]


## Building a linear model on Linear Regression
model1 <- lm(DEMAND ~.-min0-min15-min30-dummy6-dummy7-dummy8, data_training)
summary(model1)

plot(model1$residuals,type ="l")
plot(data_training$DEMAND,type="l")
lines(model1$fitted,lwd=2,col = "red")


## Predicting the variables
true_value <- data_validation[,1]
data_validation <- data_validation[,-1]
predicted_y <- predict(model1,data_validation)
error <- true_value-predicted_y
error2 <- error^2
MSE = mean(error^2)
MSE

## 288.106

## Plotting predicting and true vale on the same graph
validation <- cbind(data.frame(true_value),data.frame(predicted_y))
plot(validation$true_value,type="l")
lines(validation$predicted_y,lwd=2,col = "red")

## Since we haven't captured the Residuals
Acf(model1$residuals,lag.max = 15,main="")

#Using lag1
data_training$lag1 <- c(NA,data_training$DEMAND[1:1199])
data_training <- data_training[-1,]

model2 <- lm(DEMAND ~.-dummy6-dummy7-dummy8-dummy21-w1-min0-min15-min30, data_training)
summary(model2)

plot(model2$residuals,type ="l")
plot(data_training$DEMAND,type="l")
lines(model1$fitted,lwd=2,col = "red")

## Adding another lag
data_training$lag2 <- c(NA,NA,data_training$DEMAND[1:1198])
data_training <- data_training[-c(1,2),]

model3 <- lm(DEMAND ~.-dummy6-dummy7-dummy8-dummy21-w1-min0-min15-min30-dummy20, data_training)
summary(model3)

plot(model3$residuals,type ="l")
plot(data_training$DEMAND,type="l")
lines(model3$fitted,lwd=2,col = "red")

## Calculating the MSE 
data_validation$lag1<- c(0,data_validation$DEMAND[1:122])
data_validation$lag2<- c(0,0,data_validation$DEMAND[1:121])
true_value <- data_validation[,1]
data_validation <- data_validation[,-1]
predicted_y <- predict(model3,data_validation)
error <- true_value-predicted_y
error2 <- error^2
MSE = mean(error^2)
MSE
##122.4355

## Predicting the variables
data_test$DATE <- dmy(data_test$DATE)
data_test$TIME <- hm(data_test$TIME)
data_test <- mutate(data_test,hour = hour(data_test$TIME))
data_test <- mutate(data_test,min = minute(data_test$TIME))
data_test <- mutate(data_test,day = day(data_test$DATE))
data_test <- mutate(data_test,wday = wday(data_test$DATE))

unique(data$hour)
data_test <- mutate(data_test,dummy6 = ifelse(data_test$hour == 6, 1, 0))
data_test <- mutate(data_test,dummy7 = ifelse(data_test$hour == 7, 1, 0))
data_test <- mutate(data_test,dummy8 = ifelse(data_test$hour == 8, 1, 0))
data_test <- mutate(data_test,dummy9 = ifelse(data_test$hour == 9, 1, 0))
data_test <- mutate(data_test,dummy10 = ifelse(data_test$hour == 10, 1, 0))
data_test <- mutate(data_test,dummy11 = ifelse(data_test$hour == 11, 1, 0))
data_test <- mutate(data_test,dummy12 = ifelse(data_test$hour == 12, 1, 0))
data_test <- mutate(data_test,dummy13 = ifelse(data_test$hour == 13, 1, 0))
data_test <- mutate(data_test,dummy14 = ifelse(data_test$hour == 14, 1, 0))
data_test <- mutate(data_test,dummy15 = ifelse(data_test$hour == 15, 1, 0))
data_test <- mutate(data_test,dummy16 = ifelse(data_test$hour == 16, 1, 0))
data_test <- mutate(data_test,dummy17 = ifelse(data_test$hour == 17, 1, 0))
data_test <- mutate(data_test,dummy18 = ifelse(data_test$hour == 18, 1, 0))
data_test <- mutate(data_test,dummy19 = ifelse(data_test$hour == 19, 1, 0))
data_test <- mutate(data_test,dummy20 = ifelse(data_test$hour == 20, 1, 0))
data_test <- mutate(data_test,dummy21 = ifelse(data_test$hour == 21, 1, 0))

unique(data$wday)
## Creating dummies for Weekday
data_test <- mutate(data_test, w1= ifelse(data_test$wday == 1,1,0))
data_test <- mutate(data_test, w2= ifelse(data_test$wday == 2,1,0))
data_test <- mutate(data_test, w3= ifelse(data_test$wday == 3,1,0))
data_test <- mutate(data_test, w4= ifelse(data_test$wday == 4,1,0))
data_test <- mutate(data_test, w5= ifelse(data_test$wday == 5,1,0))
data_test <- mutate(data_test, w6= ifelse(data_test$wday == 6,1,0))

##Creating dummies for .
data_test <- mutate(data_test, min0= ifelse(data_test$min == 0,1,0))
data_test <- mutate(data_test, min15= ifelse(data_test$min == 15,1,0))
data_test <- mutate(data_test, min30= ifelse(data_test$min == 30,1,0))


data_test_final <- data_test[,-c(1,2,3,4,5,6)]


## Adding the lag variables to the dataset
data_test_final$lag1<- c(2,rep(0,188))
data_test_final$lag2<- c(2,6,rep(0,187))


## Since the MSE is less We will build the model on entire data
data <- data[,-c(1,2,4,5,6,7,8)]
data$lag1 <- c(NA,data$DEMAND[1:1322])
data$lag2 <- c(NA,NA,data$DEMAND[1:1321])

data <- data[-c(1,2),]

## Building the final model.
model4 <- lm(DEMAND ~.-dummy6-dummy7-dummy8-dummy21-w1-min0-min15-min30-dummy20, data)
summary(model4)

plot(model4$residuals,type ="l")
plot(data$DEMAND,type="l")
lines(model4$fitted,lwd=2,col = "red")

plot(data$DEMAND,type="l")
lines(predicted_y$rep.0..189.,lwd=2,col = "red")


## Finding the values of the predicted values 
predicted_y <- data.frame(rep(0,189))
for (i in 1:187){
  data_predict <- data_test_final[i,]
  predicted_y[i,1] <- predict(model3,data_predict)
  data_test_final[i+1,26] <- predicted_y[i,1]
  data_test_final[i+2,27] <- predicted_y[i,1]
}

x<- data.frame(data$DEMAND)
y <- data.frame(predicted_y$rep.0..189.) 
names(y) <- "data.DEMAND"
data_predicted <- rbind(x,y)

plot(data_predicted$data.DEMAND,type="l")
############################################################################################################
