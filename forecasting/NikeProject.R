## Nike Project

## Given the sales of Nike Quarterly
#Robertashas collected quarterly data on Nike’s revenue for the fiscal years 1999 through 2008; 
#for instance, data for fiscal year 1999 refers to the time period from June 1, 1998 through May 31, 1999. 
#Validation set = 2009 data.

#Given the Data set , The Quarters are formed in this way 
# Q1 -- June to Aug
# Q2 -- Sep to Dec
# Q3 -- Dec to Feb
# Q4 -- Mar - May 

# Load the data into R 
data <- read.csv("Revenuenike.csv",header = TRUE, sep = ",")

# The initial data was not given in this way. I have modified and made some changes so that it is easy for computations

#There are 5 crucial methods of dealing this 
# 1. Linear Regression 
# 2. Smooting Method (Data-Driven Method)
# 3. Classical Time Series Decomposition 
# 4. ARIMA Models 
# 5. Ensemble methods (Combaining 2-3 models)


## Data Exploration 
library(ggplot2)
str(data)

##Plots
#Autoplot

# Complete plots
autoplot(train.ts)

# Seasonal Plots 
ggseasonplot(train.ts,year.labels = TRUE)

# Monthly Plot
ggmonthplot(train.ts)

# Lag plot - Let us know how Lag1-4 is correlated with train.ts - View for Auto-correlation
lag.plot(train.ts,lags=4,do.lines = FALSE)

## This function will give u a BoxCox.lambda for you.
lambda <- BoxCox.lambda(train.ts) # = 0.6234
plot(BoxCox(train.ts,lambda))


# Checking for stationarity in the data

adf.test(train.ts)
kpss.test(train.ts)

# The data is Non-stationary
tsdisplay(train.ts)

# Removing the trend
tsdisplay(diff(train.ts,1))

# Removing the seasonality
tsdisplay(diff(diff(train.ts,1),4))

# Random- walk has been established.
######################################################################################################

# Decomposition Method 

# We use the same library package "forecast" for developing the Model 

library(forecast)
train.ts <- ts(data$Revenue[1:40],start=c(1999,1),frequency = 4)
valid.ts <- ts(data$Revenue[41:44],start=c(2009,1),frequency = 4)


decom_model <- stl(train.ts, s.window = "periodic")
plot(decom_model)

# Findings 
# 1. There is a constant amplitude Sesonality in the data.
# 2. The trend is little curvilinear but increasing constantly with time.


# Modelling 

fit <- decompose(train.ts, type="multiplicative")
plot(fit)

fit1 <- decompose(train.ts, type="additive")
plot(fit1)

fit2 <- stl(train.ts, t.window=15, s.window="periodic", robust=TRUE)
eeadj <- seasadj(fit2)
plot(naive(eeadj), xlab="New orders index",
     main="Naive forecasts of seasonally adjusted data")

fcast <- forecast(fit2, method="naive",level = 0.95)
plot(fcast, ylab="New orders index")

accuracy(fcast,valid.ts)

# RMSE 
# TrainingSet == 150.0758
# Test set  == 300.9574


######################################################################################################

# Arima Model


#Model -1
# Automatic
M=auto.arima(train.ts,stepwise = FALSE)
M
res <- residuals(M)
tsdisplay(res)

#Ljung–Box test
#H0: The data are independently distributed (i.e. the correlations in the population from which the 
#sample is taken are 0, so that any observed correlations in the data result from randomness of the 
#sampling process).
#Ha: The data are not independently distributed; they exhibit serial correlation.
Box.test(res, lag=16, fitdf=4, type="Ljung")

# There is no more correlation and our model is fine.

#ARIMA(0,1,2)(0,1,0)[4]
#AIC - 422.55
# auto arima selects the best model using smallest AIC, AICC or BIC value.
# AIC, AICC or BIC = penalized prediction error. More complex models are penalized more


# Make prediction

MF=forecast(M,h=1*4, level = 0.95) # predict next three years values

plot(MF)

accuracy(MF,valid.ts)
#RMSE(test) = 764.43821
#RMSE(train) = 85.16244


####################################################################################################


# Smoothing methods 
ses <- ets(train.ts,model="ZZZ")
summary(ses)

# ETS(M,A,M)

#Prediction
ses.pred <- forecast(ses,h=1*4,level = 0.95)

plot(ses.pred)

accuracy(ses.pred,valid.ts)

# RMSE 
# Train == 94.6973
# Test == 725.71542

####################################################################################################

# Of all the Models Decomposition techniques is giving a better output as the Test_RMSE is very less
# Ensemble model can be applied and checked for better accuracy.
