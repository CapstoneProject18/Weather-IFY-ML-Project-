data=scan("Trade.txt",skip=1)

# plot data 
tsdisplay(data)

tsdisplay(diff(diff(data),4))

data=ts(data,start=c(1996,1),frequency = 4) #data is quarterly, thus frequency=4
data
M=auto.arima(data)
M

# auto arima selects the best model using smallest AIC, AICC or BIC value.
# AIC, AICC or BIC = penalized prediction error. More complex models are penalized more


# Make prediction

MF=forecast(M,h=3*4) # predict next three years values
MF

plot(MF)
lines(fitted(M),col=2)


# turn offshort cutts off

M1=auto.arima(data,stepwise = F,approximation = F)
M1


# Determine whether a model(s) is(are) adequate!
# Dtermine dependence structure of residuals(=actual-predicted)
# model is adequate if residuals contain no pattern, i.e. residuals are independent

tsdisplay(residuals(M))
tsdisplay(residuals(M1))


# Use a box test to cunduct a hypotheses test to determine if residuals are independent
Box.test(residuals(M)) #Ho:data are independent, Ha: data are dependent
?Box.test

# since pvalue=0.9046 is greater than 0.05, fail to reject Ho
# conclude no evidence that data are not independent
