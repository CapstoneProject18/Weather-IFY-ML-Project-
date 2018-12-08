# setwd
setwd("~/Desktop/Satish patil/CaseStudies")

# Load the data
pumpdata <- read.csv("PumpTempData.csv",sep = ",",header = TRUE)

# Plot the data
library(ggplot2)
ggplot(pumpdata,aes(x=Date,y=Pump.Temp))+
  geom_line()

Trend <- seq(1,nrow(pumpdata),1)

Pumpdatawithtrend <- cbind(pumpdata,Trend)
#Pumpdata
ggplot(Pumpdatawithtrend,aes(x=Trend,y=Pump.Temp))+
  geom_line()+
  geom_point()

ggplot(Pumpdatawithtrend,aes(x=Time,y=Pump.Temp))+
  geom_line()+
  geom_point()


#Pumpdata normal distribution curve
ggplot(Pumpdatawithtrend,aes(Pump.Temp))+
  geom_density()

mean(Pumpdatawithtrend$Pump.Temp)
sd(Pumpdatawithtrend$Pump.Temp)

UpperLimit <- mean(Pumpdatawithtrend$Pump.Temp)+ (3*sd(Pumpdatawithtrend$Pump.Temp))
LowerLimit <- mean(Pumpdatawithtrend$Pump.Temp)-(3*sd(Pumpdatawithtrend$Pump.Temp))


# Modelling Building 


time <- function(t,lamda){
  previous <- pumpdata[1:t,c(3)]
  tempavg <- mean(previous)
  predicted <- seq(1,t,by=1)
  for(i in 1:length(predicted)){
    if(i==1){
      predicted[i] <- previous[i]
    }
    else{
      predicted[i] <- lamda*previous[i]+(1-lamda)*previous[i-1]
    }
  }
  Newdataframe <- cbind(pumpdata[1:t,],predicted)
  UCL <- mean(predicted)+2*sd(predicted)
  LCL <- mean(predicted)-2*sd(predicted)
  x <- NULL
  for(i in 1:nrow(Newdataframe)){
    if(Newdataframe$predicted[i] > UCL || Newdataframe$predicted[i] < LCL){
      x <- c(x,i)
    }
    else{
      x<- x
    }
  }
  dataframe <- Newdataframe[x,]
  return(dataframe[,c(1,4)])
  
}

time(273,0.3)
