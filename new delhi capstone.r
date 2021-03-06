#load packages and csv file
library(ggplot2)
library(dplyr)
library(gridExtra)
library(grid)
library(forecast)
library(knitr)
library(Amelia)
library(RColorBrewer)
library(ggfortify)
library(lubridate)

#Data preparation, NA values

df<-read.csv('testset.csv',sep=',')
missmap(df, legend = TRUE, col = c("#46ACC8","#EBCC2A"), y.cex = 0.001, x.cex = 0.8, rank.order = TRUE)

head(df %>% filter(nchar(as.character(datetime_utc))<6),3)

checkDATE<-function(x){
  val<-as.character(x)
  if(nchar(val)<=4) {return(0)}
  else{return(1)}
}
df$validDate<-sapply(df$datetime_utc,checkDATE)

# As they are several columns with a large number of missing values, I will select those with the less number of them.

df2 <- as.data.frame(df %>% dplyr::filter(validDate==1) %>% dplyr::select(-X_windchillm,-X_precipm,-X_wgustm,-X_heatindexm))

##Creating date/time features

getDATE<-function(x,pos1,pos2){
  val<-substr(strsplit(as.character(x),"-")[[1]][1],pos1,pos2)
  return(as.numeric(val))
}

getTIME<-function(x,pos){
  val<-strsplit(strsplit(as.character(x),"-")[[1]][2],":")[[1]][pos]
  return(as.numeric(val))
}

df2$year<-sapply(df2$datetime_utc,getDATE,1,4)
df2$month<-sapply(df2$datetime_utc,getDATE,5,6)
df2$day<-sapply(df2$datetime_utc,getDATE,7,8)
df2$hour<-sapply(df2$datetime_utc,getTIME,1)
df2$minute<-sapply(df2$datetime_utc,getTIME,2)
mymonths <- c("January","February","March","April","May","June","July","August","September","October","November","December")
df2$MonthAbb <- mymonths[ df2$month ]
df2$ordered_month <- factor(df2$MonthAbb, levels = month.name)

#Temperature feature
#Group_by `day`
#remove NA's from this column
df3<-df2[!is.na(df2$X_tempm),]

#group data by (year,month,day) by taking the mean of temp.
df4<-as.data.frame(df3 %>% select(year,ordered_month,day,X_tempm) %>% group_by(year,ordered_month,day) %>% summarise(dailyTemp = mean(X_tempm,na.rm=T)))

###Heatmap

ggplot(data=df4, aes(x=year,y=ordered_month)) + 
  geom_tile(aes(fill = dailyTemp),colour = "white") + 
  scale_fill_gradientn(colours=rev(brewer.pal(10,'Spectral'))) + 
  theme(legend.title=element_blank(),axis.title.y=element_blank(),axis.title.x=element_blank(),legend.position="top") + ggtitle("Temperature (daily average) in New-Dehli")

###Boxplot

ggplot(data=df4,aes(x=ordered_month,y=dailyTemp,color=dailyTemp)) + 
  scale_color_gradientn(colours=rev(brewer.pal(10,'Spectral'))) + 
  geom_boxplot(colour='black',size=.4,alpha=.5) + 
  geom_jitter(shape=16,position=position_jitter(0.2),size=.4) + 
  facet_wrap(~factor(year),ncol=7) + 
  theme(legend.position='none',axis.text.x = element_text(angle=45, hjust=1)) + 
  xlab('') + ylab('temperature (Celsius)')

# 
# * Both the heatmap and boxplot shows the same seasonality, i.e temperatures are higher in summer (although it seems the max. temperature is reached around April/May). 
# * We cannot see `by eyes` a potential increase of this behavior vs. Year ; 20 years of data is not enough I guess to notive a global increase 'warming' in New Dehli.

#Group by `month`
###Boxplot
# The difference here (compared to above) is that the mean is taken over the data within a month, not within a day.

df3 %>% select(ordered_month,year,X_tempm) %>% 
  group_by(ordered_month,year) %>% 
  summarise(monthlyTemp = mean(X_tempm)) %>% 
  ggplot(aes(x=ordered_month,y=monthlyTemp,color=monthlyTemp)) + 
  scale_color_gradientn(colours=rev(brewer.pal(10,'Spectral'))) + 
  geom_boxplot(colour='black',size=.4,alpha=.5) + 
  geom_jitter(shape=16,width=.2,size=2) + 
  theme(legend.title=element_blank(),legend.position='top',axis.text.x = element_text(angle=45, hjust=1)) + 
  ggtitle("Temperature (monthly average) in New-Dehli") + 
  xlab('') + ylab('temperature (Celsius)')

###Time serie
# A daily timeserie would have been a bit of an overkill for a time serie so a monthly one can capture the fluctuations (seasonality) and shows if there is a trend vs. Year (increase, statibility, decrease) 

#need to re-arrange the dataframe by year
df5<-as.data.frame(df3 %>% select(year,ordered_month,X_tempm) %>% group_by(year,ordered_month) %>% summarise(monthlyTemp = mean(X_tempm)))
df6<-arrange(df5,year)
myts <- ts(df5$monthlyTemp,start=c(1997,1), end=c(2016,12), frequency=12)
autoplot(decompose(myts))

####`Auto-Arima` prediction
d.arima <- auto.arima(myts)
d.forecast <- forecast(d.arima, level = c(95), h = 12)
autoplot(d.forecast)

###Checking `Auto-Arima` result (?)
# Because we have data over 20 years, we can split them into a train/test sample :
#   
#   * train sample : from 1997 to 2015
# * test sample : year 2016
# * run `arima` prediction over train sample
# * compare with (true) test data

df5<-as.data.frame(df3 %>% select(year,month,X_tempm) %>% group_by(year,month) %>% summarise(monthlyTemp = mean(X_tempm)))
df5$dateTS<-as.Date(paste0(df5$year,'-',df5$month,'-01'))

#select train/test sample

train <- df5 %>% dplyr::filter(year<2016)
test <- df5 %>% dplyr::filter(year>=2016) 
train_ts <- ts(train$monthlyTemp,start=c(1997,1), end=c(2015,12), frequency=12)
test_ts <- ts(test$monthlyTemp,start=c(2016,1), end=c(2016,12), frequency=12)

#arima model
m_aa = auto.arima(train_ts)
f_aa = forecast(m_aa, h=12)

#define functions to convert data from TS to dataframe (used for plotting TS with ggplot)
getMonth<-function(x){
  val<-strsplit(x,' ')[[1]][1]
  res<-match(tolower(val),tolower(month.abb))
  return(as.numeric(res))
}

getYear<-function(x){
  res<-strsplit(x,' ')[[1]][2]
  return(as.numeric(res))
}

convert_ts_df<-function(myts){
  #convert the forecast into a DF, convert the row index into a date column
  mydf<-data.frame(myts)
  dates<-rownames(mydf)
  rownames(mydf)<-1:nrow(mydf)  
  mydf$date<-dates
  
  #extract year,.month as numeric and make a Date() variable
  mydf$month<-sapply(mydf$date,getMonth)
  mydf$year<-sapply(mydf$date,getYear)
  mydf$dateTS<-as.Date(paste0(mydf$year,'-',mydf$month,'-01'))
  return(mydf)
}

arimaDF<-convert_ts_df(f_aa)

gPred <- ggplot() + geom_ribbon(data=arimaDF,aes(x=dateTS, ymin = Lo.95, ymax = Hi.95), alpha=.2)
gPred <- gPred + geom_line(data=arimaDF,aes(x=dateTS,y=Point.Forecast,color='arima_forecast'),size=1,linetype='dashed')
gPred <- gPred + 
  geom_line(data=filter(train,year>=2012),aes(x=dateTS,y=monthlyTemp,color='train'),size=1) + 
  geom_line(data=test,aes(x=dateTS,y=monthlyTemp,color='test'),size=1,alpha=.5) + 
  scale_colour_manual(name="",values=c(arima_forecast="#F21A00",train="#46ACC8",test="#0B775E")) + 
  theme(legend.position="top") + xlab('') + ylab('temperature (Celsius)')
gPred


##Correlation with other features
###vs. Humidity

#convert factor to numeric
df3$X_hum_2<-as.numeric(levels(df3$X_hum))[df3$X_hum]
df3$X_hum_2 <- ifelse(df3$X_hum_2>100,100,df3$X_hum_2)
pal <- colorRampPalette(c("blue", "yellow", "red"))
mycols=pal(12)
ggplot(data=df3,aes(x=X_tempm,y=X_hum_2,color=factor(ordered_month))) + geom_point(alpha=.5) + scale_color_manual(values=mycols) + xlim(0,50) + xlab('Temperature [Celsius]') + ylab('Humidity(%)')

# 
# * humidity percentage decreases as the temperature increases
# * humidity is lowest (hence temperature highesy) during April --> July (bottom rigth of the plot) 
# * to continue : look at other features and their correlation ; see impact on TS prediction


#select a day for testing 
test <- df3 %>% filter(year==2016 & month==6 & day==21)

#as.Date format
test$DATE<-as.POSIXct(paste(paste0(test$year,'-',test$month,'-',test$day), paste0(test$hour,':',test$min) ), format="%Y-%m-%d %H:%M")

#convert pressure mmbar in inchHg
test$X_pressureIn<-test$X_pressurem * 0.02953

coord<-c('North','NNE','NE','ENE','East','ESE','SE','SSE','South','SSW','SW','WSW','West','WNW','NW','NNW')
dx<-c(0,0.25,0.5,0.75,1,0.75,0.5,0.25,0,-0.25,-0.5,-0.75,-1,-0.75,-0.5,-0.25)
dy<-c(1,0.75,0.5,-.25,0,-0.25,-0.5,-0.75,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75)
directions<-data.frame('X_wdire'=coord,'dx' = dx, 'dy' =dy)
test<-as.data.frame(merge(test,directions,by='X_wdire'))

#conversion DATE as decimal_date for geom_segment
test$DATE_decimal<-decimal_date(test$DATE)

head(test)

panel1<-ggplot() + 
  geom_line(data=test,aes(x=DATE_decimal,y= X_tempm,color="temp"),size=1.5) +
  geom_line(data=test,aes(x=DATE_decimal,y= X_dewptm,color="dewpoint"),size=1.5) +
  scale_colour_manual(name="",values=c(temp="#F21A00",dewpoint="#0B775E")) +
  theme(legend.position='top',axis.title.x=element_blank(),axis.text.x=element_blank(),axis.ticks.x=element_blank()) +
  ylab('') + 
  ggtitle('2016-06-21')

panel2<-ggplot() + 
  geom_line(data=test,aes(x=DATE_decimal,y= X_pressureIn,color="pressure"),size=1.5) +
  geom_line(data=test,aes(x=DATE_decimal,y= X_hum_2,color="humidity"),size=1.5) +
  scale_colour_manual(name="",values=c(pressure="black",humidity="#81A88D")) +
  ylab('') +
  theme(legend.position='top',axis.title.x=element_blank(),axis.text.x=element_blank(),axis.ticks.x=element_blank())

panel3<-ggplot(data=test,aes(x=DATE_decimal,y=X_wspdm,color='windspeed')) + 
  geom_line(size=1.5) + 
  geom_segment(aes(xend= DATE_decimal + (.1/DATE_decimal), yend=X_wspdm + dy), arrow = arrow(length = unit(0.1, "cm")),color='black')  +
  xlab('') + ylab('') +
  scale_colour_manual(name="",values=c(windspeed="#46ACC8")) +
  theme(legend.position='top')

grid.arrange(panel1,panel2, panel3,ncol=1)