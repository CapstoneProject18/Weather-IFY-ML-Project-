

```{r}
# Upload packages
library(ggplot2)
library(maps)
library(ggthemes)
library(gganimate)
library(dplyr)


# Load input data
data1<-read.csv('E:/Assignment1/manik/Weather-IFY-ML-Project-/pacific data/atlantic.csv')

# Format the coordinates
dat<-as.character(data1$Latitude)
new<-substr(dat,1,nchar(dat)-1)
lat<-as.numeric(new)
dat<-as.character(data1$Longitude)
new<-substr(dat,1,nchar(dat)-1)
long<-as.numeric(new)
long<- -abs(long)
coord<-as.data.frame(long)
coord<-cbind(coord, lat)
out<-subset(coord, long < -100) ##Find the outliers
total<-cbind(data1, coord)
outRM<-total[-c(22985,22986),]  ##Remove the outliers
hu<-outRM

# Create a time column
hu$Date<-as.character(hu$Date)
hu$Time[hu$Time>0]<-hu$Time[hu$Time>0]/100
hu$Date<-as.Date(hu$Date,'%Y%m%d')
hu$Time<-ISOdatetime(hu$Date, hu$Time,0,0, 0, 0)

# Some data manipulation
hu$MaxWind.mph<-hu$Maximum.Wind*1.150779 # Convert wind speed from knot to miles per hour
hu$Name<-trimws(as.character(hu$Name))


# Select columns of interest
hu.1<-as.data.frame(hu[c("ID","Name","lat","long","MaxWind.mph", "Time","Event", "Status")])

hu.1$Event<-trimws(as.character(hu.1$Event))
hu.1$Status<-trimws(as.character(hu.1$Status))

# Select only storms which made landafall as hurricanes in 2004 and 2005
hu.list0405.1<-subset(hu.1, format.Date(Time, "%Y") %in% c("2004","2005"))
hu.list0405.2<-subset(hu.1, Event=="L" & Status=="HU")
hu.names0405<-intersect(hu.list0405.1$ID,hu.list0405.2$ID)
hu.0405<-subset(hu.1, ID %in% hu.names0405)
hu.0405<-subset(hu.0405,  format.Date(Time, "%H") %in% c("06","18"))


## Add a short time series at the end to extend the final screen
end.data<-data.frame(matrix(0, nrow = 10, ncol = 8))
end.data[,6]<-seq(max(hu.0405$Time), by = "days", length=10)
names(end.data)<-names(hu.0405)
end.data$ID<-"END"
hu.0405<-rbind(hu.0405,end.data)


# Create the animation plot
p3 <-hu.0405 %>% na.omit() %>%  ggplot() + borders("world",colour="black",fill="slategray2", resolution=0) +
  theme(panel.background = element_rect(fill = 'white'),panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  geom_point(aes(long, lat, frame=Time, colour=MaxWind.mph,size=MaxWind.mph), alpha=0.5) +
  geom_path(aes(long, lat,frame=Time, colour=MaxWind.mph,cumulative = TRUE, group=Name)) +
  geom_text(aes(x=long, y=lat-2,frame=Time, label=Name)) +
  xlab("LONGITUDE") + ylab("LATITUDE") +  
  scale_color_continuous(low = "green", high = "red",limits=c(0, 200), breaks=seq(0,200, by=50)) +  scale_size_continuous(limits=c(0, 200), breaks=seq(0,200, by=50))  +  guides(color= guide_legend(), size=guide_legend()) + 
  coord_map(projection = "mercator") +
  scale_y_continuous( limits = c( 0 , +50 )) + scale_x_continuous( limits = c( -130 , -5 ))
 
 gganimate(p3,interval = 0.3,ani.width=1000,ani.res = 1500,"output.gif")
```

