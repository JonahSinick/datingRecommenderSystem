setwd("~/Desktop/DatingRecommenderSystem")

#Libraries
##install.packages(c("readr", "dplyr", "lme4", "ggplot2", "softImpute"))
library(readr)
library(dplyr)
library(lme4)
library(ggplot2)
library(softImpute)

#Root mean square error function
rmse = function(arr1, arr2){
  sqrt(mean((arr1 - arr2)^2))
}


#Read data and rename columns
gens = read.csv("gender.dat", header= FALSE)
df = read_csv("ratings.dat", col_names = FALSE)
colnames(gens) = c("uid", "gender")
colnames(df) = c("uid", "pid", "rating")

#Select ratings of men's profiles by women
women = filter(gens, gender == "F")
men = filter(gens, gender == "M")
df = filter(df,uid %in% women$uid, pid %in% men$uid)

#Generate train set, validation set and test set according to a 60/20/20 split
set.seed(1); samps = sample(1:nrow(df)) %% 10
train = df[samps %in% 0:5,]
val = df[samps %in% 6:7,]
test = df[samps %in% 8:9,]

#Get train set rating counts by user
counts = aggregate(train["uid"], train["uid"], length)
colnames(counts)[2] = "count"

#Add fake user and fake profile for it to be possible to make predictions for new users and profiles
train = rbind(train, expand.grid(uid = max(df$uid) + 1, pid = 1:max(df$pid) , rating = mean(df$rating)))
train = rbind(train, expand.grid(uid = 1:max(df$uid) + 1, pid = max(df$pid) + 1, rating = mean(df$rating)))
train$rating = train$rating + 0.01*rnorm(nrow(train))

#Make a sparse rating matrix
m = Incomplete(train$uid, train$pid, train$rating)
#Scale the matrix
m = biScale(m, row.scale = FALSE, col.scale = FALSE, trace = TRUE, maxit = 100)

#Generate sequence of candidate regularization parameters
lam0  = lambda0(m)
lamseq=exp(seq(from=log(lam0),to=log(1),length=15))
ranks = c()
rank.max = 50

#Loop through regularization parameters and compute RMSE on validation set
bestModel = NULL
rmseBest = Inf
for( i in seq(along=lamseq)[c(1, 6)]){
  
  #Fit soft impute model
  fit=softImpute(m, lambda=lamseq[i],rank=rank.max,warm=bestModel, maxit = 1000, trace.it = TRUE, thresh = 2e-05)
  
  #Get rank
  ranks[i]=sum(round(fit$d,4)>0)
  
  if(ranks[i] == 1){
    oneFit = fit
  }
  
  #Get imputed ratings for validation set
  imps = impute(fit, val$uid, val$pid)
  #Truncate predicted ratings so that they don't extend outside of range 1 through 10
  imps = ifelse(imps > 10, 10, imps)
  imps = ifelse(imps < 1, 1, imps)
  
  #Get RMSE and log it
  r = rmse(val$rating, imps)
  cat(i,"lambda=",lamseq[i],"rank.max",rank.max,"rank",ranks[i],"rmse", r, "\n")
  
  #Break out of loop if RMSE has gone up
  if(r > rmseBest){
    break
  }else{
    r = rmseBest
    bestModel = fit
  }
}

#Evaluate performance on the test set

uids = unique(test$uid)

#Generates 
generateAccuracy = function(fit){
  test$predicted = impute(fit, test$uid, test$pid)
  means = sapply(uids, function(uid){
    if(uid %% 100 == 0){
      print(uid)
    }
    slice = test[test$uid == uid,c("rating", "predicted")]
    m = max(slice$rating)
    slice$tops = ifelse(slice$rating == m, 1, 0)
    len = sum(slice$tops)
    slice = slice[order(-slice["predicted"]),]
    mean(slice$tops[1:len])
  })  
  means
}

oneMeans = generateAccuracy(oneFit)
bestMeans = generateAccuracy(bestModel)
meandf = data.frame(uid = uids,oneMeans, bestMeans)
meandf = inner_join(counts, meandf)
library(reshape2)
meandf$diff = meandf$bestMeans - meandf$oneMeans

#Improvement for 2957 users
numImproved = table(meandf$diff > 0 )[["TRUE"]] -  table(meandf$diff < 0 )[["TRUE"]]
fracImproved = numImproved/length(diff)  # = 4.96%
mean(meandf$bestMeans)
meandf$logCount = log(meandf$count + 1)

#Fractional decrease in average of bad recommendations
fracDecreaseBad  = function(df){
  1 - (1 - mean(df$bestMeans))/(1 - mean(df$oneMeans))
}
fracDecreaseBad(meandf) # = 2.56%

#Get fractions for users with at least n ratings
s = sapply(1:1000, function(num){
  fracDecreaseBad(filter(meandf, count >= num)) # = 2.56%
})

#Plot these fractions
plotdf = data.frame(x = 1:400, y = 100*s[1:400])
g = ggplot(plotdf) + geom_smooth(aes(x, y)) +  theme_bw()
g = g + xlab("Minimum number of profiles rated")
g = g + ylab("Average % decrease in bad recommendations")
g

write_csv(meandf, "meandf.csv")
