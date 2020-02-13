## Project: Sentiment Analysis
## Author: Matias Barra
## Date: 03/02/2020
## Data Preprocess Script
####------------------------------------ Set Environment and upload datasets --------------------####

## Packages 
pacman::p_load(caret, ggplot2, plotly, rstudioapi, corrplot, dplyr, corrplot,data.table,
               reshape,reshape2, randomForest, doParallel, kknn, C50, e1071, party, ROSE)


loc   <- grep("ip_data_preprocess.R",list.files(recursive=TRUE),value=TRUE)
iloc  <- which(unlist(gregexpr("/ip_data_preprocess.R",loc)) != -1)
myloc <- paste(getwd(),loc[iloc],sep="/")
setwd(substr(myloc,1,nchar(myloc)-20))


# ## Github
# 
# current_path = rstudioapi::getActiveDocumentContext()$path # save working directory
# setwd(dirname(current_path))
# setwd("..")

## DataSets files 

iphone_matrix <- read.csv("../../data/iphone/iphone_smallmatrix_labeled_8d.csv", header = TRUE)


####------------------------------------ Explore the data --------------------####

# Explorative plot
plot_ly(iphone_matrix, x= ~iphone_matrix$iphonesentiment, type='histogram')
plot_ly(iphone_matrix, x= ~iphone_matrix$iphone, type='histogram')

# Check for NAs
apply(iphone_matrix, 2, function(x) any(is.na(x))) # no NA in iPhone data

#### Reduce Sentiment to 3 categories ####
# I decided to use only 3 categories: negative (0 and 1), neutral (2 and 3) and positive (4 and 5). 

# recode sentiment to combine factor levels 0 & 1, 2 & 3 and 4 & 5
iphone_matrix$iphonesentiment <- recode(iphone_matrix$iphonesentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 2, '4' = 3, '5' = 3) 

# Explorative plot
hist_ip_sent <- plot_ly(iphone_matrix, x= ~iphone_matrix$iphonesentiment, type='histogram')

## Correlation Matrix only using iPhone data 
#select the vars that actually say something about the iphone
iphone_unique <- iphone_matrix %>% select(iphone, ios, iphonecampos, iphonecamneg, iphonecamunc,
                                          iphonedispos, iphonedisneg, iphonedisunc,iphoneperpos, 
                                          iphoneperneg, iphoneperunc, iosperpos, iosperneg, 
                                          iosperunc, iphonesentiment)

# Using cor() and corpolot() to to identify correlated variables
# create a new data set and remove features highly correlated with the dependant 
iphoneCor <- cor(iphone_unique)
corrplot(iphoneCor, type = "upper", tl.pos = "td",
         method = "square", tl.cex = 0.8, order= "hclust", tl.col = 'black', diag = TRUE)

# Create decision tree for iphone
iphoneDtree <- ctree_control(maxdepth = 10)
iphoneDT <- ctree(iphonesentiment ~ ., data = iphone_unique)
plot(iphoneDT)

# Linear Relationships are fairly weak as they are not strongly correlated
# Decision Tree highlights: iphone, ios perpos, iphonecamneg

#------- Analyzing Feature Variance

# nearZeroVar() with saveMetrics = TRUE returns an object containing a table 
# including: frequency ratio, percentage unique, zero variance and near zero variance 

ip_nzvMetrics <- nearZeroVar(iphone_unique, saveMetrics = TRUE)
ip_nzvMetrics

# nearZeroVar() with saveMetrics = FALSE returns an vector 
ip_nzv <- nearZeroVar(iphone_unique, saveMetrics = FALSE) 
ip_nzv

# create a new data set and remove near zero variance features
iphoneNZV <- iphone_unique[,-ip_nzv]
str(iphoneNZV)
rm(ip_nzv, ip_nzvMetrics)

saveRDS(iphoneNZV, file = "../../data/iphone/iphoneNZV.rds")

####------- PCA (Principal Component Analysis) ----####
# removes all of your features and replaces them with mathematical representations of their variance
# data = training and testing from iphone_matrix (no feature selection) 
# create object containing centered, scaled PCA components from training set
# excluded the dependent variable and set threshold to 0.95

preprocessParams <- preProcess(iphone_matrix[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParams) # need 25 Components to capture 95% of output

#Split the data
ipMatrixpartition <- createDataPartition(iphone_matrix$iphonesentiment, times = 1, p = .7, list = FALSE)
ipMatrixtrain <- iphone_matrix[ipMatrixpartition,]
ipMatrixtest <- iphone_matrix[-ipMatrixpartition,]
ipMatrixtrain$iphonesentiment <- as.factor(ipMatrixtrain$iphonesentiment)
ipMatrixtest$iphonesentiment <- as.factor(ipMatrixtest$iphonesentiment)


# use predict to apply pca parameters, create training, exclude dependant
train_pca <- predict(preprocessParams, ipMatrixtrain[,-59])
# add the dependent to training 
train_pca$iphonesentiment <- ipMatrixtrain$iphonesentiment

# use predict to apply pca parameters, create testing, exclude dependant
test_pca <- predict(preprocessParams, ipMatrixtest[,-59])
# add the dependent to test
test_pca$iphonesentiment <- ipMatrixtest$iphonesentiment

# inspect results
str(train_pca)
str(test_pca)

saveRDS(train_pca, file ="../../data/iphone/train_pca.rds")
saveRDS(test_pca, file ="../../data/iphone/test_pca.rds")

hist_ip_sent
# hist_ip_sent we can observe that here is a data unbalaced problem, to solve this problem we goint to implement
# some technique like caret::downSample() and upSample

#### New Approach - Resampling dataSet to deal with imbalanced dataSets with binary  ####
table(iphone_unique$iphonesentiment)
# 1    2    3 
# 2352 1642 8979  We can see the imbalance 

# split the data 
Over_partition <- createDataPartition(iphone_unique$iphonesentiment, times = 1, p = .7, list = FALSE)
trainOver <- iphone_unique[Over_partition,]
testOver <- iphone_unique[-Over_partition,]

# Creating new data frames using the majority(3) and the others options in order 
# to get binary data imbalanced and apply ROSE method 
df3 <- filter(trainOver, iphonesentiment %in% c(3))
df3_1 <- filter(trainOver, iphonesentiment %in% c(1, 3))
df3_2 <- filter(trainOver, iphonesentiment %in% c(2, 3))

# Applying ROSE method
df3_1 <- ovun.sample(iphonesentiment~., data = df3_1, method = "over", N = 12572)$data
df3_2 <- ovun.sample(iphonesentiment~., data = df3_2, method = "over", N = 12572)$data

# Create a iphoneROSE df, with balanced data for iphonesentiment variable
df3_1 <- filter(df3_1, iphonesentiment %in% c(1))
df3_2 <- filter(df3_2, iphonesentiment %in% c(2))
ipTrainOver <- rbind(df3, df3_1, df3_2)
rm(df3, df3_1, df3_2)

ipTrainOver$iphonesentiment <- as.factor(ipTrainOver$iphonesentiment)
testOver$iphonesentiment <- as.factor(testOver$iphonesentiment)

saveRDS(ipTrainOver, file = "../../data/iphone/ipTrainOver.rds") 
saveRDS(testOver, file = "../../data/iphone/testOver.rds") 


### Adding PCA aproach to Over dataSet
preprocessParameters <- preProcess(iphoneOver[,-15], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParameters) # need 8 Components to capture 95% of output

# use predict to apply pca parameters, create training, exclude dependant
trainOver_pca <- predict(preprocessParameters, trainOver[,-15])
# add the dependent to training 
trainOver_pca$iphonesentiment <- trainOver$iphonesentiment

# use predict to apply pca parameters, create testing, exclude dependant
testOver_pca <- predict(preprocessParameters, testOver[,-15])
# add the dependent to test
testOver_pca$iphonesentiment <- testOver$iphonesentiment

# inspect results
str(trainOver_pca)
str(testOver_pca)
rm(preprocessParameters)

trainOver_pca$iphonesentiment <- as.factor(trainOver_pca$iphonesentiment)
testOver_pca$iphonesentiment <- as.factor(testOver_pca$iphonesentiment)

saveRDS(trainOver_pca, file = "../../data/trainOver_pca.rds") 
saveRDS(testOver_pca, file = "../../data/testOver_pca.rds")


####--------- Aproach count number of sentiment 

## Create a funtiont to count sentiment
iphone_matrix$sentCount <- rowSums(iphone_matrix[,-59])




## Apply to repeat the funtion for each row and create a column with the number of WAPs with good signal
data_full$num_good_signal <- apply(data_full[,c(11:530)], 1, sentCount)


# Compareing RF models  

models <- list(rfNZV = rfNZVip.m1,
               rfPCA = rfPCAip.m1,
               rfOver = rfOver,
               rfOverPCA = rfOverPCA)

resampling <- resamples(models)
bwplot(resampling)
