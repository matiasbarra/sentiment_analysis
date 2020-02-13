## Project: Sentiment Analysis
## Author: Matias Barra
## Date: 03/02/2020
####------------------------------------ Set Environment and upload datasets --------------------####

## Packages 
pacman::p_load(caret, ggplot2, plotly, rstudioapi, corrplot, dplyr, corrplot,data.table,
               reshape,reshape2, randomForest, doParallel, kknn, C50, e1071, party, ggpubr, magrittr)

loc   <- grep("data_preprocess.R",list.files(recursive=TRUE),value=TRUE)
iloc  <- which(unlist(gregexpr("/ip_data_preprocess.R",loc)) != -1)
myloc <- paste(getwd(),loc[iloc],sep="/")
setwd(substr(myloc,1,nchar(myloc)-20))

## Setup Paralell Programming 
detectCores() # 4 cores available
cl <- makeCluster(3) # create cluster keeping 1 core to operative system
registerDoParallel(cl) # Register cluster
getDoParWorkers() # check if there are now 3 cores working

## DataSets files 
large_matrix <- read.csv("../../data/LargeMatrix.csv", header = TRUE)
large_matrix <- large_matrix[, -which(names(large_matrix) %in% c("X", "id"))]
iphone_matrix <- read.csv("../../data/iphone/iphone_smallmatrix_labeled_8d.csv")
galaxy_matrix <- read.csv("../../data/galaxy/galaxy_smallmatrix_labeled_8d.csv")

## Recode sentiment to combine factor levels 0 & 1, 2 & 3 and 4 & 5
iphone_matrix$iphonesentiment <- dplyr::recode(iphone_matrix$iphonesentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 2, '4' = 3, '5' = 3) 
galaxy_matrix$galaxysentiment <- dplyr::recode(galaxy_matrix$galaxysentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 2, '4' = 3, '5' = 3)


### Split the data  
# iphone
ipMatrixpartition <- createDataPartition(iphone_matrix$iphonesentiment, times = 1, p = .7, list = FALSE)
iphoneTrain <- iphone_matrix[ipMatrixpartition,]
iphoneTest <- iphone_matrix[-ipMatrixpartition,]

# galaxy
gaMatrixpartition <- createDataPartition(galaxy_matrix$galaxysentiment, times = 1, p = .7, list = FALSE)
galaxyTrain <- galaxy_matrix[gaMatrixpartition,]
galaxyTest <- galaxy_matrix[-gaMatrixpartition,]

####---------------------------- PCA (Principal Component Analysis) --------------------------------####
## iphone
ipPreprocessParams <- preProcess(iphoneTrain[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(ipPreprocessParams) # need 25 Components to capture 95% of output

## galaxy
gaPreprocessParam <- preProcess(galaxyTrain[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(gaPreprocessParams) # need 24 Components to capture 95% of output


# use predict to apply pca parameters, create training, exclude dependant
ipTrain.pca <- predict(ipPreprocessParams, iphoneTrain[,-59])
gaTrain.pca <- predict(gaPreprocessParams, galaxyTrain[,-59])

# add the dependent to training
ipTrain.pca$iphonesentiment <- iphoneTrain$iphonesentiment
gaTrain.pca$galaxysentiment <- galaxyTrain$galaxysentiment

# use predict to apply pca parameters, create testing, exclude dependant
ipTest.pca <- predict(ipPreprocessParams, iphoneTest[,-59])
gaTest.pca <- predict(gaPreprocessParams, galaxyTest[,-59])

ipLarge.pca <- predict(ipPreprocessParams, large_matrix)
gaLarge.pca <- predict(gaPreprocessParams, large_matrix)

ipLarge.pca$iphonesentiment <- NULL
gaLarge.pca$iphonesentiment <- NULL

# add the dependent to test
ipTest.pca$iphonesentiment <- iphoneTest$iphonesentiment
gaTest.pca$galaxysentiment <- galaxyTest$galaxysentiment


# inspect results
str(ipTrain.pca)
str(ipTest.pca)
str(ipLarge.pca)

str(gaTrain.pca)
str(gaTest.pca)
str(gaLarge.pca)

ipTrain.pca$iphonesentiment <- as.factor(ipTrain.pca$iphonesentiment)
gaTrain.pca$galaxysentiment <- as.factor(gaTrain.pca$galaxysentiment)

# Model 1 / RF with RF package
## iphone
mtry_ipTrain.pca <- tuneRF(ipTrain.pca[,-26], ipTrain.pca[,26], 
                           ntreeTry = 100, stepFactor = 2, improve = 0.05, trace = TRUE, plot = TRUE)

rfPCAip <- randomForest(y = ipTrain.pca[,26], x = ipTrain.pca[,-26], 
                           importance = T, ntree = 100, mtry = 10, trControl = control)
saveRDS(rfPCAip, file = "../../models/iphone/rfPCAip.rds")
rfPCAip.m1 <- readRDS("../../models/iphone/rfPCAip.rds") 

## galaxy
mtry_gaTrain.pca <- tuneRF(gaTrain.pca[,-25], gaTrain.pca[,25], 
                           ntreeTry = 100, stepFactor = 2, improve = 0.05, trace = TRUE, plot = TRUE)

rfPCAga <- randomForest(y = gaTrain.pca[,25], x = gaTrain.pca[,-25], 
                        importance = T, ntree = 100, mtry = 4, trControl = control)
saveRDS(rfPCAga, file = "../../models/iphone/rfPCAga.rds")
rfPCAga <- readRDS("../../models/iphone/rfPCAga.rds") 

# Review the final model and results 
rfPCAip
rfPCAga

# Predict outcomes with the test data
#iPhone
pred_rfPCAip <-  as.data.frame(predict(rfPCAip,ipTest.pca, reshape = T))
colnames(pred_rfPCAip) <- "ipPredictions"
#samsung
pred_rfPCAga <-  as.data.frame(predict(rfPCAga,gaTest.pca, reshape = T))
colnames(pred_rfPCAip) <- "gaPredictions"

pred_LargeIphone <- as.data.frame(predict(rfPCAip, ipLarge.pca, reshape = T))
colnames(pred_LargeIphone) <- "iphonePredictions"
pred_LargeGalaxy <- as.data.frame(predict(rfPCAga, gaLarge.pca, reshape = T))
colnames(pred_LargeGalaxy) <- "galaxyPredictions"

# Recode predictions Results
pred_LargeIphone$iphonePredictions <- dplyr::recode(pred_LargeIphone$iphonePredictions, 
                                                    '1' = "Negative", '2' = "Neutral", '3' = "Positive") 

pred_LargeGalaxy$galaxyPredictions <- dplyr::recode(pred_LargeGalaxy$galaxyPredictions, 
                                                    '1' = "Negative", '2' = "Neutral", '3' = "Positive") 

## Prediction Plots
#iPhone
predIphone <- as.data.frame(pred_LargeIphone %>% group_by(iphonePredictions) %>% summarise(counts = n()))

ggplot(predIphone, aes(x = iphonePredictions, y = counts)) +
  geom_bar(fill = "#5986a9", stat = "identity") +
  ggtitle("Sentiment Predictions for Iphone") +
  geom_text(aes(label = counts), vjust = -0.3) + theme_pubclean()

predIphone_1 <- predIphone %>%
  arrange(desc(iphonePredictions)) %>%
  mutate(prop = round(counts*100/sum(counts), 1),
         lab.ypos = cumsum(prop) - 0.5*prop)

plot_ly(predIphone_1, labels = ~iphonePredictions, values = ~prop, type = 'pie') %>%
  layout(title = 'Sentiment Predictions For Iphone',
         xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
         yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))


# galaxy
predGalaxy <- as.data.frame(pred_LargeGalaxy %>% group_by(galaxyPredictions) %>% summarise(counts = n()))

ggplot(predGalaxy, aes(x = galaxyPredictions, y = counts)) +
  geom_bar(fill = "#599bbc", stat = "identity") +
  ggtitle("Sentiment Predictions for Samsung Galaxy") +
  geom_text(aes(label = counts), vjust = -0.3) + theme_pubclean()

predGalaxy_1 <- predGalaxy %>%
  arrange(desc(galaxyPredictions)) %>%
  mutate(prop = round(counts*100/sum(counts), 1),
         lab.ypos = cumsum(prop) - 0.5*prop)

plot_ly(predGalaxy_1, labels = ~galaxyPredictions, values = ~prop, type = 'pie') %>%
  layout(title = 'Sentiment Predictions For Iphone',
         xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
         yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))

# Stop Cluster. After performing your tasks
stopCluster(cl)

