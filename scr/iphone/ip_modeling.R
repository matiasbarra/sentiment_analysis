## Project: Sentiment Analysis
## Author: Matias Barra
## Date: 03/02/2020
## DModeling Script
####------------------------------------ Set Environment and upload datasets --------------------####

## Packages 
pacman::p_load(caret, ggplot2, plotly, rstudioapi, corrplot, dplyr, corrplot,data.table,
               reshape,reshape2, randomForest, doParallel, kknn, C50, e1071, party, ROSE)



# Reading dataSets
iphoneNZV <- readRDS("../../data/iphone/iphoneNZV.rds")
train_pca <- readRDS("../../data/iphone/train_pca.rds")
test_pca <- readRDS("../../data/iphone/test_pca.rds")
ipTrainOver <- readRDS("../../data/iphone/ipTrainOver.rds") 
testOver <- readRDS("../../data/iphone/testOver.rds") 
trainOver_pca <- readRDS("../../data/trainOver_pca.rds") 
testOver_pca <- readRDS("../../data/testOver_pca.rds")

####-------------------------------------------------------- Train Models --------------------####
set.seed(123)
###
# Set control for models
control <- trainControl(method = "repeatedcv", 
                        number = 10, 
                        repeats = 3,
                        allowParallel = TRUE,
                        returnData = T)


## Setup Paralell Programming 
detectCores() # 4 cores available
cl <- makeCluster(3) # create cluster keeping 1 core to operative system
registerDoParallel(cl) # Register cluster
getDoParWorkers() # check if there are now 3 cores working



####------------------------------------------ Models with NZV dataset ------####

# split the data 
ipNZVpartition <- createDataPartition(iphoneNZV$iphonesentiment, times = 1, p = .7, list = FALSE)
ipNZVtrain <- iphoneNZV[ipNZVpartition,]
ipNZVtest <- iphoneNZV[-ipNZVpartition,]
ipNZVtrain$iphonesentiment <- as.factor(ipNZVtrain$iphonesentiment)
ipNZVtest$iphonesentiment <- as.factor(ipNZVtest$iphonesentiment)

# Model 1 / RF with RF package
mtry_rfNZViphone <- tuneRF(ipNZVtrain[,-10], ipNZVtrain[,10], 
                           ntreeTry = 100, stepFactor = 2, improve = 0.05, trace = TRUE, plot = TRUE)

rfNZVip.m1 <- randomForest(y = ipNZVtrain[,10], x = ipNZVtrain[,-10], 
                           importance = T, ntree = 100, mtry = 2, trControl = control)
saveRDS(rfNZVip.m1, file = "../../models/iphone/rfNZVip.m1.rds")
rfNZVip.m1 <- readRDS("../../models/iphone/rfNZVip.m1.rds") 

# Model 2 / RF with caret package
rfNZVip.m2 <- caret::train(iphonesentiment~., data = ipNZVtrain, 
                           method = "rf", trControl=control, tuneLength = 2)
saveRDS(rfNZVip.m2, file = "../../models/iphone/rfNZVip.m2.rds")
rfNZVip.m2 <- readRDS("../../models/iphone/rfNZVip.m2.rds") 


# Model 3 / KNN with train.knn 
kknnNZVip.m3 <- train.kknn(formula = iphonesentiment~., data = ipNZVtrain, kmax = 11,
                           distance = 2, kernel = "optimal", trControl = control, scale = T, center = T)
saveRDS(kknnNZVip.m3, file = "../../models/iphone/kknnNZVip.m3.rds")
kknnNZVip.m3 <- readRDS("../../models/iphone/kknnNZVip.m3.rds") 

# Model 4 / SVM (from the e1071 package) 
svmNZVip.m4 <- svm(formula = iphonesentiment~., data = ipNZVtrain, trControl = control, scale = T, center = T)
saveRDS(svmNZVip.m4, file = "../../models/iphone/svmNZVip.m4.rds")
svmNZVip.m4 <- readRDS("../../models/iphone/svmNZViphone_m4.rds")  

## Test the models
pred_rfNZVip.m1 <- predict(rfNZVip.m1,ipNZVtest)
pred_rfNZVip.m2 <- predict(rfNZVip.m2,ipNZVtest)
pred_kknnNZVip.m3 <- predict(kknnNZVip.m3,ipNZVtest)
pred_svmNZVip.m4 <- predict(svmNZVip.m4,ipNZVtest)

## Postresamples 
postR_rfNZVip.m1 <- as.data.frame(postResample(pred = pred_rfNZVip.m1, obs = ipNZVtest$iphonesentiment))
postR_rfNZVip.m2 <- as.data.frame(postResample(pred = pred_rfNZVip.m2, obs = ipNZVtest$iphonesentiment))
postR_kknnNZVip.m3 <- as.data.frame(postResample(pred = pred_kknnNZVip.m3, obs = ipNZVtest$iphonesentiment))
postR_svmNZVip.m4 <- as.data.frame(postResample(pred = pred_svmNZVip.m4, obs = ipNZVtest$iphonesentiment))

resultsNZVclass <- cbind(postR_rfNZVip.m1, postR_rfNZVip.m2, postR_kknnNZVip.m3, postR_svmNZVip.m4)
colnames(resultsNZVclass) <- c("RF_RFPack", "RF_Caret", "KKNN", "SVM")

# Create a confusion matrix from  predictions NZV
cm_rfNZVip.m1 <- confusionMatrix(pred_rfNZVip.m1, ipNZVtest$iphonesentiment) 
cm_rfNZVip.m2 <- confusionMatrix(pred_rfNZVip.m2, ipNZVtest$iphonesentiment) 
cm_kknnNZVip.m3 <- confusionMatrix(pred_kknnNZVip.m3, ipNZVtest$iphonesentiment) 
cm_svmNZVip.m4 <- confusionMatrix(pred_svmNZVip.m4, ipNZVtest$iphonesentiment) 


####------------------------------------------ Models with PCA aproach ------####

# Model 1 / RF with RF package
mtry_rfPCAiphone <- tuneRF(train_pca[,-26], train_pca[,26], 
                           ntreeTry = 100, stepFactor = 2, improve = 0.05, trace = TRUE, plot = TRUE)

rfPCAip.m1 <- randomForest(y = train_pca[,26], x = train_pca[,-26], 
                           importance = T, ntree = 100, mtry = 10, trControl = control)
saveRDS(rfPCAip.m1, file = "../../models/iphone/rfPCAip.m1.rds")
rfPCAip.m1 <- readRDS("../../models/iphone/rfPCAip.m1.rds") 


# Model 2 / RF with caret package
rfPCAip.m2 <- caret::train(iphonesentiment~., data = train_pca, 
                           method = "rf", trControl=control, tuneLength = 2)
saveRDS(rfPCAip.m2, file = "../../models/iphone/rfPCAip.m2.rds")
rfPCAip.m2 <- readRDS("../../models/iphone/rfPCAip.m2.rds")

# Model 3 / KNN with train.knn 
kknnPCAip.m3 <- train.kknn(formula = iphonesentiment~., data = train_pca, kmax = 11,
                           distance = 2, kernel = "optimal", trControl = control, scale = T, center = T)
saveRDS(kknnPCAip.m3, file = "../../models/iphone/kknnPCAip.m3.rds")
kknnPCAip.m3 <- readRDS("../../models/iphone/kknnPCAip.m3.rds")

# Model 4 / SVM (from the e1071 package) 
svmPCAip.m4 <- svm(formula = iphonesentiment~., data = train_pca, trControl = control, scale = T, center = T)
saveRDS(svmPCAip.m4, file = "../../models/iphone/svmPCAip.m4.rds")
svmPCAip.m4 <- readRDS("../../models/iphone/svmPCAip.m4.rds")  

## Test the models
pred_rfPCAip.m1 <- predict(rfPCAip.m1, test_pca)
pred_rfPCAip.m2 <- predict(rfPCAip.m2, test_pca)
pred_kknnPCAip.m3 <- predict(kknnPCAip.m3, test_pca)
pred_svmPCAip.m4 <- predict(svmPCAip.m4, test_pca)

## PostResamples
postR_rfPCAip.m1 <- as.data.frame(postResample(pred = pred_rfPCAip.m1, obs = test_pca$iphonesentiment))
postR_rfPCAip.m2 <- as.data.frame(postResample(pred = pred_rfPCAip.m2, obs = test_pca$iphonesentiment))
postR_kknnPCAip.m3 <- as.data.frame(postResample(pred = pred_kknnPCAip.m3, obs = test_pca$iphonesentiment))
postR_svmPCAip.m4 <- as.data.frame(postResample(pred = pred_svmPCAip.m4, obs = test_pca$iphonesentiment))

resultsPCAclass <- cbind(postR_rfPCAip.m1, postR_rfPCAip.m2, postR_kknnPCAip.m3, postR_svmPCAip.m4)
colnames(resultsPCAclass) <- c("RF_RFPack", "RF_Caret", "KKNN", "SVM")


####------------------------------------------ Models Oversampling data ------####

#Training Models whit balanced data frame
# Model 1 / RF with RF package
mtryOver <- tuneRF(ipTrainOver[,-15], ipTrainOver[,15], 
                   ntreeTry = 100, stepFactor = 2, improve = 0.05, trace = TRUE, plot = TRUE)
rm(mtryOver)

rfOver <- randomForest(y = ipTrainOver[,15], x = ipTrainOver[,-15], 
                       importance = T, ntree = 100, mtry = 6, trControl = control)
saveRDS(rfOver, file = "../../models/iphone/rfOver.rds")

# Model 2 / RF with caret package
rfOver.m2 <- caret::train(iphonesentiment~., data = ipTrainOver, 
                           method = "rf", trControl= control, tuneLength = 2)
saveRDS(rfOver.m2, file = "../../models/iphone/rfOver.m2.rds")
rfOver.m2 <- readRDS("../../models/iphone/rfOver.m2.rds")


## Test the models
pred_rfOver<- predict(rfOver, testOver)
pred_rfOver.m2 <- predict(rfOver.m2, testOver)

## PostResamples
postR_rfOver <- as.data.frame(postResample(pred = pred_rfOver, obs = testOver$iphonesentiment))
postR_rfOver.m2 <- as.data.frame(postResample(pred = pred_rfOver.m2, obs = testOver$iphonesentiment))

#Confusion Matrix
rfOver_cm <- confusionMatrix(pred_rfOver, testOver$iphonesentiment)
rfOver.m2cm <- confusionMatrix(pred_rfOver.m2, testOver$iphonesentiment)x

resultsOver <- cbind(postR_rfOver, postR_rfOver.m2)
colnames(resultsOver) <- c("RF_RFPack", "RF_Caret")


####------------------------------------------ Models Oversampling data + PCA ------####

# Model 1 / RF with RF package
mtryOverPCA <- tuneRF(trainOver_pca[,-9], trainOver_pca[,9], 
                      ntreeTry = 100, stepFactor = 2, improve = 0.05, trace = TRUE, plot = TRUE)
rm(mtryOverPCA)

rfOverPCA <- randomForest(y = trainOver_pca[,9], x = trainOver_pca[,-9], 
                          importance = T, ntree = 100, mtry = 1, trControl = control)
saveRDS(rfOverPCA, file = "../../models/iphone/rfOverPCA.rds")


# Model 2 / RF with caret package
rfOverPCA.m2 <- caret::train(iphonesentiment~., data = trainOver_pca, 
                          method = "rf", trControl= control, tuneLength = 2)
saveRDS(rfOverPCA.m2, file = "../../models/iphone/rfOverPCA.m2.rds")
rfOverPCA.m2 <- readRDS("../../models/iphone/rfOverPCA.m2.rds")

## Test the models
pred_rfOverPCA <- predict(rfOverPCA, testOver_pca)
pred_rfOverPCA.m2 <- predict(rfOverPCA.m2, testOver_pca)

## PostResamples
postR_rfOverPCA <- as.data.frame(postResample(pred = pred_rfOverPCA, obs = testOver_pca$iphonesentiment))
postR_rfOverPCA.m2 <- as.data.frame((postResample(pred = pred_rfOverPCA.m2, obs = testOver_pca$iphonesentiment)))

#Confusion Matrix
rfOverPCA_cm <- confusionMatrix(pred_rfOverPCA, testOver_pca$iphonesentiment)
rfOverPCA.m2 <- confusionMatrix(pred_rfOverPCA.m2, testOver_pca$iphonesentiment)

resultsOverPCA <- cbind(postR_rfOverPCA, postR_rfOverPCA.m2)
colnames(resultsOverPCA) <- c("RF_RFPack", "RF_Caret")


# Stop Cluster. After performing your tasks, stop your cluster. 
stopCluster(cl)

####------------------------------------------ Results ------####

# Joint NZV and PCA Results 
resultsNZVclass$approach <- "NZV"
resultsPCAclass$approach <- "PCA"
resultsOver$approach <- "Over"
resultsOverPCA$approach <- "OverPCA"

ResultsNZV_PCA <- rbind(resultsNZVclass, resultsPCAclass)
ResultsOver_OverPCA <- rbind(resultsOver, resultsOverPCA)

# Looking the "ResultsNZV_PCA"

Results_Kappa <- ResultsNZV_PCA[-c(1,3),]
Results_Accuracy <- ResultsNZV_PCA[-c(2,4),]

Res_Kappa_melt <- melt(Results_Kappa)
Res_Acc_melt <- melt(Results_Accuracy)

# Near-Zero-Variance approach delivers better results
p.kappa <- ggplotly(ggplot(Res_Kappa_melt, aes(x = variable,y=value,fill=variable)) + geom_bar(stat = "identity")+facet_wrap(~approach)+
                      coord_flip()+ggtitle("Kappa Comparison between NZV and PCA Approach"))
p.accuracy <- ggplotly(ggplot(Res_Acc_melt, aes(x = variable,y=value,fill=variable)) + geom_bar(stat = "identity")+facet_wrap(~approach)+
                         coord_flip()+ggtitle("Accuracy Comparison between NZV and PCA Approach"))

