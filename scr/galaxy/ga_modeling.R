## Project: Sentiment Analysis
## Author: Matias Barra
## Date: 03/02/2020
## DModeling Script
####------------------------------------ Set Environment and upload datasets --------------------####

## Packages 
pacman::p_load(caret, ggplot2, plotly, rstudioapi, corrplot, dplyr, corrplot,data.table,
               reshape,reshape2, randomForest, doParallel, kknn, C50, e1071, party, ROSE)



# Reading dataSets
ga_train_pca <- readRDS("../../data/galaxy/ga_train_pca.rds")
ga_test_pca <- readRDS("../../data/galaxy/ga_test_pca.rds")
# ipTrainOver <- readRDS("../../data/iphone/ipTrainOver.rds") 
# testOver <- readRDS("../../data/iphone/testOver.rds") 
# trainOver_pca <- readRDS("../../data/trainOver_pca.rds") 
# testOver_pca <- readRDS("../../data/testOver_pca.rds")

## Setup Paralell Programming 
detectCores() # 4 cores available
cl <- makeCluster(3) # create cluster keeping 1 core to operative system
registerDoParallel(cl) # Register cluster
getDoParWorkers() # check if there are now 3 cores working



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


####------------------------------------------ Models with PCA aproach ------####

# Model 1 / RF with RF package
mtry_rfPCAgalax <- tuneRF(ga_train_pca[,-26], ga_train_pca[,26], 
                           ntreeTry = 100, stepFactor = 2, improve = 0.05, trace = TRUE, plot = TRUE)

rfPCAga.m1 <- randomForest(y = ga_train_pca[,26], x = ga_train_pca[,-26], 
                           importance = T, ntree = 100, mtry = 5, trControl = control)
saveRDS(rfPCAga.m1, file = "../../models/galaxy/rfPCAga.m1.rds")
# rfPCAga.m1 <- readRDS("../../models/galaxy/rfPCAga.m1.rds") 
rm(mtry_rfPCAgalax)

# Model 2 / RF with caret package
rfPCAga.m2 <- caret::train(galaxysentiment~., data = ga_train_pca, 
                           method = "rf", trControl=control, tuneLength = 2)
saveRDS(rfPCAga.m2, file = "../../models/galaxy/rfPCAga.m2.rds")
# rfPCAga.m2 <- readRDS("../../models/galaxy/rfPCAga.m2.rds")

# Model 3 / KNN with train.knn 
kknnPCAga.m3 <- train.kknn(formula = galaxysentiment~., data = ga_train_pca, kmax = 11,
                           distance = 2, kernel = "optimal", trControl = control, scale = T, center = T)
saveRDS(kknnPCAga.m3, file = "../../models/galaxy/kknnPCAga.m3.rds")
# kknnPCAga.m3 <- readRDS("../../models/galaxy/kknnPCAga.m3.rds")

# Model 4 / SVM (from the e1071 package) 
svmPCAga.m4 <- svm(formula = galaxysentiment~., data = ga_train_pca, trControl = control, scale = T, center = T)
saveRDS(svmPCAga.m4, file = "../../models/galaxy/svmPCAga.m4.rds")
# svmPCAga.m4 <- readRDS("../../models/galaxy/svmPCAga.m4.rds")  

## Test the models
pred_rfPCAga.m1 <- predict(rfPCAga.m1, ga_test_pca)
pred_rfPCAga.m2 <- predict(rfPCAga.m2, ga_test_pca)
pred_kknnPCAga.m3 <- predict(kknnPCAga.m3, ga_test_pca)
pred_svmPCAga.m4 <- predict(svmPCAga.m4, ga_test_pca)

## PostResamples
postR_rfPCAga.m1 <- as.data.frame(postResample(pred = pred_rfPCAga.m1, obs = ga_test_pca$galaxysentiment))
postR_rfPCAga.m2 <- as.data.frame(postResample(pred = pred_rfPCAga.m2, obs = ga_test_pca$galaxysentiment))
postR_kknnPCAga.m3 <- as.data.frame(postResample(pred = pred_kknnPCAga.m3, obs = ga_test_pca$galaxysentiment))
postR_svmPCAga.m4 <- as.data.frame(postResample(pred = pred_svmPCAga.m4, obs = ga_test_pca$galaxysentiment))

resultsPCAclass <- cbind(postR_rfPCAga.m1, postR_rfPCAga.m2, postR_kknnPCAga.m3, postR_svmPCAga.m4)
colnames(resultsPCAclass) <- c("RF_RFPack", "RF_Caret", "KKNN", "SVM")



# Stop Cluster. After performing your tasks, stop your cluster. 
stopCluster(cl)


