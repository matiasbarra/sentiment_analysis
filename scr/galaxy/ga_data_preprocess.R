## Project: Sentiment Analysis
## Author: Matias Barra
## Date: 03/02/2020
####------------------------------------ Set Environment and upload datasets --------------------####

## Packages 
pacman::p_load(caret, ggplot2, plotly, rstudioapi, corrplot, dplyr, corrplot,data.table,
               reshape,reshape2, randomForest, doParallel, kknn, C50, e1071, party)


loc   <- grep("ga_data_preprocess.R",list.files(recursive=TRUE),value=TRUE)
iloc  <- which(unlist(gregexpr("/ga_data_preprocess.R",loc)) != -1)
myloc <- paste(getwd(),loc[iloc],sep="/")
setwd(substr(myloc,1,nchar(myloc)-20))


# ## Github
# 
# current_path = rstudioapi::getActiveDocumentContext()$path # save working directory
# setwd(dirname(current_path))
# setwd("..")

## DataSets files 

galaxy_matrix <- read.csv("../../data/galaxy/galaxy_smallmatrix_labeled_8d.csv", header = TRUE)

####------------------------------------ Explore the data --------------------####

# Explorative plot
hist_ga_sent <- plot_ly(galaxy_matrix, x= ~galaxy_matrix$galaxysentiment, type='histogram')
plot_ly(galaxy_matrix, x= ~galaxy_matrix$galaxy, type='histogram')
# We can see daba inbalanced problems

# Check for NAs
apply(galaxy_matrix, 2, function(x) any(is.na(x))) # no NA in galaxy data

#### Reduce Sentiment to 3 categories ####
# I decided to use only 3 categories: negative (0 and 1), neutral (2 and 3) and positive (4 and 5). 

# recode sentiment to combine factor levels 0 & 1 and 4 & 5
galaxy_matrix$galaxysentiment <- recode(galaxy_matrix$galaxysentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 2, '4' = 3, '5' = 3) 

# Explorative plot
hist_ga_sent_1 <- plot_ly(galaxy_matrix, x= ~galaxy_matrix$iphonesentiment, type='histogram')

## Correlation Matrix only using iPhone data 
#select the vars that actually say something about the iphone
galaxy_unique <- galaxy_matrix %>% select(samsunggalaxy, googleandroid, samsungcampos, samsungcamneg, samsungcamunc, 
                                          samsungdispos, samsungdisneg, samsungdisunc, samsungperpos, 
                                          samsungperneg, samsungperunc, googleperpos, googleperneg, googleperunc,
                                          galaxysentiment)

# Using cor() and corpolot() to to identify correlated variables
# create a new data set and remove features highly correlated with the dependant 
galaxyCor <- cor(galaxy_unique)
corrplot(galaxyCor, type = "upper", tl.pos = "td",
         method = "square", tl.cex = 0.8, order= "hclust", tl.col = 'black', diag = TRUE)

# Create decision tree for galaxy
galaxyDtree <- ctree_control(maxdepth = 10)
galaxyDT <- ctree(galaxysentiment ~ ., data = galaxy_unique)
plot(galaxyDT)

# Linear Relationships are fairly weak as they are not strongly correlated
# Decision Tree highlights: googleandroid, googleperneg

#------- Analyzing Feature Variance

# nearZeroVar in Galaxy DataSet give us only 2 variables, this is why I decide avoide and go to PCA approach

####------- PCA (Principal Component Analysis) ----####
# removes all of your features and replaces them with mathematical representations of their variance
# data = training and testing from iphone_matrix (no feature selection) 
# create object containing centered, scaled PCA components from training set
# excluded the dependent variable and set threshold to 0.95

preprocessParams <- preProcess(galaxy_matrix[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParams) # need 25 Components to capture 95% of output

#Split the data
gaMatrixpartition <- createDataPartition(galaxy_matrix$galaxysentiment, times = 1, p = .7, list = FALSE)
gaMatrixtrain <- galaxy_matrix[gaMatrixpartition,]
gaMatrixtest <- galaxy_matrix[-gaMatrixpartition,]
gaMatrixtrain$galaxysentiment <- as.factor(gaMatrixtrain$galaxysentiment)
gaMatrixtest$galaxysentiment <- as.factor(gaMatrixtest$galaxysentiment)


# use predict to apply pca parameters, create training, exclude dependant
ga_train_pca <- predict(preprocessParams, gaMatrixtrain[,-59])
# add the dependent to training 
ga_train_pca$galaxysentiment <- gaMatrixtrain$galaxysentiment

# use predict to apply pca parameters, create testing, exclude dependant
ga_test_pca <- predict(preprocessParams, gaMatrixtest[,-59])
# add the dependent to test
ga_test_pca$galaxysentiment <- gaMatrixtest$galaxysentiment

# inspect results
str(ga_train_pca)
str(ga_test_pca)

saveRDS(ga_train_pca, file ="../../data/galaxy/ga_train_pca.rds")
saveRDS(ga_test_pca, file ="../../data/iphone/ga_test_pca.rds")




