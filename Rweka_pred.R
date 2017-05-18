library(RWekajars)
library(RWeka)
library(rJava)
library(graphics)
library(stats)
library(utils)
library(grid)
library(partykit)
library(mlbench)
library(e1071)
library(foreign)

#example from system
w <- read.arff("E:/software/Program/Weka/Weka-3-8/data/weather.nominal.arff" )

#idendify a decision tree
m <- J48(play~., data=w)

#use 10 fold cross-validation
e <- evaluate_Weka_classifier(m, cost = matrix(c(0,2,1,0), ncol = 2),
                              numFolds = 10, complexity = TRUE,
                              seed = 123, class = TRUE)

WOW("J48") #weka control parameters

#example of JRip 
library(caret)
library(RWeka)
data(iris)
TrainData <- iris[,1:4]
TrainClasses <- iris[,5]
jripFit <- train(TrainData, TrainClasses,method = "JRip")
jripFit <- train(TrainData, TrainClasses,method = "JRip",preProcess = c("center", "scale"),
                 tuneLength = 10,trControl = trainControl(method = "cv"))
