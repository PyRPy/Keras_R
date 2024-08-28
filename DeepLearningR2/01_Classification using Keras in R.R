
# Classification using Keras in R  ----------------------------------------
# example code from
# https://www.kaggle.com/code/yashgyy/deep-learning-with-r-keras-iris-dataset
library(keras)
library(caret)

### data prepparation
head(iris)

index<-createDataPartition(iris$Species,p=0.7,list=F)
Train_Features <- data.matrix(iris[index,-5]) # remove column 5
Train_Labels <- iris[index,5]
Test_Features <- data.matrix(iris[-index,-5])
Test_Labels <- iris[-index,5]

to_categorical(as.numeric(Train_Labels))[,c(-1)] -> Train_Labels
to_categorical(as.numeric(Test_Labels))[,c(-1)] -> Test_Labels

summary(Train_Labels)
str(Train_Features)


# normalize input
as.matrix(apply(Train_Features, 2, function(x) (x-min(x))/(max(x) - min(x)))) -> Train_Features
as.matrix(apply(Test_Features, 2, function(x) (x-min(x))/(max(x) - min(x)))) -> Test_Features

### construct model
model <- keras_model_sequential()
model %>%
  layer_dense(units=10,activation = "relu",input_shape = ncol(Train_Features)) %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dense(units = 3, activation = "softmax")

summary(model)

### define the hyper parameters
model %>% compile(loss = "categorical_crossentropy",
                  optimizer = optimizer_adagrad(),
                  metrics = c('accuracy')
)

### run the model
history <- model %>% fit(Train_Features,Train_Labels,validation_split = 0.10,epochs=300,batch_size = 5,shuffle = T)


### plot and evaluate on the test data
plot(history)
model %>% evaluate(Test_Features,Test_Labels)
