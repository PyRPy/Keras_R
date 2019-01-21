# iris data set classification by 'keras'
# Reference 
# https://www.kaggle.com/yashgyy/deep-learning-with-r-keras-iris-dataset

library(keras)

library(caret)

head(iris)

# Prepare 'training' and 'testing' sets
index<-unlist(createDataPartition(iris$Species,p=0.7,list=TRUE)) # unlist the index

Train_Features <- data.matrix(iris[index,-5])
Train_Labels <- iris[index,5]

Test_Features <- data.matrix(iris[-index,-5])
Test_Labels <- iris[-index,5]

length(index) # check how many data in training set

# convert to categorical
to_categorical(as.numeric(Train_Labels))[,c(-1)] -> Train_Labels
to_categorical(as.numeric(Test_Labels))[,c(-1)] -> Test_Labels

summary(Train_Labels)
str(Train_Features)

# scale the features
as.matrix(apply(Train_Features, 2, function(x) (x-min(x))/(max(x) - min(x)))) -> Train_Features
as.matrix(apply(Test_Features, 2, function(x) (x-min(x))/(max(x) - min(x)))) -> Test_Features

# Prepare the model
model <- keras_model_sequential()

model %>%
  layer_dense(units=10,activation = "relu",input_shape = ncol(Train_Features)) %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dense(units = 3, activation = "softmax")

# check the model configuration
summary(model)

# set up the model parameters
model %>% compile(loss = "categorical_crossentropy",
                  optimizer = optimizer_adagrad(),
                  metrics = c('accuracy')
)

# run the model
history <- model %>% fit(Train_Features,Train_Labels,
                         validation_split = 0.10,
                         epochs=100,  # 100 iterations enough
                         batch_size = 5,
                         shuffle = T)

# plot the history
plot(history)

# model evaluation
model %>% evaluate(Test_Features,Test_Labels)

