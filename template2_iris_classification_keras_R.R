# iris classifcation by 'keras'
# Reference 2
# https://theoreticalecology.wordpress.com/2018/06/06/an-introduction-to-machine-learning-with-keras-in-r/
library(keras)
head(iris)
# index <- sample(1:nrow(iris), size = round(0.7*n), replace=FALSE)
index <- sample(nrow(iris))
data_train <- iris[index,]
data_test <- iris[-index, ]


y <- data_train[, "Species"]

x <- data_train[,1:4]

# scale to [0,1]
x <- as.matrix(apply(x, 2, function(x) (x-min(x))/(max(x) - min(x))))


# one hot encode classes / create DummyFeatures

levels(y) <- 1:length(y)
y <- to_categorical(as.integer(y) - 1 , num_classes = 3)
y


# create sequential model
model <- keras_model_sequential()


# add layers, first layer needs input dimension

model %>%
  layer_dense(input_shape = ncol(x), units = 10, activation = "relu") %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dense(units = 3, activation = "softmax")


# add a loss function and optimizer
model %>%
    compile(
    loss = "categorical_crossentropy",
    optimizer = "adagrad",
    metrics = "accuracy"
)

# fit model with our training data set, training will be done for 200 times data set
fit <- model %>%
    fit(
    x = x,
    y = y,
    shuffle = T,
    batch_size = 5,
    validation_split = 0.3,
    epochs = 200
)

plot(fit)

# add dropout to reduce overfitting
model <- keras_model_sequential()

model %>%
  
  layer_dense(input_shape = ncol(x), units = 10, activation = "relu") %>%
  
  layer_dropout(0.2) %>%
  
  layer_dense(units = 10, activation = "relu") %>%
  
  layer_dropout(0.2) %>%
  
  layer_dense(units = 3, activation = "softmax")

model %>%
  
  compile(
    
    loss = "categorical_crossentropy",
    
    optimizer = "adagrad",
    
    metrics = "accuracy"
    
  )

fit = model %>%
  
  fit(
    
    x = x,
    
    y = y,
    
    shuffle = T,
    
    validation_split = 0.3,
    
    epochs = 200,
    
    batch_size = 5
    
  )

plot(fit)

                   
                   