# keras: Deep Learning in R
# https://www.datacamp.com/community/tutorials/keras-r-deep-learning
# load library ------------------------------------------------------------

library(keras)


# Reading Data From Files -------------------------------------------------

iris <- read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"), 
                 header = FALSE) 
# write.csv(iris, "iris.csv")
head(iris)
str(iris)
dim(iris)


# Data Exploration --------------------------------------------------------

names(iris) <- c("Sepal.Length", "Sepal.Width", "Petal.Length", 
                 "Petal.Width", "Species")

plot(iris$Petal.Length, 
     iris$Petal.Width, 
     pch=21, bg=c("red","green3","blue")[unclass(iris$Species)], 
     xlab="Petal Length", 
     ylab="Petal Width")

head(iris)

# Overall correlation between `Petal.Length` and `Petal.Width` 
cor(iris$Petal.Length, iris$Petal.Width)

# Store the overall correlation in `M`
M <- cor(iris[,1:4])

# Plot the correlation plot with `M`
library(corrplot)
corrplot(M, method="circle")


# Data Preprocessing ------------------------------------------------------

# Pull up a summary of `iris`
summary(iris)

# Inspect the structure of `iris`
str(iris)

# Normalizing Your Data With A User Defined Function (UDF)
# Build your own `normalize()` function
normalize <- function(x){
  num <- x - min(x)
  denom <- max(x) - min(x)
  return (num/denom)
}

# normalize the data
iris[,1:4] <- as.data.frame(lapply(iris[1:4], normalize))

# return the first part of data
head(iris)

# Normalize Your Data With keras
iris[,5] <- as.numeric(iris[,5]) -1

# convert 'iris' into a matrix
iris <- as.matrix(iris)

# set 'iris' dimname to NULL
dimnames(iris) <- NULL

head(iris)

# normalize the 'iris' data
# iris <- normalize(iris[, 1:4])

# return summary of 'iris'
summary(iris)

# Training And Test Sets --------------------------------------------------

# Determine sample size
ind <- sample(2, nrow(iris), replace=TRUE, prob=c(0.67, 0.33))

# Split the `iris` data
iris.training <- iris[ind==1, 1:4]
iris.test <- iris[ind==2, 1:4]

# Split the class attribute
iris.trainingtarget <- iris[ind==1, 5]
iris.testtarget <- iris[ind==2, 5]

# One-Hot Encoding
# One hot encode training target values
iris.trainLabels <- to_categorical(iris.trainingtarget)

# One hot encode test target values
iris.testLabels <- to_categorical(iris.testtarget)

# Print out the iris.testLabels to double check the result
print(iris.testLabels)


# Constructing the Model --------------------------------------------------

# Initialize a sequential model
model <- keras_model_sequential()

# Add layers to the model
model %>% 
  layer_dense(units = 8, activation = 'relu', input_shape = c(4)) %>% 
  layer_dense(units = 3, activation = 'softmax')

# Print a summary of a model
summary(model)

# Get model configuration
get_config(model)

# Get layer configuration
get_layer(model, index = 1)

# List the model's layers
model$layers

# List the input tensors
model$inputs


# Compile And Fit The Model -----------------------------------------------

# Compile the model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

# Fit the model 
model %>% fit(
  iris.training, 
  iris.trainLabels, 
  epochs = 100, 
  batch_size = 5, 
  validation_split = 0.2
)


# Visualize The Model Training History ------------------------------------
# Store the fitting history in `history` 
history <- model %>% fit(
  iris.training, 
  iris.trainLabels, 
  epochs = 200,
  batch_size = 5, 
  validation_split = 0.2
)

# Plot the history
plot(history)

# Plot the model loss of the training data
plot(history$metrics$loss, main="Model Loss", 
     xlab = "epoch", ylab="loss", col="blue", type="l",
     ylim = c(0, 1.0))

# Plot the model loss of the test data
lines(history$metrics$val_loss, col="green")

# Add legend
legend("topright", c("train","test"), col=c("blue", "green"), lty=c(1,1))

# Plot the accuracy of the training data 
plot(history$metrics$acc, main="Model Accuracy", 
     xlab = "epoch", ylab="accuracy", col="blue", type="l",
     ylim = c(0, 1))

# Plot the accuracy of the validation data
lines(history$metrics$val_acc, col="green")

# Add Legend
legend("bottomright", c("train","test"), col=c("blue", "green"), lty=c(1,1))


# Predict Labels of New Data ----------------------------------------------
# Predict the classes for the test data
classes <- model %>% 
  predict_classes(iris.test, batch_size = 128)

# Confusion matrix
table(iris.testtarget, classes)


# Evaluating Your Model ---------------------------------------------------
# Evaluate on test data and labels
score <- model %>% evaluate(iris.test, iris.testLabels, batch_size = 128)

# Print the score
print(score)


# Fine-tuning Your Model --------------------------------------------------
# Adding Layers

# Initialize the sequential model
model <- keras_model_sequential() 

# Add layers to model
model %>% 
  layer_dense(units = 8, activation = 'relu', input_shape = c(4)) %>% 
  layer_dense(units = 5, activation = 'relu') %>% 
  layer_dense(units = 3, activation = 'softmax')

# Compile the model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

# Fit the model to the data
model %>% fit(
  iris.training, iris.trainLabels, 
  epochs = 200, batch_size = 5, 
  validation_split = 0.2
)

# Evaluate the model
score <- model %>% evaluate(iris.test, iris.testLabels, batch_size = 128)

# Print the score
print(score)

# Hidden Units
# Initialize a sequential model
model <- keras_model_sequential() 

# Add layers to the model
model %>% 
  layer_dense(units = 28, activation = 'relu', input_shape = c(4)) %>% 
  layer_dense(units = 3, activation = 'softmax')

# Compile the model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

# Fit the model to the data
model %>% fit(
  iris.training, iris.trainLabels, 
  epochs = 200, batch_size = 5, 
  validation_split = 0.2
)

# Evaluate the model
score <- model %>% evaluate(iris.test, iris.testLabels, batch_size = 128)

# Print the score
print(score)

# Optimization Parameters
# Initialize a sequential model
model <- keras_model_sequential() 

# Build up your model by adding layers to it
model %>% 
  layer_dense(units = 8, activation = 'relu', input_shape = c(4)) %>% 
  layer_dense(units = 3, activation = 'softmax')

# Define an optimizer
sgd <- optimizer_sgd(lr = 0.01)

# Use the optimizer to compile the model
model %>% compile(optimizer=sgd, 
                  loss='categorical_crossentropy', 
                  metrics='accuracy')

# Fit the model to the training data
model %>% fit(
  iris.training, iris.trainLabels, 
  epochs = 200, batch_size = 5, 
  validation_split = 0.2
)

# Evaluate the model
score <- model %>% evaluate(iris.test, iris.testLabels, batch_size = 128)

# Print the loss and accuracy metrics
print(score)


# Saving, Loading or Exporting Your Model ---------------------------------

save_model_hdf5(model, "my_model.h5")
model2 <- load_model_hdf5("my_model.h5")

score <- model2 %>% evaluate(iris.test, iris.testLabels, batch_size = 128)
print(score)
