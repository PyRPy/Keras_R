
# MNIST Example -----------------------------------------------------------
# https://cloud.r-project.org/web/packages/keras3/vignettes/getting_started.html
# install.packages("keras3")
# keras3::install_keras(backend = "tensorflow")


### Preparing the Data

library(keras3)
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# reshape
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
# rescale
x_train <- x_train / 255
x_test <- x_test / 255

# one-hot encode
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

### Defining the Model: sequential model
model <- keras_model_sequential(input_shape = c(784))
model |>
  layer_dense(units = 256, activation = 'relu') |>
  layer_dropout(rate = 0.4) |>
  layer_dense(units = 128, activation = 'relu') |>
  layer_dropout(rate = 0.3) |>
  layer_dense(units = 10, activation = 'softmax')

summary(model)
plot(model) # not working

### compile the model
model |> compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

### Training and Evaluation
history <- model |> fit(
  x_train, y_train,
  epochs = 30, batch_size = 128,
  validation_split = 0.2
)

plot(history)

### evaluate on the test data
model |> evaluate(x_test, y_test) # 0.9818

### predict on new data
probs <- model |> predict(x_test)
max.col(probs) - 1L

# [1] 7 2 1 0 4
y_test[1:10]
y_test[11:20]
max.col(probs)[1] == mnist$test$y[1]
max.col(probs)[1]

### manually calculate prediction accuracy
sum(mnist$test$y == (max.col(probs) - 1L)) / length(mnist$test$y) # 0.9818
