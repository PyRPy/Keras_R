# overfitting and underfitting
# https://jjallaire.github.io/deep-learning-with-r-notebooks/notebooks/4.4-overfitting-and-underfitting.nb.html
# load data same as in ch03
library(keras)
imdb <- dataset_imdb(num_words = 10000)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% imdb

vectorize_sequences <- function(sequences, dimension = 10000) {
  # Create an all-zero matrix of shape (len(sequences), dimension)
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences))
    # Sets specific indices of results[i] to 1s
    results[i, sequences[[i]]] <- 1
  results
}

# Our vectorized training data
x_train <- vectorize_sequences(train_data)
# Our vectorized test data
x_test <- vectorize_sequences(test_data)
# Our vectorized labels
y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)

# Fighting overfitting
#------------------ Reducing the network's size---------------------------

original_model <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>% 
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")
original_model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

# use smaller network
smaller_model <- keras_model_sequential() %>% 
  layer_dense(units = 4, activation = "relu", input_shape = c(10000)) %>% 
  layer_dense(units = 4, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")
smaller_model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

# plot losses from different models
library(ggplot2)
library(tidyr)
plot_training_losses <- function(losses) {
  loss_names <- names(losses)
  losses <- as.data.frame(losses)
  losses$epoch <- seq_len(nrow(losses))
  losses %>% 
    gather(model, loss, loss_names[[1]], loss_names[[2]]) %>% 
    ggplot(aes(x = epoch, y = loss, colour = model)) +
    geom_point()
}

# Validating our approach
val_indices <- 1:10000
x_val <- x_train[val_indices,]
partial_x_train <- x_train[-val_indices,]
y_val <- y_train[val_indices]
partial_y_train <- y_train[-val_indices]

# fit model using original model
original_hist <- original_model %>% fit(
  partial_x_train, partial_y_train, 
  epochs=20, batch_size = 512,
  validation_data = list(x_val, y_val))

# fit model using smaller model
smaller_hist <- smaller_model %>% fit(
  partial_x_train, partial_y_train, 
  epochs=20, batch_size = 512,
  validation_data = list(x_val, y_val))

# compare the losses over epoch
plot_training_losses(losses = list(
  original_model = original_hist$metrics$val_loss,
  smaller_model = smaller_hist$metrics$val_loss
))

# smaller network show less performance degradation when overfitting

# use a bigger model, for comparison
bigger_model <- keras_model_sequential() %>% 
  layer_dense(units = 512, activation = "relu", input_shape = c(10000)) %>% 
  layer_dense(units = 512, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")
bigger_model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c('acc')
)

# fit data on bigger mdoel
bigger_hist <- bigger_model %>% fit(
  partial_x_train, partial_y_train, 
  epochs=20, batch_size = 512,
  validation_data = list(x_val, y_val))

# plot bigger model vs original model
plot_training_losses(losses = list(
  original_model = original_hist$metrics$val_loss,
  bigger_model = bigger_hist$metrics$val_loss
))

# bigger model start overfitting very early

# plot training errors 
plot_training_losses(losses = list(
  original_model = original_hist$metrics$loss,
  bigger_model = bigger_hist$metrics$loss
))
# bigger model performs 'perfect' on training data set, clearly overfitting occurs

# another method to reduce overfitting
# -------------------using weight regularization-------------------------
# add l2 regulization 
l2_model <- keras_model_sequential() %>% 
  layer_dense(units = 16, kernel_regularizer = regularizer_l2(0.001),
              activation = "relu", input_shape = c(10000)) %>% 
  layer_dense(units = 16, kernel_regularizer = regularizer_l2(0.001),
              activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

l2_model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

# fit model on l2 model
l2_hist <- l2_model %>% fit(
  partial_x_train, partial_y_train, 
  epochs=20, batch_size = 512,
  validation_data = list(x_val, y_val))

# compare l2 with original
plot_training_losses(losses = list(
  original_model = original_hist$metrics$val_loss,
  l2_model = l2_hist$metrics$val_loss
))

# l2 model is more resistent to overfitting than the original model
# L1 regularization
# regularizer_l1(0.001)
# L1 and L2 regularization at the same time
# regularizer_l1_l2(l1 = 0.001, l2 = 0.001)

# using dropout method
#------------------------dropout method-------------------------
dpt_model <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = "sigmoid")
dpt_model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

dpt_hist <- dpt_model %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_test, y_test)
)

# compare dropout vs original model
plot_training_losses(losses = list(
  original_model = original_hist$metrics$val_loss,
  dpt_model = dpt_hist$metrics$val_loss
))

# lower losses for dropout model in validation dataset
#-------------most common ways to reduce overfitting----------------------
# 1---Getting more training data.
# 2---Reducing the capacity of the network.
# 3---Adding weight regularization.
# 4---Adding dropout.