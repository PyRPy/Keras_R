# regression model 
# load Boston dataset

library(keras)
dataset <- dataset_boston_housing()

c(c(train_data, train_targets), c(test_data, test_targets)) %<-% dataset
str(train_data)
class(train_data) # matrix

str(test_data)
class(test_data) # matrix too

# check one observation
train_data[1, ]

# Preparing the data
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
train_data <- scale(train_data, center = mean, scale = std)
test_data <- scale(test_data, center = mean, scale = std)

# display for inspections
mean
std
train_data[1, ]
test_data[1, ]

# construct neural network
# Because we will need to instantiate the same model multiple times,
# we use a function to construct it.
build_model <- function() {
  model <- keras_model_sequential() %>% 
    layer_dense(units = 64, activation = "relu", 
                input_shape = dim(train_data)[[2]]) %>% 
    layer_dense(units = 64, activation = "relu") %>% 
    layer_dense(units = 1) 
  
  model %>% compile(
    optimizer = "rmsprop", 
    loss = "mse", 
    metrics = c("mae")
  )
}

# K-fold validation
k <- 4
indices <- sample(1:nrow(train_data))
folds <- cut(1:length(indices), breaks = k, labels = FALSE) 

num_epochs <- 100
all_scores <- c()
for (i in 1:k) {
  cat("processing fold #", i, "\n")
  # Prepare the validation data: data from partition # k
  val_indices <- which(folds == i, arr.ind = TRUE) 
  val_data <- train_data[val_indices,]
  val_targets <- train_targets[val_indices]
  
  # Prepare the training data: data from all other partitions
  partial_train_data <- train_data[-val_indices,]
  partial_train_targets <- train_targets[-val_indices]
  
  # Build the Keras model (already compiled)
  model <- build_model()
  
  # Train the model (in silent mode, verbose=0)
  model %>% fit(partial_train_data, partial_train_targets,
                epochs = num_epochs, batch_size = 1, verbose = 0)
  
  # Evaluate the model on the validation data
  results <- model %>% evaluate(val_data, val_targets, verbose = 0)
  all_scores <- c(all_scores, results$mean_absolute_error)
}  

# display scores
all_scores
mean(all_scores)

# clean memory
# Some memory clean-up
k_clear_session()

# train the model longer
num_epochs <- 70 # 500, 150
all_mae_histories <- NULL
for (i in 1:k) {
  cat("processing fold #", i, "\n")
  
  # Prepare the validation data: data from partition # k
  val_indices <- which(folds == i, arr.ind = TRUE)
  val_data <- train_data[val_indices,]
  val_targets <- train_targets[val_indices]
  
  # Prepare the training data: data from all other partitions
  partial_train_data <- train_data[-val_indices,]
  partial_train_targets <- train_targets[-val_indices]
  
  # Build the Keras model (already compiled)
  model <- build_model()
  
  # Train the model (in silent mode, verbose=0)
  history <- model %>% fit(
    partial_train_data, partial_train_targets,
    validation_data = list(val_data, val_targets),
    epochs = num_epochs, batch_size = 1, verbose = 0
  )
  mae_history <- history$metrics$val_mean_absolute_error
  all_mae_histories <- rbind(all_mae_histories, mae_history)
}

# average of the per-epoch MAE scores for all folds
average_mae_history <- data.frame(
  epoch = seq(1:ncol(all_mae_histories)),
  validation_mae = apply(all_mae_histories, 2, mean)
)

average_mae_history # for 150 epoch, 4 folds average in each epoch

# plot for inspection
library(ggplot2)
ggplot(average_mae_history, aes(x = epoch, y = validation_mae)) + geom_line()

ggplot(average_mae_history, aes(x = epoch, y = validation_mae)) + geom_smooth()

# select best model parameters and rebuild model for predictions
model <- build_model()

# Train the model (in silent mode, verbose=0)
model %>% fit(train_data, train_targets,
              epochs = num_epochs, batch_size = 1, verbose = 0)

# Evaluate the model on the validation data
results <- model %>% evaluate(test_data, test_targets, verbose = 0)

# check result again
results
