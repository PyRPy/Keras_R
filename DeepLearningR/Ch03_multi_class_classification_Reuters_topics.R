# multiple label classification
# https://jjallaire.github.io/deep-learning-with-r-notebooks/notebooks/3.5-classifying-newswires.nb.html
# text classification, 46 topics

# get the Reuters datasets
library(keras)
reuters <- dataset_reuters(num_words = 10000)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% reuters

# inspect the data
str(train_data)
length(train_data)
length(test_data)

# check word indices
train_data[[1]]

# decode back to words
word_index <- dataset_reuters_word_index()
reverse_word_index <- names(word_index)
names(reverse_word_index) <- word_index
decoded_newswire <- sapply(train_data[[1]], function(index) {
  # Note that our indices were offset by 3 because 0, 1, and 2
  # are reserved indices for "padding", "start of sequence", and "unknown".
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
})

# display words
cat(decoded_newswire)

train_labels[[1]]

# Preparing the data
vectorize_sequences <- function(sequences, dimension = 10000) {
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences))
    results[i, sequences[[i]]] <- 1
  results
}
x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)

# one-hot encoding
to_one_hot <- function(labels, dimension = 46) {
  results <- matrix(0, nrow = length(labels), ncol = dimension)
  for (i in 1:length(labels))
    results[i, labels[[i]] + 1] <- 1
  results
}
one_hot_train_labels <- to_one_hot(train_labels)
one_hot_test_labels <- to_one_hot(test_labels)

# alternatively
# one_hot_train_labels <- to_categorical(train_labels)
# one_hot_test_labels <- to_categorical(test_labels)

# Building neural network
# consider potential bottle neck effect
model <- keras_model_sequential() %>% 
  layer_dense(units = 64, activation = "relu", input_shape = c(10000)) %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 46, activation = "softmax")


model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

# Validation steps
val_indices <- 1:1000
x_val <- x_train[val_indices,]
partial_x_train <- x_train[-val_indices,]
y_val <- one_hot_train_labels[val_indices,]
partial_y_train = one_hot_train_labels[-val_indices,]

# train the model
history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

# plot the loss and accuracy history
plot(history)

# select the model at 10 epochs
model <- keras_model_sequential() %>% 
  layer_dense(units = 64, activation = "relu", input_shape = c(10000)) %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 46, activation = "softmax")

model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)
history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 10,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)
results <- model %>% evaluate(x_test, one_hot_test_labels)

results

# common sense check
test_labels_copy <- test_labels
test_labels_copy <- sample(test_labels_copy)
length(which(test_labels == test_labels_copy)) / length(test_labels)

# predict on new data
predictions <- model %>% predict(x_test)

dim(predictions)
predictions[1,]
sum(predictions[1,])

# the largest probability is the 'label' predicted
which.max(predictions[1,])
