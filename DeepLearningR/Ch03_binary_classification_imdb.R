# reference :
# https://jjallaire.github.io/deep-learning-with-r-notebooks/notebooks/3.4-classifying-movie-reviews.nb.html
# classify movie reviews into "positive" reviews and "negative" reviews, 
# just based on the text content of the reviews.

# load data set
library(keras)
imdb <- dataset_imdb(num_words = 10000)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% imdb

# check the data structures
str(train_data[[1]]) # sparse matrix
str(train_labels)

train_labels[[1]]

max(sapply(train_data, max))

# word_index is a dictionary mapping words to an integer index
word_index <- dataset_imdb_word_index()
# We reverse it, mapping integer indices to words
reverse_word_index <- names(word_index)
names(reverse_word_index) <- word_index
# We decode the review; note that our indices were offset by 3
# because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
decoded_review <- sapply(train_data[[1]], function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
})

# see what's in the text
cat(decoded_review)

# Preparing the data
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

str(x_train)
str(x_train[1,])

# Our vectorized labels
y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)

# Building our network
#-----------------------------------------------------------------
#                    output = relu(dot(W, input) + b)
#-----------------------------------------------------------------

# three layers
model <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>% 
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

# compile model
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

# Validating our approach
val_indices <- 1:10000
x_val <- x_train[val_indices,]
partial_x_train <- x_train[-val_indices,]
y_val <- y_train[val_indices]
partial_y_train <- y_train[-val_indices]

# fit the model on training data
# pass a list to validation_data
# https://keras.rstudio.com/articles/examples/imdb_fasttext.html
history <- model %>% fit(
  partial_x_train, partial_y_train, 
  epochs=20, batch_size = 512,
  validation_data = list(x_val, y_val))

str(history)

# plot history
plot(history)

# select the 'best' parameters for the model
model <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>% 
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)
model %>% fit(x_train, y_train, epochs = 4, batch_size = 512)

results <- model %>% evaluate(x_test, y_test)
results

# predict on new data
model %>% predict(x_test[1:10,])

# use glm model
str(x_train)
str(y_train)
model.glm <- glm(y_train ~ x_train, family="binomial")
# memory allocation problem - close to 1 G