# lstm with reversed order and bidirectional

k_clear_session()

# --- lstm with reversed order --- #
library(keras)
max_features <- 10000  # Number of words to consider as features
maxlen <- 500          # Cut texts after this number of words 
# (among top max_features most common words)
# Load data
imdb <- dataset_imdb(num_words = max_features)
c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb
# Reverse sequences
x_train <- lapply(x_train, rev) # reverse seq
x_test <- lapply(x_test, rev) # reverse seq
# Pad sequences
x_train <- pad_sequences(x_train, maxlen = maxlen)
x_test <- pad_sequences(x_test, maxlen = maxlen)
model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_features, output_dim = 128) %>% 
  layer_lstm(units = 32) %>% 
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

history <- model %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 128,
  validation_split = 0.2
)

plot(history)

k_clear_session()

# --- bidirectional lstm --- #
model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_features, output_dim = 32) %>% 
  bidirectional(
    layer_lstm(units = 32)
  ) %>% 
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)
history <- model %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 128,
  validation_split = 0.2
)

plot(history)