# how to use tensorboard
# text classification with tensorboard

library(keras)
# library(tensorflow)
# try this https://github.com/rstudio/keras/issues/62
# it collapses keras, cannot run; have to reinstall from scratch !!!
# install_keras()
max_features <- 2000 # top 2000 words only
max_len <- 500 # cut-off texts beyond this number of words

imdb <- dataset_imdb(num_words = max_features)
c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb
x_train <- pad_sequences(x_train, maxlen = max_len)
x_test = pad_sequences(x_test, maxlen = max_len)

# remove name =

model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_features, output_dim = 128,
                  input_length = max_len) %>% 
  layer_conv_1d(filters = 32, kernel_size = 7, activation = "relu") %>% 
  layer_max_pooling_1d(pool_size = 5) %>% 
  layer_conv_1d(filters = 32, kernel_size = 7, activation = "relu") %>% 
  layer_global_max_pooling_1d() %>% 
  layer_dense(units = 1)

# model <- keras_model_sequential() %>%
#   layer_embedding(input_dim = max_features, output_dim = 128,
#                   input_length = max_len) %>%
#   layer_conv_1d(filters = 32, kernel_size = 7, activation = "relu") %>%
#   layer_max_pooling_1d(pool_size = 5) %>%
#   layer_conv_1d(filters = 32, kernel_size = 7, activation = "relu") %>%
#   layer_global_max_pooling_1d() %>%
#   layer_dense(units = 1)

summary(model)

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

# create a directory for tensorboard log files
dir.create("my_log_dir")

tensorboard("my_log_dir")

callbacks = list(
  callback_tensorboard(
    log_dir = "my_log_dir",
    histogram_freq = 1
    # embeddings_freq = 1 # removed
  )
)

history <- model %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 128,
  validation_split = 0.2,
  callbacks = callbacks
)

# correction from book website
# https://manning-content.s3.amazonaws.com/download/1/6d1e9c3-9483-455c-8a47-54bb4bbd3205/Allaire_DeepLearningwithR_err4.html