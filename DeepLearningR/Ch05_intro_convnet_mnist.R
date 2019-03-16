# intro to convnets
# 
library(keras)

# a convnet takes as input tensors of shape (image_height, image_width, image_channels)
model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(28, 28, 1)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu")

# display model
summary(model)

# flattern and then connect it with dense network like in ch02
model <- model %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 10, activation = "softmax")

# check model structure again
summary(model)

# (3, 3, 64) outputs are flattened into vectors of shape (576) before going through two dense layers.

# load dataset
mnist <- dataset_mnist()
c(c(train_images, train_labels), c(test_images, test_labels)) %<-% mnist
train_images <- array_reshape(train_images, c(60000, 28, 28, 1))
train_images <- train_images / 255
test_images <- array_reshape(test_images, c(10000, 28, 28, 1))
test_images <- test_images / 255
train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

# compile the model
model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

# fit model on training data
model %>% fit(
  train_images, train_labels, 
  epochs = 5, batch_size=64
)
# model runs significantly slower than densely connected network

# evaluate model on test dataset
results <- model %>% evaluate(test_images, test_labels)

results
# $`loss`
# [1] 0.03408702
# 
# $acc
# [1] 0.9899