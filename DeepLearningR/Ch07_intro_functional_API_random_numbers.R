# intro to functional API with Keras

library(keras)
# seq_model <- keras_model_sequential() %>%
#   layer_dense(units = 32, activation = "relu", input_shape = c(64)) %>%
#   layer_dense(units = 32, activation = "relu") %>%
#   layer_dense(units = 10, activation = "softmax")

# --- equivalent API --- # 
input_tensor <- layer_input(shape = c(64))

output_tensor <- input_tensor %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

model <- keras_model(input_tensor, output_tensor)
summary(model)

model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy", 
  metrics = c('accuracy')
)

# generate random numbers for inputs
x_train <- array(runif(1000 * 64), dim = c(1000, 64))
y_train <- array(runif(1000 * 10), dim = c(1000, 10))

# generate 'pseudo' test data set
x_test <- array(runif(100 * 64), dim = c(100, 64))
y_test <- array(runif(100 * 10), dim = c(100, 10))

model %>% fit(x_train, y_train, epochs = 10, batch_size = 128)
model %>% evaluate(x_train, y_train)

model %>% evaluate(x_test, y_test)
# it's random, best you could guess correct is : 1/10 or 10% as 0.1 for acc