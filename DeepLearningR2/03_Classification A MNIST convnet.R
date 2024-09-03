
# A MNIST convnet ---------------------------------------------------------
# https://cloud.r-project.org/web/packages/keras3/vignettes/intro_to_keras_for_engineers.html
library(tensorflow, exclude = c("shape", "set_random_seed"))
library(keras3)
use_backend("tensorflow")

# Load the data and split it between train and test sets
c(c(x_train, y_train), c(x_test, y_test)) %<-% keras3::dataset_mnist()

# Scale images to the [0, 1] range
x_train <- x_train / 255
x_test <- x_test / 255

# Make sure images have shape (28, 28, 1)
x_train <- array_reshape(x_train, c(-1, 28, 28, 1))
x_test <- array_reshape(x_test, c(-1, 28, 28, 1))

dim(x_train)

# Model parameters
num_classes <- 10
input_shape <- c(28, 28, 1)

model <- keras_model_sequential(input_shape = input_shape)
model |>
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") |>
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") |>
  layer_max_pooling_2d(pool_size = c(2, 2)) |>
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") |>
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") |>
  layer_global_average_pooling_2d() |>
  layer_dropout(rate = 0.5) |>
  layer_dense(units = num_classes, activation = "softmax")

summary(model)

# compile model
model |> compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = list(
    metric_sparse_categorical_accuracy(name = "acc")
  )
)

# run the model
batch_size <- 128
epochs <- 10

callbacks <- list(
  callback_model_checkpoint(filepath="model_at_epoch_{epoch}.keras"),
  callback_early_stopping(monitor="val_loss", patience=2)
)

model |> fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  validation_split = 0.15,
  callbacks = callbacks
)

# evaluate model
score <- model |> evaluate(x_test, y_test, verbose = 0)

# save the final model
save_model(model, "final_model.keras", overwrite=TRUE)

# reload the model
model <- load_model("final_model.keras")

# make predctions
predictions <- model |> predict(x_test)

### manually calculate prediction accuracy
# higher accuracy, but model is trained much slower on CPU only computer
sum(y_test == (max.col(predictions) - 1L)) / length(y_test) # 0.9899
