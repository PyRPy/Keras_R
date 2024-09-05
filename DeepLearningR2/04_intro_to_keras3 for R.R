
# Introduction to Keras for Researchers -----------------------------------
# https://cloud.r-project.org/web/packages/keras3/vignettes/intro_to_keras_for_researchers.html
library(keras3)
library(tensorflow)

### tensors
x <- tf$constant(rbind(c(5, 2), c(1, 3)))
print(x)

as.array(x)
x$dtype
x$shape

### create constant tensors
tf$ones(shape = shape(2, 1))
tf$zeros(shape = shape(2, 1))

x <- random_normal(shape = c(2, 2), mean = 0.0, stddev = 1.0)
x <- random_uniform(shape = c(2, 2), minval = 0, maxval = 10)

### variables
initial_value <- random_normal(shape=c(2, 2))
a <- tf$Variable(initial_value)
print(a)

new_value <- random_normal(shape=c(2, 2))
a$assign(new_value) # update variable a with new value
print(a)

new_value <- random_normal(shape=c(2, 2))
a$assign(new_value)

### Doing math in TensorFlow
a <- random_normal(shape=c(2, 2))
b <- random_normal(shape=c(2, 2))

c <- a + b
d <- tf$square(c)
e <- tf$exp(d) # to each element in the tensor

### Gradients
a <- random_normal(shape=c(2, 2))
b <- random_normal(shape=c(2, 2))

with(tf$GradientTape() %as% tape, {
  tape$watch(a)  # Start recording the history of operations applied to `a`
  c <- tf$sqrt(tf$square(a) + tf$square(b))  # Do some math using `a`
  # What's the gradient of `c` with respect to `a`?
  dc_da <- tape$gradient(c, a)
  print(dc_da)
})

# a variable is watched automatically
a <- tf$Variable(a)

with(tf$GradientTape() %as% tape, {
  c <- tf$sqrt(tf$square(a) + tf$square(b))
  dc_da <- tape$gradient(c, a)
  print(dc_da)
})

# nest tapes for higher order derivatives
with(tf$GradientTape() %as% outer_tape, {
  with(tf$GradientTape() %as% tape, {
    c <- tf$sqrt(tf$square(a) + tf$square(b))
    dc_da <- tape$gradient(c, a)
  })
  d2c_da2 <- outer_tape$gradient(dc_da, a)
  print(d2c_da2)
})

### Keras layers (the implementation uses R6 OOP concept)
# define a simple layer
Linear <- new_layer_class(
  "Linear",
  initialize = function(units = 32, input_dim = 32) {
    super$initialize()
    self$w <- self$add_weight(
      shape = shape(input_dim, units),
      initializer = "random_normal",
      trainable = TRUE
    )
    self$b <- self$add_weight(
      shape = shape(units),
      initializer = "zeros",
      trainable = TRUE
    )
  },
  call = function(inputs) {
    tf$matmul(inputs, self$w) + self$b
  }
)

# Instantiate our layer.
linear_layer <- Linear(units=4, input_dim=2)

# The layer can be treated as a function.
# Here we call it on some data.
y <- linear_layer(tf$ones(shape(2, 2)))
print(y)

linear_layer$weights

### Layer weight creation
Linear <- new_layer_class(
  "Linear",
  initialize = function(units = 32) {
    super$initialize()
    self$units <- units
  },
  build = function(input_shape) {
    self$w <- self$add_weight(
      shape = shape(input_shape[-1], self$units),
      initializer = "random_normal",
      trainable = TRUE
    )
    self$b <- self$add_weight(
      shape = shape(self$units),
      initializer = "zeros",
      trainable = TRUE
    )
  },
  call = function(inputs) {
    tf$matmul(inputs, self$w) + self$b
  }
)

# Instantiate our layer.
linear_layer <- Linear(units = 4)

# This will also call `build(input_shape)` and create the weights.
y <- linear_layer(tf$ones(shape(2, 2)))
print(y)


### Layer gradients
# Prepare a dataset.
c(c(x_train, y_train), .) %<-% dataset_mnist()

x_train <- array_reshape(x_train, c(60000, 784)) / 255

dataset <- tfdatasets::tensor_slices_dataset(list(x_train, y_train)) %>%
  tfdatasets::dataset_shuffle(buffer_size=1024) %>%
  tfdatasets::dataset_batch(64)

# Instantiate our linear layer (defined above) with 10 units.
linear_layer <- Linear(units = 10)

# Instantiate a logistic loss function that expects integer targets.
loss_fn <- loss_sparse_categorical_crossentropy(from_logits=TRUE)

# Instantiate an optimizer.
optimizer <- optimizer_sgd(learning_rate=1e-3)

# Iterate over the batches of the dataset.
coro::loop(for(data in dataset) {
  # Open a GradientTape.
  with(tf$GradientTape() %as% tape, {
    # Forward pass.
    logits <- linear_layer(data[[1]])

    # Loss value for this batch.
    loss_value <- loss_fn(data[[2]], logits)
  })

  # Get gradients of the loss wrt the weights.
  gradients <- tape$gradient(loss_value, linear_layer$trainable_weights)

  # Update the weights of our linear layer.
  optimizer$apply_gradients(zip_lists(gradients, linear_layer$trainable_weights))
})
loss_value

### Trainable and non-trainable weights
ComputeSum <- new_layer_class(
  "ComputeSum",
  initialize = function(input_dim) {
    super$initialize()
    # Create a non-trainable weight.
    self$total <- self$add_weight(
      initializer = "zeros",
      shape = shape(input_dim),
      trainable = FALSE
    )
  },
  call = function(inputs) {
    self$total$assign_add(tf$reduce_sum(inputs, axis=0L))
    self$total
  }
)

my_sum <- ComputeSum(input_dim = 2)
x <- tf$ones(shape(2, 2))

as.array(my_sum(x)) # [1] 2 2
as.array(my_sum(x)) # [1] 4 4
as.array(my_sum(x)) # [1] 6 6

### Layers that own layers
# Let's reuse the Linear class
# with a `build` method that we defined above.

MLP <- new_layer_class(
  "MLP",
  initialize = function() {
    super$initialize()
    self$linear_1 <- Linear(units = 32)
    self$linear_2 <- Linear(units = 32)
    self$linear_3 <- Linear(units = 10)
  },
  call = function(inputs) {
    x <- self$linear_1(inputs)
    x <- tf$nn$relu(x)
    x <- self$linear_2(x)
    x <- tf$nn$relu(x)
    return(self$linear_3(x))
  }
)

mlp <- MLP()

# The first call to the `mlp` object will create the weights.
y <- mlp(tf$ones(shape=shape(3, 64)))

# Weights are recursively tracked.
length(mlp$weights)

# equivalent to the following built-in option
mlp <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 10)
summary(mlp)

### Tracking losses created by layers
# A layer that creates an activity sparsity regularization loss
ActivityRegularization <- new_layer_class(
  "ActivityRegularization",
  initialize = function(rate=1e-2) {
    super$initialize()
    self$rate <- rate
  },
  call = function(inputs) {
    self$add_loss(self$rate * tf$reduce_sum(tf$abs(inputs)))
    inputs
  }
)

# Let's use the loss layer in a MLP block.
SparseMLP <- new_layer_class(
  "SparseMLP",
  initialize = function() {
    super$initialize()
    self$linear_1 <- Linear(units = 32)
    self$reg <- ActivityRegularization(rate = 1e-2)
    self$linear_3 <- Linear(units = 10)
  },
  call = function(inputs) {
    x <- self$linear_1(inputs)
    x <- tf$nn$relu(x)
    x <- self$reg(x)
    return(self$linear_3(x))
  }
)

mlp <- SparseMLP()
y <- mlp(tf$ones(shape(10, 10)))

mlp$losses  # List containing one float32 scalar

# Losses correspond to the *last* forward pass.
mlp <- SparseMLP()
mlp(tf$ones(shape(10, 10)))

length(mlp$losses)
mlp(tf$ones(shape(10, 10)))
length(mlp$losses)  # No accumulation

### Let's demonstrate how to use these losses in a training loop.

# Prepare a dataset
c(c(x_train, y_train), .) %<-% dataset_mnist()
x_train <- array_reshape(x_train, c(60000, 784)) / 255

dataset <- tfdatasets::tensor_slices_dataset(list(x_train, y_train)) %>%
  tfdatasets::dataset_shuffle(buffer_size=1024) %>%
  tfdatasets::dataset_batch(64)

# A new MLP.
mlp <- SparseMLP()

# Loss and optimizer.
loss_fn <- loss_sparse_categorical_crossentropy(from_logits=TRUE)
optimizer <- optimizer_sgd(learning_rate=1e-3)

coro::loop(for(data in dataset) {
  x <- data[[1]]
  y <- data[[2]]
  with(tf$GradientTape() %as% tape, {
    # Forward pass.
    logits <- mlp(x)

    # External loss value for this batch.
    loss <- loss_fn(y, logits)

    # Add the losses created during the forward pass.
    loss <- loss + Reduce(`+`, mlp$losses)

    # Get gradients of the loss wrt the weights.
    gradients <- tape$gradient(loss, mlp$trainable_weights)

    # Update the weights of our linear layer.
    optimizer$apply_gradients(zip_lists(gradients, mlp$trainable_weights))
  })
})

print(loss)

### Keeping track of training metrics
# Instantiate a metric object
accuracy <- metric_sparse_categorical_accuracy()

# Prepare our layer, loss, and optimizer.
model <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 10)
loss_fn <- loss_sparse_categorical_crossentropy(from_logits = TRUE)
optimizer <- optimizer_adam(learning_rate=1e-3)

for (epoch in seq_len(2)) {
  coro::loop(for (data in dataset) {
    x <- data[[1]]
    y <- data[[2]]
    with(tf$GradientTape() %as% tape, {
      # Forward pass.
      logits <- model(x)

      # External loss value for this batch.
      loss_value <- loss_fn(y, logits)
    })

    # Update the state of the `accuracy` metric.
    accuracy$update_state(y, logits)

    # Update the weights of the model to minimize the loss value.
    gradients <- tape$gradient(loss_value, model$trainable_weights)
    optimizer$apply_gradients(zip_lists(gradients, model$trainable_weights))

  })
  cat("Epoch:", epoch, "Accuracy:", as.numeric(accuracy$result()), "\n")
  accuracy$reset_state()
}
# Epoch: 1 Accuracy: 0.882
# Epoch: 2 Accuracy: 0.94615

### Compiled functions
# Prepare our layer, loss, and optimizer.
model <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 10)
loss_fn <- loss_sparse_categorical_crossentropy(from_logits = TRUE)
optimizer <- optimizer_adam(learning_rate=1e-3)

# Create a training step function.
train_on_batch <- tf_function(function(x, y) {
  with(tf$GradientTape() %as% tape, {
    # Forward pass.
    logits <- model(x)
    # External loss value for this batch.
    loss_value <- loss_fn(y, logits)
  })
  # Update the weights of the model to minimize the loss value.
  gradients <- tape$gradient(loss_value, model$trainable_weights)
  optimizer$apply_gradients(zip_lists(gradients, model$trainable_weights))
  loss_value
})

# Prepare a dataset.
c(c(x_train, y_train), .) %<-% dataset_mnist()
x_train <- array_reshape(x_train, c(60000, 784)) / 255

dataset <- tfdatasets::tensor_slices_dataset(list(x_train, y_train)) %>%
  tfdatasets::dataset_shuffle(buffer_size=1024) %>%
  tfdatasets::dataset_batch(64)

i <- 0
coro::loop(for (data in dataset) {
  i <- i + 1
  x <- data[[1]]
  y <- data[[2]]
  loss <- train_on_batch(x, y)
  if (i %% 100 == 0)
    cat("Loss:", as.numeric(loss), "\n")
})

### Training mode & inference mode
Dropout <- new_layer_class(
  "Dropout",
  initialize = function(rate) {
    super$initialize()
    self$rate <- rate
  },
  call = function(inputs, training = NULL) {
    if (!is.null(training) && training) {
      return(tf$nn$dropout(inputs, rate = self$rate))
    }
    inputs
  }
)

MLPWithDropout <- new_layer_class(
  "MLPWithDropout",
  initialize = function() {
    super$initialize()
    self$linear_1 <- Linear(units = 32)
    self$dropout <- Dropout(rate = 0.5)
    self$linear_3 <- Linear(units = 10)
  },
  call = function(inputs, training = NULL) {
    x <- self$linear_1(inputs)
    x <- tf$nn$relu(x)
    x <- self$dropout(x, training = training)
    self$linear_3(x)
  }
)

mlp <- MLPWithDropout()
y_train <- mlp(tf$ones(shape(2, 2)), training=TRUE)
y_test <- mlp(tf$ones(shape(2, 2)), training=FALSE)

### The Functional API for model-building

# We use an `Input` object to describe the shape and dtype of the inputs.
# This is the deep learning equivalent of *declaring a type*.
# The shape argument is per-sample; it does not include the batch size.
# The functional API focused on defining per-sample transformations.
# The model we create will automatically batch the per-sample transformations,
# so that it can be called on batches of data.
inputs <- layer_input(shape = 16, dtype = "float32")

outputs <- inputs %>%
  Linear(units = 32) %>% # We are reusing the Linear layer we defined earlier.
  Dropout(rate = 0.5) %>% # We are reusing the Dropout layer we defined earlier.
  Linear(units = 10)

# A functional `Model` can be defined by specifying inputs and outputs.
# A model is itself a layer like any other.
model <- keras_model(inputs, outputs)

# A functional model already has weights, before being called on any data.
# That's because we defined its input shape in advance (in `Input`).
length(model$weights)

# Let's call our model on some data, for fun.
y <- model(tf$ones(shape(2, 16)))
y$shape

# You can pass a `training` argument in `__call__`
# (it will get passed down to the Dropout layer).
y <- model(tf$ones(shape(2, 16)), training=TRUE)

### built-in training infrastructure to implement the MNIST example
inputs <- layer_input(shape = 784, dtype="float32")
outputs <- inputs %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 10)
model <- keras_model(inputs, outputs)

# Specify the loss, optimizer, and metrics with `compile()`.
model %>% compile(
  loss = loss_sparse_categorical_crossentropy(from_logits=TRUE),
  optimizer=optimizer_adam(learning_rate=1e-3),
  metrics=list(metric_sparse_categorical_accuracy()),
)

# Train the model with the dataset for 2 epochs.
model %>% fit(dataset, epochs=2)
# 938/938 ━━━━━━━━ loss: 0.2122 - sparse_categorical_accuracy: 0.9384

# prediction
predictions <- model %>% predict(dataset)

model %>% evaluate(dataset)

### alter the model to your own
CustomModel <- new_model_class(
  "CustomModel",
  initialize = function(...) {
    super$initialize(...)
    self$loss_tracker <- metric_mean(name="loss")
    self$accuracy <- metric_sparse_categorical_accuracy()
    self$loss_fn <- loss_sparse_categorical_crossentropy(from_logits=TRUE)
    self$optimizer <- optimizer_adam(learning_rate=1e-3)
  },
  train_step = function(data) {
    c(x, y = NULL, sample_weight = NULL) %<-% data
    with(tf$GradientTape() %as% tape, {
      y_pred <- self(x, training=TRUE)
      loss <- self$loss_fn(y = y, y_pred = y_pred, sample_weight=sample_weight)
    })
    gradients <- tape$gradient(loss, self$trainable_variables)
    self$optimizer$apply_gradients(
      zip_lists(gradients, self$trainable_variables)
    )

    # Update metrics (includes the metric that tracks the loss)
    self$loss_tracker$update_state(loss)
    self$accuracy$update_state(y, y_pred, sample_weight=sample_weight)
    # Return a list mapping metric names to current value
    list(
      loss = self$loss_tracker$result(),
      accuracy = self$accuracy$result()
    )
  },
  metrics = mark_active(function() {
    list(self$loss_tracker, self$accuracy)
  })
)

inputs <- layer_input(shape = 784, dtype="float32")
outputs <- inputs %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 10)
model <- CustomModel(inputs, outputs)
model %>% compile()
model %>% fit(dataset, epochs=2)
