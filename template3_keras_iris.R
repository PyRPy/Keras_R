# Building DNNs with Keras in R
# https://www.r-bloggers.com/an-introduction-to-machine-learning-with-keras-in-r/
library(keras)
# use_session_with_seed(1,disable_parallel_cpu = FALSE)

# read data ---------------------------------------------------------------


data("iris")
dat <- iris[sample(nrow(iris)),]
head(dat)
y <- dat[, "Species"]
x <- dat[,1:4]

# scale data --------------------------------------------------------------

# scale to [0,1]
x <- as.matrix(apply(x, 2, function(x) (x-min(x))/(max(x) - min(x))))


# # one hot encode classes / create DummyFeatures -------------------------
levels(y) <- 1:length(y)
y <- to_categorical(as.integer(y) - 1 , num_classes = 3)

# construct model ---------------------------------------------------------
# create sequential model
model <- keras_model_sequential()

# add layers, first layer needs input dimension
model %>%
  layer_dense(input_shape = ncol(x), units = 10, activation = "relu") %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dense(units = 3, activation = "softmax")

# add a loss function and optimizer
model %>%
  compile(
    loss = "categorical_crossentropy",
    optimizer = "adagrad",
    metrics = "accuracy"
  )

# fit model with training data set, 200 times
fit <- model %>%
  fit(
    x = x,
    y = y,
    shuffle = T,
    batch_size = 5,
    validation_split = 0.3,
    epochs = 200
  )


# plot accuracy -----------------------------------------------------------

plot(fit)

# DNN with dropout --------------------------------------------------------
model <-  keras_model_sequential()
model %>%
  layer_dense(input_shape = ncol(x), units = 10, activation = "relu") %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 3, activation = "softmax")

model %>%
  compile(
    loss = "categorical_crossentropy",
    optimizer = "adagrad",
    metrics = "accuracy"
  )

fit = model %>%
  fit(
    x = x,
    y = y,
    shuffle = T,
    validation_split = 0.3,
    epochs = 200,
    batch_size = 5
  )
plot(fit)
