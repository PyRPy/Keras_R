# use rnn for forecasting
# A temperature forecasting problem

# download data
dir.create("~/Downloads/jena_climate", recursive = TRUE)
download.file(
  "https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip",
  "~/Downloads/jena_climate/jena_climate_2009_2016.csv.zip"
)
unzip(
  "~/Downloads/jena_climate/jena_climate_2009_2016.csv.zip",
  exdir = "~/Downloads/jena_climate"
)

# inspect the data first
library(tibble)
library(readr)
data_dir <- "~/Downloads/jena_climate"
fname <- file.path(data_dir, "jena_climate_2009_2016.csv")
data <- read_csv(fname)

head(data)
glimpse(data)
str(data)

# Here is the plot of temperature (in degrees Celsius) over time
library(ggplot2)
ggplot(data, aes(x = 1:nrow(data), y = `T (degC)`)) + geom_line()

# plot 10 days data (data sampled every 10 mins, 6 x 24 x 10 = 1440)
ggplot(data[1:1440,], aes(x = 1:1440, y = `T (degC)`)) + geom_line()

# --- proprocessing data ---

# lookback = 1440, i.e. our observations will go back 10 days.
# steps = 6, i.e. our observations will be sampled at one data point per hour.
# delay = 144, i.e. our targets will be 24 hours in the future.

# --- remove timestamp---
data <- data.matrix(data[,-1])

# --- normalize data ---
train_data <- data[1:200000,]
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
data <- scale(data, center = mean, scale = std)
# normalize data based on 'train_data'

# --- generator function ---
# It yields a list (samples, targets), where samples is one batch of input 
# data and targets is the corresponding array of target temperatures.


# data - The original array of floating-point data, which you normalized in listing 6.32.
# lookback - How many timesteps back the input data should go.
# delay - How many timesteps in the future the target should be.
# min_index and max_index - Indices in the data array that delimit which timesteps to draw from. 
# This is useful for keeping a segment of the data for validation and another for testing.
# shuffle - Whether to shuffle the samples or draw them in chronological order.
# batch_size - The number of samples per batch.
# step - The period, in timesteps, at which you sample data. You'll set it 6 in order to 
# draw one data point every hour.

library(dplyr)
generator <- function(data, lookback, delay, min_index, max_index,
                      shuffle = FALSE, batch_size = 128, step = 6) {
  if (is.null(max_index))
    max_index <- nrow(data) - delay - 1
  i <- min_index + lookback
  function() {
    if (shuffle) {
      rows <- sample(c((min_index+lookback):max_index), size = batch_size)
    } else {
      if (i + batch_size >= max_index)
        i <<- min_index + lookback
      rows <- c(i:min(i+batch_size-1, max_index))
      i <<- i + length(rows)
    }
    
    samples <- array(0, dim = c(length(rows), 
                                lookback / step,
                                dim(data)[[-1]]))
    targets <- array(0, dim = c(length(rows)))
    
    for (j in 1:length(rows)) {
      indices <- seq(rows[[j]] - lookback, rows[[j]] - 1, 
                     length.out = dim(samples)[[2]])
      samples[j,,] <- data[indices,]
      targets[[j]] <- data[rows[[j]] + delay,2]
    }            
    
    list(samples, targets)
  }
}

# --- split data sets ---
lookback <- 1440
step <- 6
delay <- 144
batch_size <- 128

# the training generator looks at the first 200,000 timesteps
train_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 1,
  max_index = 200000,
  shuffle = TRUE,
  step = step, 
  batch_size = batch_size
)

# the validation generator looks at the following 100,000, and 
val_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 200001,
  max_index = 300000,
  step = step,
  batch_size = batch_size
)

# the test generator looks at the remainder.
test_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 300001,
  max_index = NULL,
  step = step,
  batch_size = batch_size
)

# This is how many steps to draw from `val_gen`
# in order to see the whole validation set:
val_steps <- (300000 - 200001 - lookback) / batch_size
# This is how many steps to draw from `test_gen`
# in order to see the whole test set:
test_steps <- (nrow(data) - 300001 - lookback) / batch_size

# --- common sense baseline ---

# a common sense approach would be to always predict that the temperature 
# 24 hours from now will be equal to the temperature right now
# using the Mean Absolute Error metric (MAE). Mean Absolute Error is simply 
# equal to: mean(abs(preds - targets))

# not working...solved problem after loading 'dplyr' package
library(dplyr)
evaluate_naive_method <- function() {
  batch_maes <- c()
  for (step in 1:val_steps) {
    c(samples, targets) %<-% val_gen()
    preds <- samples[,dim(samples)[[2]],2]
    mae <- mean(abs(preds - targets))
    batch_maes <- c(batch_maes, mae)
  }
  print(mean(batch_maes))
}

evaluate_naive_method() # 0.2789863, 0.28 * temperature_std

# --- two dense layer ---
# a fully connected model that starts by flattening the data and 
# then runs it through two dense layers
library(keras)
model <- keras_model_sequential() %>% 
  layer_flatten(input_shape = c(lookback / step, dim(data)[-1])) %>% 
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dense(units = 1)

# check input dimensions
lookback/step # 240
dim(data)[-1] # 14
dim(data)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 20,
  validation_data = val_gen,
  validation_steps = val_steps
)

# plot loss curves
plot(history) # resinstall 'digest' package

# the simple well-performing baseline might be unlearnable, even if it's 
# technically part of the hypothesis space. 

# --- first recurrent baseline(GRU gated recurrent unit) --- #
model <- keras_model_sequential() %>% 
  layer_gru(units = 32, input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 20,
  validation_data = val_gen,
  validation_steps = val_steps
)


# plot loss curve again --- remember baseline value is mae = 0.28
plot(history)

# common sense baseline, such demonstrating the value of machine learning here, 
# as well as the superiority of recurrent networks compared to 
# sequence-flattening dense networks on this type of task.

# --- rnn(GRU) with drop-out --- #
# Using the same dropout mask at every timestep allows the network to properly 
# propagate its learning error through time; a temporally random dropout mask 
# would instead disrupt this error signal and be harmful to the learning process.

model <- keras_model_sequential() %>% 
  layer_gru(units = 32, dropout = 0.2, recurrent_dropout = 0.2,
            input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 30,
  validation_data = val_gen,
  validation_steps = val_steps
)

plot(history)

# no longer overfitting during the first 30 epochs
# but no big improvement in result mae

# --- universal machine learning workflow --- #
# it is a generally a good idea to increase the capacity of your network until 
# overfitting becomes your primary obstacle (assuming that you are already 
# taking basic steps to mitigate overfitting, such as using dropout). 

# all intermediate layers should return their full sequence of outputs 
# (a 3D tensor) rather than their output at the last timestep

# --- rnn(GRU) with drop-out - more layers --- #
model <- keras_model_sequential() %>% 
  layer_gru(units = 32, 
            dropout = 0.1, 
            recurrent_dropout = 0.5,
            return_sequences = TRUE,
            input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_gru(units = 64, activation = "relu",
            dropout = 0.1,
            recurrent_dropout = 0.5) %>% 
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 30,
  validation_data = val_gen,
  validation_steps = val_steps
)

plot(history)

# --- need to run the following --- #
# improve ours results by a bit, although not very significantly
# --- bidirectional rnn --- #
#  A bidirectional RNN exploits the order-sensitivity of RNNs: it simply 
# consists of two regular RNNs, such as the GRU or LSTM layers that you are 
# already familiar with, each processing input sequence in one direction 
# (chronologically and antichronologically), then merging their representations.
# a bidirectional RNN is able to catch patterns that may have been overlooked 
# by a one-direction RNN.

reverse_order_generator <- function( data, lookback, delay, min_index, max_index,
                                     shuffle = FALSE, batch_size = 128, step = 6) {
  if (is.null(max_index))
    max_index <- nrow(data) - delay - 1
  i <- min_index + lookback
  function() {
    if (shuffle) {
      rows <- sample(c((min_index+lookback):max_index), size = batch_size)
    } else {
      if (i + batch_size >= max_index)
        i <<- min_index + lookback
      rows <- c(i:min(i+batch_size, max_index))
      i <<- i + length(rows)
    }
    
    samples <- array(0, dim = c(length(rows), 
                                lookback / step,
                                dim(data)[[-1]]))
    targets <- array(0, dim = c(length(rows)))
    
    for (j in 1:length(rows)) {
      indices <- seq(rows[[j]] - lookback, rows[[j]], 
                     length.out = dim(samples)[[2]])
      samples[j,,] <- data[indices,]
      targets[[j]] <- data[rows[[j]] + delay,2]
    }            
    
    list(samples[,ncol(samples):1,], targets)
  }
}

train_gen_reverse <- reverse_order_generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 1,
  max_index = 200000,
  shuffle = TRUE,
  step = step, 
  batch_size = batch_size
)

val_gen_reverse = reverse_order_generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 200001,
  max_index = 300000,
  step = step,
  batch_size = batch_size
)

# --- (GRU) with bidirectional --- #
model <- keras_model_sequential() %>% 
  layer_gru(units = 32, input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

history <- model %>% fit_generator(
  train_gen_reverse,
  steps_per_epoch = 500,
  epochs = 10, # changed to 10 from 30
  validation_data = val_gen_reverse,
  validation_steps = val_steps
)

plot(history)
# worse result than common sense baseline | observed >0.45

# --- LSTM with bidirectional --- #
model <- keras_model_sequential() %>% 
  bidirectional(
    layer_gru(units = 32), input_shape = list(NULL, dim(data)[[-1]])
  ) %>% 
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 10, # changed to 10 from 40
  validation_data = val_gen,
  validation_steps = val_steps
)

plot(history)
# mae 0.28
