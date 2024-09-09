
# Timeseries anomaly detection using an Autoencoder -----------------------

library(dplyr, warn.conflicts = FALSE)
library(ggplot2)
theme_set(theme_minimal())

library(listarrays)
library(tfdatasets, exclude = c("shape"))
library(keras3)

### get data
get_data <- function(file) {
  readr::read_csv(file, col_types = "Td")
}



# put the data in the working directory
df_small_noise   <- get_data("art_daily_small_noise.csv")
head(df_small_noise)
df_daily_jumpsup <- get_data("art_daily_jumpsup.csv")
head(df_daily_jumpsup)

### Visualize the data
plot_ts <- function(df) {
  ggplot(df, aes(x = timestamp, y = value)) + geom_line() +
    scale_x_datetime(date_breaks = "1 day", date_labels = "%b-%d")
}

# no anomoly
plot_ts(df_small_noise) + ggtitle("Without Anomaly")

# with anomalies
plot_ts(df_daily_jumpsup) + ggtitle("With Anomaly")

### Prepare training data
# 24 * 60 / 5 = 288 timesteps per day
# 288 * 14 = 4032 data points in total

df_train <- df_small_noise |>
  mutate(value = (value - mean(value)) / sd(value))

cat("Number of training samples:", nrow(df_train), "\n")

### Create sequences
TIME_STEPS <- 288

as_dataset <- function(df) {
  x <- as.matrix(df$value)
  ds <- timeseries_dataset_from_array(x, NULL, sequence_length = TIME_STEPS)
  # Because the dataset is small, cast TF Dataset to an R array for convenience.
  ds |> as_array_iterator() |> iterate() |> bind_on_rows()
}

x_train <- as_dataset(df_train)
writeLines(sprintf("Training input shape: (%s)", toString(dim(x_train))))

### convolutional reconstruction autoencoder model.
model <- keras_model_sequential(input_shape = c(TIME_STEPS, 1)) |>
  layer_conv_1d(
    filters = 32, kernel_size = 7, padding = "same",
    strides = 2, activation = "relu"
  ) |>
  layer_dropout(rate = 0.2) |>
  layer_conv_1d(
    filters = 16, kernel_size = 7, padding = "same",
    strides = 2, activation = "relu"
  ) |>
  layer_conv_1d_transpose(
    filters = 16, kernel_size = 7, padding = "same",
    strides = 2, activation = "relu"
  ) |>
  layer_dropout(rate = 0.2) |>
  layer_conv_1d_transpose(
    filters = 32, kernel_size = 7, padding = "same",
    strides = 2, activation = "relu"
  ) |>
  layer_conv_1d_transpose(filters = 1, kernel_size = 7, padding = "same")

model |> compile(optimizer=optimizer_adam(learning_rate=0.001), loss="mse")
model

### Train the model
history = model |> fit(
  x_train, x_train,
  epochs = 10,
  validation_split = 0.1,
  callbacks = c(
    callback_early_stopping(
      monitor = "val_loss", patience = 5, mode = "min"
    )
  )
)

### track the training
plot(history)

### Detecting anomalies
# Get train MAE loss.
x_train_pred <- model |> predict(x_train)

train_mae_loss <- apply(abs(x_train_pred - x_train), 1, mean)

hist(train_mae_loss, breaks = 50)

# Get reconstruction loss threshold.
threshold <- max(train_mae_loss)
cat("Reconstruction error threshold: ", threshold, "\n")

### Compare recontruction
# Checking how the first sequence is learnt
plot(NULL, NULL, ylab = 'Value',
     xlim = c(0, TIME_STEPS),
     ylim = range(c(x_train[1,,], x_train_pred[1,,])))
lines(x_train[1,,])
lines(x_train_pred[1,,], col = 'red')
legend("topleft", lty = 1,
       legend = c("actual", "predicted"),
       col = c("black", "red"))

### Prepare test data
df_test <- df_daily_jumpsup |>
  mutate(value =
           (value - mean(df_small_noise$value)) /
           sd(df_small_noise$value))

df_test |> head()
plot_ts(df_test)

# Create sequences from test values.
x_test <- as_dataset(df_test)

# Get test MAE loss.
x_test_pred <- model |> predict(x_test)
test_mae_loss <- apply(abs(x_test_pred - x_test), 1, mean)

hist(test_mae_loss, breaks = 50, xlab = "test MAE loss", ylab = "No of samples")

# Detect all the samples which are anomalies.
anomalies <- test_mae_loss > threshold
cat("Number of anomaly samples:", sum(anomalies), "\n")
cat("Indices of anomaly samples:", which(anomalies), "\n", fill = TRUE)

is_anomaly <- test_mae_loss > threshold
is_anomaly <- is_anomaly &
  zoo::rollsum(is_anomaly, TIME_STEPS,
               align = "right", na.pad = TRUE) >= TIME_STEPS

with(df_test, {
  plot(value ~ timestamp, type = 'l', xaxt = 'n', las = 2)
  axis.POSIXct(1, at = seq(timestamp[1], tail(timestamp, 1), by = "days"),
               format = "%b-%d")
})

with(df_test[which(is_anomaly),], {
  points(value ~ timestamp, col = "red")
})
