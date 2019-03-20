# call back templates
library(keras)

# Interrupts training when accuracy has stopped improving for more than one
# epoch (that is, two epochs)
callbacks_list <- list(
  callback_early_stopping(
    monitor = "acc",
    patience = 1
  )
)

callback_model_checkpoint(
  filepath = "my_model.h5",
  monitor = "val_loss",
  save_best_only = TRUE
)


# monitor accuracy
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)
model %>% fit(
  x, y,
  epochs = 10,
  batch_size = 32,
  callbacks = callbacks_list,
  validation_data = list(x_val, y_val)
)

# change in learning rate on plateau
callbacks_list <- list(
  callback_reduce_lr_on_plateau(
    monitor = "val_loss",
    factor = 0.1,
    patience = 10
  )
)

model %>% fit(
  x, y,
  epochs = 10,
  batch_size = 32,
  callbacks = callbacks_list,
  validation_data = list(x_val, y_val)
)