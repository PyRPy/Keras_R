# multi-output models - template only, no inputs
# --- three-output model --- #
library(keras)
vocabulary_size <- 50000
num_income_groups <- 10
posts_input <- layer_input(shape = list(NULL),
                           dtype = "int32", name = "posts")

embedded_posts <- posts_input %>%
  layer_embedding(input_dim = 256, output_dim = vocabulary_size)

base_model <- embedded_posts %>%
  layer_conv_1d(filters = 128, kernel_size = 5, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 5) %>%
  layer_conv_1d(filters = 256, kernel_size = 5, activation = "relu") %>%
  layer_conv_1d(filters = 256, kernel_size = 5, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 5) %>%
  layer_conv_1d(filters = 256, kernel_size = 5, activation = "relu") %>%
  layer_conv_1d(filters = 256, kernel_size = 5, activation = "relu") %>%
  layer_global_max_pooling_1d() %>%
  layer_dense(units = 128, activation = "relu")

# output layers are given names
age_prediction <- base_model %>%
  layer_dense(units = 1, name = "age")
income_prediction <- base_model %>%
  layer_dense(num_income_groups, activation = "softmax", name = "income")
gender_prediction <- base_model %>%
  layer_dense(units = 1, activation = "sigmoid", name = "gender")
model <- keras_model(
  posts_input,
  list(age_prediction, income_prediction, gender_prediction)
)

# with multiple losses
model %>% compile(
  optimizer = "rmsprop",
  loss = c("mse", "categorical_crossentropy", "binary_crossentropy")
)
# equivalent - give names to the output layers
# model %>% compile(
#   optimizer = "rmsprop",
#   loss = list(
#     age = "mse",
#     income = "categorical_crossentropy",
#     gender = "binary_crossentropy"
#   )
# )

# loss weighting

model %>% compile(
  optimizer = "rmsprop",
  loss = c("mse", "categorical_crossentropy", "binary_crossentropy"),
  loss_weights = c(0.25, 1, 10)
)

# model %>% compile(
#   optimizer = "rmsprop",
#   loss = list(
#     age = "mse",
#     income = "categorical_crossentropy",
#     gender = "binary_crossentropy"
#   ),
#   loss_weights = list(
#     age = 0.25,
#     income = 1,
#     gender = 10
#   )
# )

# feed data - psuedo code
# age_targets, income_targets, gender_targets are 'R arrays'
model %>% fit(
  posts, list(age_targets, income_targets, gender_targets),
  epochs = 10, batch_size = 64
)

# model %>% fit(
#   posts, list(
#     age = age_targets,
#     income = income_targets,
#     gender = gender_targets
#   ),
#   epochs = 10, batch_size = 64
# )



