# function API for two-input Q&A model

library(keras)
text_vocabulary_size <- 10000
ques_vocabulary_size <- 10000
answer_vocabulary_size <- 500

# construct two-input
text_input <- layer_input(shape = list(NULL),
                          dtype = "int32", name = "text")

encoded_text <- text_input %>%
  layer_embedding(input_dim = text_vocabulary_size+1, output_dim = 32) %>%
  layer_lstm(units = 32)

question_input <- layer_input(shape = list(NULL),
                              dtype = "int32", name = "question")

encoded_question <- question_input %>%
  layer_embedding(input_dim = ques_vocabulary_size+1, output_dim = 16) %>%
  layer_lstm(units = 16)

# combine two into one : tex + question 
concatenated <- layer_concatenate(list(encoded_text, encoded_question))

# this leads to 'answer'
answer <- concatenated %>%
  layer_dense(units = answer_vocabulary_size, activation = "softmax")

# construct model
model <- keras_model(list(text_input, question_input), answer)
model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("acc")
)

# generate psuedo input data
num_samples <- 1000
max_length <- 100
random_matrix <- function(range, nrow, ncol) {
  matrix(sample(range, size = nrow * ncol, replace = TRUE),
         nrow = nrow, ncol = ncol)
}
text <- random_matrix(1:text_vocabulary_size, num_samples, max_length)
question <- random_matrix(1:ques_vocabulary_size, num_samples, max_length)
answers <- random_matrix(0:1, num_samples, answer_vocabulary_size)

# fit model on data
model %>% fit(
  list(text, question), answers,
  epochs = 10, batch_size = 128
)

model %>% fit(
  list(text = text, question = question), answers,
  epochs = 10, batch_size = 128
)

# model runs now after being corrected from Errata, Jan 14, 2019
# https://manning-content.s3.amazonaws.com/download/1/6d1e9c3-9483-455c-8a47-54bb4bbd3205/Allaire_DeepLearningwithR_err4.html