# Reference : https://blogs.rstudio.com/tensorflow/posts/2018-07-17-activity-detection/
# setwd("~/Google Drive/Examples_R/HAPT Data Set")

library(keras)     # Neural Networks
library(tidyverse) # Data cleaning / Visualization
library(knitr)     # Table printing
library(rmarkdown) # Misc. output utilities 
library(ggridges)  # Visualization

# Activity labels
activityLabels <- read.table("activity_labels.txt", 
                             col.names = c("number", "label")) 

activityLabels %>% kable(align = c("c", "l"))

# Column 1: experiment number ID, 
# Column 2: user number ID, 
# Column 3: activity number ID 
# Column 4: Label start point 
# Column 5: Label end point 

labels <- read.table(
  "RawData/labels.txt",
  col.names = c("experiment", "userId", "activity", "startPos", "endPos")
)

labels %>% 
  head(50) %>% 
  paged_table()

# File names
dataFiles <- list.files("RawData")
dataFiles %>% head()

fileInfo <- data_frame(filePath = dataFiles) %>%
  filter(filePath != "labels.txt") %>% 
  separate(filePath, sep = '_', 
           into = c("type", "experiment", "userId"), 
           remove = FALSE) %>% 
  mutate(experiment = str_remove(experiment, "exp"),
    userId = str_remove_all(userId, "user|\\.txt")) %>% 
  spread(type, filePath)

fileInfo %>% head() %>% kable()

# Reading and gathering data
# Read contents of single file to a dataframe with accelerometer and gyro data.
readInData <- function(experiment, userId){
  genFilePath = function(type) {
    paste0("RawData/", type, "_exp",experiment, "_user", userId, ".txt")
  }  
  
  bind_cols(
    read.table(genFilePath("acc"), col.names = c("a_x", "a_y", "a_z")),
    read.table(genFilePath("gyro"), col.names = c("g_x", "g_y", "g_z"))
  )
}

# Function to read a given file and get the observations contained along
# with their classes.

loadFileData <- function(curExperiment, curUserId) {
  
  # load sensor data from file into dataframe
  allData <- readInData(curExperiment, curUserId)
  
  extractObservation <- function(startPos, endPos){
    allData[startPos:endPos,]
  }
  
  # get observation locations in this file from labels dataframe
  dataLabels <- labels %>% 
    filter(userId == as.integer(curUserId), 
           experiment == as.integer(curExperiment))
  
  
  # extract observations as dataframes and save as a column in dataframe.
  dataLabels %>% 
    mutate(
      data = map2(startPos, endPos, extractObservation)
    ) %>% 
    select(-startPos, -endPos)
}

# scan through all experiment and userId combos and gather data into a dataframe. 
allObservations <- map2_df(fileInfo$experiment, fileInfo$userId, loadFileData) %>% 
  right_join(activityLabels, by = c("activity" = "number")) %>% 
  rename(activityName = label)

# cache work. 
write_rds(allObservations, "allObservations.rds")
allObservations %>% dim()

# Exploring the data
allObservations %>% 
  mutate(recording_length = map_int(data,nrow)) %>% 
  ggplot(aes(x = recording_length, y = activityName)) +
  geom_density_ridges(alpha = 0.8)

# Filtering activities
desiredActivities <- c(
  "STAND_TO_SIT", "SIT_TO_STAND", "SIT_TO_LIE", 
  "LIE_TO_SIT", "STAND_TO_LIE", "LIE_TO_STAND"  
)

filteredObservations <- allObservations %>% 
  filter(activityName %in% desiredActivities) %>% 
  mutate(observationId = 1:n())

filteredObservations %>% paged_table()

# Training/testing split
# get all users
userIds <- allObservations$userId %>% unique()

# randomly choose 24 (80% of 30 individuals) for training
set.seed(42) # seed for reproducibility
trainIds <- sample(userIds, size = 24)

# set the rest of the users to the testing set
testIds <- setdiff(userIds,trainIds)

# filter data. 
trainData <- filteredObservations %>% 
  filter(userId %in% trainIds)

testData <- filteredObservations %>% 
  filter(userId %in% testIds)

# Visualizing activities
unpackedObs <- 1:nrow(trainData) %>% 
  map_df(function(rowNum){
    dataRow <- trainData[rowNum, ]
    dataRow$data[[1]] %>% 
      mutate(
        activityName = dataRow$activityName, 
        observationId = dataRow$observationId,
        time = 1:n() )
  }) %>% 
  gather(reading, value, -time, -activityName, -observationId) %>% 
  separate(reading, into = c("type", "direction"), sep = "_") %>% 
  mutate(type = ifelse(type == "a", "acceleration", "gyro"))

unpackedObs %>% 
  ggplot(aes(x = time, y = value, color = direction)) +
  geom_line(alpha = 0.2) +
  geom_smooth(se = FALSE, alpha = 0.7, size = 0.5) +
  facet_grid(type ~ activityName, scales = "free_y") +
  theme_minimal() +
  theme( axis.text.x = element_blank() )

# Preprocessing
# Padding observations
padSize <- trainData$data %>% 
  map_int(nrow) %>% 
  quantile(p = 0.98) %>% 
  ceiling()
padSize

convertToTensor <- . %>% 
  map(as.matrix) %>% 
  pad_sequences(maxlen = padSize)

trainObs <- trainData$data %>% convertToTensor()
testObs <- testData$data %>% convertToTensor()

dim(trainObs)

# One-hot encoding
oneHotClasses <- . %>% 
{. - 7} %>%        # bring integers down to 0-6 from 7-12
  to_categorical() # One-hot encode

trainY <- trainData$activity %>% oneHotClasses()
testY <- testData$activity %>% oneHotClasses()

# Modeling
# Architecture
input_shape <- dim(trainObs)[-1]
num_classes <- dim(trainY)[2]

filters <- 24     # number of convolutional filters to learn
kernel_size <- 8  # how many time-steps each conv layer sees.
dense_size <- 48  # size of our penultimate dense layer. 

# Initialize model
model <- keras_model_sequential()
model %>% 
  layer_conv_1d(
    filters = filters,
    kernel_size = kernel_size, 
    input_shape = input_shape,
    padding = "valid", 
    activation = "relu"
  ) %>%
  layer_batch_normalization() %>%
  layer_spatial_dropout_1d(0.15) %>% 
  layer_conv_1d(
    filters = filters/2,
    kernel_size = kernel_size,
    activation = "relu",
  ) %>%
  # Apply average pooling:
  layer_global_average_pooling_1d() %>% 
  layer_batch_normalization() %>%
  layer_dropout(0.2) %>% 
  layer_dense(
    dense_size,
    activation = "relu"
  ) %>% 
  layer_batch_normalization() %>%
  layer_dropout(0.25) %>% 
  layer_dense(
    num_classes, 
    activation = "softmax",
    name = "dense_output"
  ) 

summary(model)

# Training
# Compile model
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "rmsprop",
  metrics = "accuracy"
)

trainHistory <- model %>%
  fit(
    x = trainObs, y = trainY,
    epochs = 350,
    validation_data = list(testObs, testY),
    callbacks = list(
      callback_model_checkpoint("best_model.h5", 
                                save_best_only = TRUE)
    )
  )

#Evaluation
# dataframe to get labels onto one-hot encoded prediction columns
oneHotToLabel <- activityLabels %>% 
  mutate(number = number - 7) %>% 
  filter(number >= 0) %>% 
  mutate(class = paste0("V",number + 1)) %>% 
  select(-number)

# Load our best model checkpoint
bestModel <- load_model_hdf5("best_model.h5")

tidyPredictionProbs <- bestModel %>% 
  predict(testObs) %>% 
  as_data_frame() %>% 
  mutate(obs = 1:n()) %>% 
  gather(class, prob, -obs) %>% 
  right_join(oneHotToLabel, by = "class")

predictionPerformance <- tidyPredictionProbs %>% 
  group_by(obs) %>% 
  summarise(
    highestProb = max(prob),
    predicted = label[prob == highestProb]
  ) %>% 
  mutate(
    truth = testData$activityName,
    correct = truth == predicted
  ) 

predictionPerformance %>% paged_table()

predictionPerformance %>% 
  mutate(result = ifelse(correct, 'Correct', 'Incorrect')) %>% 
  ggplot(aes(highestProb)) +
  geom_histogram(binwidth = 0.01) +
  geom_rug(alpha = 0.5) +
  facet_grid(result~.) +
  ggtitle("Probabilities associated with prediction by correctness")

predictionPerformance %>% 
  group_by(truth, predicted) %>% 
  summarise(count = n()) %>% 
  mutate(good = truth == predicted) %>% 
  ggplot(aes(x = truth,  y = predicted)) +
  geom_point(aes(size = count, color = good)) +
  geom_text(aes(label = count), 
            hjust = 0, vjust = 0, 
            nudge_x = 0.1, nudge_y = 0.1) + 
  guides(color = FALSE, size = FALSE) +
  theme_minimal()

