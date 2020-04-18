# how to install Keras R version on Macbooks in RStudio
# https://tensorflow.rstudio.com/installation/
# https://keras.rstudio.com/

install.packages("tensorflow")
library(tensorflow)
install_tensorflow()

library(tensorflow)
tf$constant("Hellow Tensorflow")

devtools::install_github("rstudio/keras")

library(keras)
install_keras()
