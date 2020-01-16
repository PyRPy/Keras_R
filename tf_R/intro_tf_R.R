# introduction to tensorflow
library(tensorflow)
# tf_config()
session <- tf$Session()
a <- tf$constant(5, name = "NumAdults")
b <- tf$constant(6, name = "NumChildren")
c <- tf$add(a, b)
print(session$run(c))
