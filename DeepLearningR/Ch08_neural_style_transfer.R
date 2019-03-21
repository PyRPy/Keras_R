# neural style transfer
library(keras)
# This is the path to the image you want to transform.
target_image_path <- "style_transfer/creative_commons_elephant.png" 
# This is the path to the style image.
style_reference_image_path <- "style_transfer/tulip_flower.png"

# Dimensions of the generated picture.
img <- image_load(target_image_path)

width <- img$size[[1]]
height <- img$size[[2]]
img_nrows <- 400
img_ncols <- as.integer(width * img_nrows / height)  

# auxiliary functions for loading, pre-processing and post-processing the images 
# that will go in and out of the VGG19 convnet
preprocess_image <- function(path) {
  img <- image_load(path, target_size = c(img_nrows, img_ncols)) %>%
    image_to_array() %>%
    array_reshape(c(1, dim(.)))
  imagenet_preprocess_input(img)
}

deprocess_image <- function(x) {
  x <- x[1,,,]
  # Remove zero-center by mean pixel
  x[,,1] <- x[,,1] + 103.939
  x[,,2] <- x[,,2] + 116.779
  x[,,3] <- x[,,3] + 123.68
  # 'BGR'->'RGB'
  x <- x[,,c(3,2,1)]
  x[x > 255] <- 255
  x[x < 0] <- 0
  x[] <- as.integer(x)/255
  x
}

target_image <- k_constant(preprocess_image(target_image_path))
style_reference_image <- k_constant(
  preprocess_image(style_reference_image_path)
)
# This placeholder will contain our generated image
combination_image <- k_placeholder(c(1, img_nrows, img_ncols, 3)) 

# We combine the 3 images into a single batch
input_tensor <- k_concatenate(list(target_image, style_reference_image, 
                                   combination_image), axis = 1)
# We build the VGG19 network with our batch of 3 images as input.
# The model will be loaded with pre-trained ImageNet weights.
model <- application_vgg19(input_tensor = input_tensor, 
                           weights = "imagenet", 
                           include_top = FALSE)

cat("Model loaded\n")
# content loss
content_loss <- function(base, combination) {
  k_sum(k_square(combination - base))
}

# style loss
gram_matrix <- function(x) {
  features <- k_batch_flatten(k_permute_dimensions(x, c(3, 1, 2)))
  gram <- k_dot(features, k_transpose(features))
  gram
}

style_loss <- function(style, combination){
  S <- gram_matrix(style)
  C <- gram_matrix(combination)
  channels <- 3
  size <- img_nrows*img_ncols
  k_sum(k_square(S - C)) / (4 * channels^2  * size^2)
}

# total loss / regulariztion loss
total_variation_loss <- function(x) {
  y_ij  <- x[,1:(img_nrows - 1L), 1:(img_ncols - 1L),]
  y_i1j <- x[,2:(img_nrows), 1:(img_ncols - 1L),]
  y_ij1 <- x[,1:(img_nrows - 1L), 2:(img_ncols),]
  a <- k_square(y_ij - y_i1j)
  b <- k_square(y_ij - y_ij1)
  k_sum(k_pow(a + b, 1.25))
}

# Named list mapping layer names to activation tensors
outputs_dict <- lapply(model$layers, `[[`, "output")
names(outputs_dict) <- lapply(model$layers, `[[`, "name")

# Name of layer used for content loss
content_layer <- "block5_conv2" 
# Name of layers used for style loss
style_layers = c("block1_conv1", "block2_conv1",
                 "block3_conv1", "block4_conv1",
                 "block5_conv1")
# Weights in the weighted average of the loss components
total_variation_weight <- 1e-4
style_weight <- 1.0
content_weight <- 0.025
# Define the loss by adding all components to a `loss` variable
loss <- k_variable(0.0) 
layer_features <- outputs_dict[[content_layer]] 
target_image_features <- layer_features[1,,,]
combination_features <- layer_features[3,,,]
loss <- loss + content_weight * content_loss(target_image_features,
                                             combination_features)
for (layer_name in style_layers){
  layer_features <- outputs_dict[[layer_name]]
  style_reference_features <- layer_features[2,,,]
  combination_features <- layer_features[3,,,]
  sl <- style_loss(style_reference_features, combination_features)
  loss <- loss + ((style_weight / length(style_layers)) * sl)
}

loss <- loss + 
  (total_variation_weight * total_variation_loss(combination_image))