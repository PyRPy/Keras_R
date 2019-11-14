# Dynamic linear models with tfprobability

# Dynamic linear regression example: Capital Asset Pricing Model ---------

devtools::install_github("rstudio/tfprobability")
# tensorflow::install_tensorflow(version = "2.0.0")
library(tensorflow)
library(tfprobability)
library(reticulate)
library(tidyverse)
library(zeallot)


# load data ---------------------------------------------------------------

df <- read_table(
  "capm.txt",
  col_types = list(X1 = col_date(format = "%Y.%m"))) %>%
  rename(month = X1)
df %>% glimpse()


# plot data ---------------------------------------------------------------

df %>% gather(key = "symbol", value = "return", -month) %>%
  ggplot(aes(x = month, y = return, color = symbol)) +
  geom_line() +
  facet_grid(rows = vars(symbol), scales = "free")


# linear model ------------------------------------------------------------

# excess returns of the asset under study
ibm <- df$IBM - df$RKFREE

# market excess returns
x <- df$MARKET - df$RKFREE

fit <- lm(ibm ~ x)
summary(fit)

# IBM is found to be a conservative investment, the slope being ~ 0.5. But 
# is this relationship stable over time?


# divide the dataset into a training and a testing part -------------------

# zoom in on ibm
ts <- ibm %>% matrix()

# forecast 12 months
n_forecast_steps <- 12
ts_train <- ts[1:(length(ts) - n_forecast_steps), 1, drop = FALSE]

# make sure we work with float32 here
ts_train <- tf$cast(ts_train, tf$float32)
ts <- tf$cast(ts, tf$float32)

# define the model on the complete series
linreg <- ts %>%
  sts_dynamic_linear_regression(
    design_matrix = cbind(rep(1, length(x)), x) %>% tf$cast(tf$float32)
  )

## Error: ModuleNotFoundError: No module named 'tensorflow_probability'
## due to version mismatch ???