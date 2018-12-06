## General Linear Models The Basics
## Simulating ideal data for a general linear model
## https://www.r-bloggers.com/general-linear-models-the-basics/amp/
n <- 100
beta <- 2.2
alpha <- 30

## We also need some covariate data, we will just generate a sequence of n numbers from 0 to 1:
x <- seq(0, 1, length.out = n)

## The model's expectation is thus this straight line:

y_true <- beta * x + alpha
plot(x, y_true)

## Let's generate some error:

sigma <- 2.4
set.seed(42)
error <- rnorm(n, sd = sigma)
y_obs <- y_true + error
plot(x, y_obs)
lines(x, y_true)

## Also, check out the errors:

hist(error)

## Fitting a model

m1 <- lm(y_obs ~ x)
coef(m1)
summary(m1)
# Finally, notice
# that the Residual standard error is close to the value we used for
# sigma, which is because it is an estimate of sigma from our
# simulated data.

## residuals vs fitted values

plot(m1, 1)
# We are looking for 'heteroskedasticity' which is a fancy way of
# saying the errors aren't equal across the range of predictions (remember
# I said sigma is a constant?).

## residuals distribution
plot(m1, 2)
