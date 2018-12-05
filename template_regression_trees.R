## 67.3 Regression tree
## Reference
## https://rafalab.github.io/dsbook/index.html 
library(dslabs)
library(rpart)
library(tidyverse)
fit <- rpart(margin ~ ., data = polls_2008)

## plot the tree with text
plot(fit, margin = 0.1)
text(fit, cex = 0.75)

## plot the margin vs. day with fitted line
polls_2008 %>% 
  mutate(y_hat = predict(fit)) %>% 
  ggplot() +
  geom_point(aes(day, margin)) +
  geom_step(aes(day, y_hat), col="red")

# So how do we pick cp? We can use cross validation just like with any tuning parameter.
library(caret)
train_rpart <- train(margin ~ ., 
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0, 0.05, len = 25)),
                     data = polls_2008)
ggplot(train_rpart)

# To see the resulting tree, we access the finalModel and plot it:
plot(train_rpart$finalModel, margin = 0.1)
text(train_rpart$finalModel, cex = 0.75)

polls_2008 %>% 
  mutate(y_hat = predict(train_rpart)) %>% 
  ggplot() +
  geom_point(aes(day, margin)) +
  geom_step(aes(day, y_hat), col="red")
