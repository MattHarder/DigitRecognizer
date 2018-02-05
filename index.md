---
title: "DigitRecognizeR"
author: "Matthew Harder"
date: "2/1/2018"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Digit Recognizer


First thing we're going to need is the libraries for mxnet and caTools. Mxnet isn't on CRAN so it needs to be downloaded on its own. Then we need to load the data and set the seed for reproducible results while training.
```{r cars}
library(caTools)
library(mxnet)

# Load data
df.train <- read.csv('train.csv')
df.test <- read.csv('test.csv')

# Set seed
set.seed(101)
```

### Training and Test Data 

Now that we've got the data, we need to do a train-test split on the training data so we can see test our model without having to submit results to Kaggle. 

```{r pressure, echo=FALSE}
samples <- sample.split(df.train$label, 0.7)
df.sample <- df.train[samples, ]
df.sample.test <- df.train[!samples, ]
```

### Preparing Data
Next we need to put data back into its original 28 x 28 pixel form so the convolutional layers can work. I've commented out or slightly changed the code from when I was building and testing the model.
```{r}
# Put training data into arrays
train_x <- t(df.train[, -1])
train_y <- df.train[, 1]
train_array <- train_x
dim(train_array) <- c(28, 28, 1, ncol(train_x))

# Put testing data into arrays
test_x <- t(df.test)
# test_y <- df.test[, 1]
test_array <- test_x
dim(test_array) <- c(28, 28, 1, ncol(test_x))
```

### Model Setup
Almost there, time to set up how we want the neural net model to run. I've chosen to use 3 convolutional layers followed by 2 fully connected layers.
```{r}
# Choose activation and pooling
activ <- 'relu'
pooling <- 'max'

# Symbolic Model
data <- mx.symbol.Variable('data')
# 1st convolution layer
conv.1 <- mx.symbol.Convolution(data = data, kernel = c(5,5), num_filter = 20)
bn.1 <- mx.symbol.BatchNorm(data = conv.1)
activ.1 <- mx.symbol.Activation(data = bn.1, act_type = activ)
# 2nd convolution layer
conv.2 <- mx.symbol.Convolution(data = activ.1, kernel = c(5,5), num_filter = 20)
bn.2 <- mx.symbol.BatchNorm(data = conv.2)
activ.2 <- mx.symbol.Activation(data = bn.2, act_type = activ)
# 3rd convolution layer
conv.3 <- mx.symbol.Convolution(data = activ.2, kernel = c(5,5), num_filter = 20)
bn.3 <- mx.symbol.BatchNorm(data = conv.3)
activ.3 <- mx.symbol.Activation(data = bn.3, act_type = activ)
# 1st fully connected layer
flatten <- mx.symbol.Flatten(data = activ.3)
fc.1 <- mx.symbol.FullyConnected(data = flatten, num.hidden = 500)
activ.4 <- mx.symbol.Activation(data = fc.1, act_type = activ)
# 2nd fully connected layer
fc.2 <- mx.symbol.FullyConnected(data = activ.4, num_hidden = 40)
activ.5 <- mx.symbol.Activation(data = fc.2, act_type = activ)

# Output
NN.model <- mx.symbol.SoftmaxOutput(data = activ.5)
```

### Running Model
Finally time to run the model. This part of the model takes a while so I hope you don't need your computer any time soon.
```{r}
# Set up CPU to be used
devices <- mx.cpu()

# Train
model <- mx.model.FeedForward.create(NN.model, X = train_array, y= train_y, ctx = devices,
                                     num.round = 100, array.batch.size = 40, learning.rate = 0.01,
                                     momentum = 0.9, eval.metric = mx.metric.accuracy,
                                     epoch.end.callback = mx.callback.log.train.metric(200))
```

### Testing
I've included the code for testing the model on the training split data too, but that is commented out. The results are output into a .csv file ready for upload.
```{r}
predicted <- predict(model, test_array)
# Assign labels
predicted_labels <- max.col(t(predicted)) - 1
# Get accuracy
# print(sum(diag(table(test_y, predicted_labels)))/12600)

submission <- data.frame(ImageId=1:ncol(test_x), Label=predicted_labels)
write.csv(submission, file = 'submission.csv', row.names = FALSE, quote = FALSE)
```



