#### Packages #### 
library(dummies)
library(olsrr)
library(car)
library(dplyr)
library(e1071)
library(neuralnet)
library(randomForest)
library(MASS)
library(faraway)
library(ggplot2)
library(ggmosaic)
library(plotrix)
library(corrplot)
library(formattable)

#### Data Preparation ####
data <- read.csv("AppleStore.csv",header = TRUE)head(data)
str(data)
dim(data)

data <- data[,-c(1,3,5,11)] # remove x, track_name, currency, and version. 
colnames(data) #looks cool now!

data[,13] = factor(data[,13]) # vpp_lic is from integer to factor

# Let's manipulate variable 8 and 9 to run ML algorithms better!
data[,9] = gsub('\\s+', '',data[,9])
for(i in 1:nrow(data)){
  data[i,9] = unlist(strsplit(data[i,9], split = "&", fixed = TRUE))[1]
}

data[,8] = as.character(data[,8])
for(i in 1:nrow(data)){
  data[i,8] = unlist(strsplit(data[i,8], split = "+", fixed = TRUE))
}
data[,8] <- as.factor(data[,8])

#### Explanatory Data Analysis ####
colnames(data)
head(data)

# Size Distribution: Most are small
sizeDistribution <- ggplot(data = data, 
                           aes(x = size_bytes)) + geom_histogram(fill = 'royalblue2')
sizeDistribution


# Price distrubiton: Most apps do not cost a thing and we find that there is no outlier. 
priceDistribution <- ggplot(aes(x = price), data = data)+
  geom_histogram(fill = 'royalblue2', binwidth = 3)+
  scale_y_log10()+
  ggtitle('price')
priceDistribution 

# Plotting a bar graph for categories: Game category dominates!
categoryCount <- ggplot(aes(x = prime_genre), data = data)+
  geom_bar(fill = 'royalblue2')+
  coord_flip()+
  ggtitle("Categories")
categoryCount

# Content rating distribution: Most of our data focus on applications that are for +4 ages. 
contRating <- ggplot(data = data, aes(x = cont_rating)) + geom_bar(fill = 'royalblue2')
contRating 

### What do we have so far? The most of applications in our data are game, for +4 age, cheap and have small size.

# Genre vs Rating: Games category gets high rating among the other types.
genreRating <- ggplot(data = data, aes(x = reorder(prime_genre,user_rating), y = user_rating)) + 
  geom_boxplot() + geom_point(position = "jitter") + theme(axis.text.x = element_text(angle = 90))
genreRating 

# Content Rating vs ipadSc_urls
contIpad <- ggplot(data) + geom_mosaic(aes(x = product(cont_rating),fill=factor(ipadSc_urls.num))) +
  labs(x = "cont_rating", y = "ipadSc_urls.num", title = "cont_rating - ipadSc_urls.num \n")  
contIpad

# Correlation between numerical variables: the correlation between user_rating and ipad is 0.18.
M <- cor(data.frame(data[,c(c(2:7,10:12))]))
corrplot(M, type = "upper", tl.pos = "d")
corrplot(M, add = TRUE, type = "lower", col = "Black", method = "number",
         diag = FALSE, tl.pos = "n", cl.pos = "n")

# Create dummy variable to be able to use categorical variables
data2 <- dummy.data.frame(data, sep = ".")
data2$vpp_lic <- as.factor(data2$vpp_lic)
str(data2)
dim(data2) # now the number of variables increased from 13 to 38.

data2 <- data2[,-1] # Remove id from the dataset
colnames(data2)
head(data2)

# All data was moved to the right on x axis for implementing Shapiro-Test.
data2[,5] = data2[,5] + 1 

# While creating machine learning model, I will train my model on some part of the data 
# and test the accuracy of model on the part of the data.
# Otherwise, I can face overlearning problem. 

# Test-Train Sets
set.seed(12349)
testNumbers <- sample(1:dim(data2)[1],size = dim(data2)[1] - 5000, replace = FALSE) 
testSet <- data2[testNumbers,]
trainSet <- data2[-testNumbers,]
dim(trainSet) # 5000 observation, 38 variables
dim(testSet)
#### Linear Regression Model #### 
# In linear model, my dependent variable which I want to predict is user_rating and I included all features in model1. 
# H0: The coefficients associated with the variables is equal to zero in Linear Model
# H1: the coefficients are not equal to zero 
model1 <- lm(user_rating ~ . , data= trainSet) 
summary(model1) # p value < 0.05, so we will reject the null hypothesis that means 
# there exists a relationship between the independent variable in question and the dependent variable


# As the output shows, the predictors jointly explain 61.5 of the observed variance on the dependent variable ???user_rating??? (adj-R2=.61); 
# It is tempting to interpret the 61.5 explained variance as ???high???, but whether this is appropriate depends on context.
# the amount of variance-explained differs significantly from zero, F (34,4965)=233.9
#  Because higher degrees of freedom generally mean larger sample sizes, 
# a higher degree of freedom means more power to reject a false null hypothesis and find a significant result.

# Stepwise Regression
# The stepwise regression (or stepwise selection) consists of iteratively adding and removing predictors, 
# in the predictive model, in order to 
# find the subset of variables in the data set resulting in the best performing model, that is a model that lowers prediction error.
out1 <- ols_step_forward_p(model1,details = TRUE)
print(out1)

# Train set had 37 variable and after implementing Stepwise regression, we have 17 variables which predict user_rating best!

n <- out1$predictors #let's see our predictors!

f <- as.formula(paste("user_rating ~",paste(n[!n %in% "user_rating_ver"],collapse = "+"))) # and form LM! 
print(f)

model2 <- lm(f, data = trainSet)
summary(model2)

# After implementing best variables to our model, there is no change at all between model1 and model2. 

# Does the model fit the data? Let's see
# H0: Residuals have normal distrubitions.
# H1: Residuals don't have normal distributions. 

shapiro.test(model2$residuals) # Residuals do not have normal dist. p value < 0.05, reject H0. 
qqnorm(model2$residuals);qqline(model2$residuals) # Transformation is needed!

boxcox(model2, plotit = TRUE) #Let's estimate the maximum likelihood function for linear transformation.

ols_plot_cooksd_chart(model1)
ols_plot_resid_lev(model1)

# Transformation
f.trans <- as.formula(paste("user_rating^2.5 ~",paste(n[!n %in% "user_rating"],collapse = "+"))) 
model3 <- lm(f.trans, data= trainSet)
summary(model3)

# H0: Residuals have normal distribution after transformation.
# H1: Residuals don't have normal dist.

shapiro.test(model3$residuals) # Reject H0, bec. p value < 0.05. 
qqnorm(model3$residuals);qqline(model3$residuals) 
boxcox(model3, plotit = TRUE,lambda = seq(-0.25, 1.75, by = 0.05))

#Assuming that residuals are normally distrributed. - however, they are not! 

# H0: There is multicollinearity problem.
# H1: There is no.
vif(model3) # VIF's value < 5, so reject H0. 
# Because as a rule of thumb, a VIF value that exceeds 5 or 10 indicates a problematic amount of collinearity. 

plot(cooks.distance(model3)) #let's detect outliers or leverage points. There are many, it's impossible to eliminate all!
# Therefore, I think machine learning models will performance better results. 

# Prediction for Linear Model
predicted.LM <- predict(model3, testSet)
mse.LM <- mean((predicted.LM - (testSet$user_rating)^2.5)^2)
mse.LM #mse is 291.35 for LM


#### Neural Network ####
n2 <- colnames(trainSet)
f2 <- as.formula(paste("user_rating ~",paste(n[!n %in% "user_rating"],collapse = "+"))) 
f <- as.formula(paste("user_rating ~",paste(n[!n %in% "user_rating_ver"],collapse = "+")))

set.seed(1234)
NN_fit.1 <- neuralnet(f2, data = trainSet)
plot(NN_fit.1, rep = 'best')

#*# Every cell has their weight, and NN model which we are using is default that means it has only one hidden layer.
predicted.NN.fit1 <- predict(NN_fit.1,testSet)
mse.NN.fit1 <- mean((predicted.NN.fit1 - testSet$user_rating)^2)
mse.NN.fit1 # mse is 2.336

#### Support Vector Machines ####
svm.fit <- svm(user_rating ~. , kernel = "linear", data = trainSet)
predicted.SVM <- predict(svm.fit, testSet)
mse.SVM = mean((predicted.SVM - testSet$user_rating)^2)
mse.SVM ### 1.283

#### Random Forest ####
RF.fit1 <- randomForest(user_rating ~ . , data = trainSet)
predicted.RF.fit1 <- predict(RF.fit1, testSet)
mse.RF = mean((predicted.RF.fit1 - testSet$user_rating)^2)
mse.RF # mse is 0.19

importance(RF.fit1) # relative importance of predictors (highest <-> most important)
varImpPlot(RF.fit1) # plot results



