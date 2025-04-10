# Author Frank Lehrbass, May 2022
# Ein R Skript für das FOM Zertifikat 
# nur zum Zwecke der Lehre
# KEINE GARANTIE bzgl Korrektheit o.a. 
# This R coding is WITHOUT ANY WARRANTY

#Zuvor verwendete Variablen leeren
rm(list=ls(all=TRUE))

knitr::opts_chunk$set(message = FALSE)
knitr::opts_chunk$set(warning = FALSE)

# Bibliotheken importieren

library(lmtest) # Regressionsdiagnostik
library(sandwich) #HAC SE
library(car) #VIF und Co
library(leaps) #for regsubsets
library(gamlr) #for LASSO etc
library(tseries) #adf.test
library(corrplot) #for corr visualization
library(GGally)#for plotting ggpairs

#for CART
library(rpart)
library(rpart.plot)

#for MLP
library(neuralnet)
library(NeuralNetTools)

# ------------------------------------------------------------------------------
# DGP
# ------------------------------------------------------------------------------
SEED = 2021

n=50 #50
n_boot = 2000 #2000 fits to big data considerations in slides
n_epochs = 100 #100 is sufficient and 1000 serves as upper limit for this toy example
set.seed(SEED)
x1 = runif(n,-1,1)
z = runif(n,-0.1,0.1)
eps = rnorm(n,0,0.0005)
y=x1^2+eps
x2=y+z
x3=rnorm(n,0,0.0000005)

df1 = data.frame(cbind(y,x1,x2,x3))

# ------------------------------------------------------------------------------
# Visual analysis
# ------------------------------------------------------------------------------

x11()#open new window
ggpairs(df1)#first column shows: y = x1² with smallest noise!
#and y linear fct of x2 but with more noise
M=cor(df1)
x11()
corrplot(M,method="circle")

# ------------------------------------------------------------------------------
# Linear Regression
# ------------------------------------------------------------------------------

lm1 = lm(y~., data = df1)

# ------------------------------------------------------------------------------
# Diagnostics 
# ------------------------------------------------------------------------------

reg = lm1 #replace here if u want to diag another regression

## Test auf Stationaritaet: --------------------------------
adf.test(y)
adf.test(x1)
#rest is stationary by construction

##Fehlspezifikation-----------------------------------------
resettest(reg)#H0 No spec error

##Hetero----------------------------------------------------
plot(residuals(reg), type='l')
#Breusch-Pagan-Test auf Heteroskedastizit?t
bptest(reg)#H0: Homoscedasticity!

##Multicoll-------------------------------------------------
vif(reg)

##conclusion-----------------------------------------------
#Newey-West-Korrektur nicht noetig 
summary(lm1)

#INSIGHT---------------------------------------------------------------------------
#choosing x1 would be the right choice if we are going to apply nlr
#However, if the learner is going to be a linear regressor, then we should prefer x2
#important graphical inspection!!!

# ------------------------------------------------------------------------------
# Linear Regression and BOOTSTRAP technique intro
# ------------------------------------------------------------------------------
# Bootstrap regression coefficients
set.seed(SEED)
betas_boot = matrix(0,n_boot,4)
for (b in 1:n_boot){
  boot.data = df1[sample(nrow(df1), n, replace = TRUE),]
  betas_boot[b,] = coef(lm(y~., data = boot.data))
}
#ESTIMATES and their SE
cbind(matrix((apply(betas_boot, MARGIN = 2, mean)),4,1),matrix(apply(betas_boot, MARGIN = 2, sd),4,1))
summary(lm1)

#the residual standard error and the coefs are well matched but the SE per x are way off

# ------------------------------------------------------------------------------
# Linear Regression and Big Data
# ------------------------------------------------------------------------------

#increase sample size, x3 gets significant
set.seed(SEED)
n * n_boot
big_n = n * n_boot
big_x1 = runif(big_n,-1,1)
big_z = runif(big_n,-0.1,0.1)
big_eps = rnorm(big_n,0,0.0005)
big_y=big_x1^2+big_eps
big_x2=big_y+big_z
big_x3=rnorm(big_n,0,0.0000005)

big_df1 = data.frame(cbind(big_y,big_x1,big_x2,big_x3))

big_lm1 = lm(big_y~big_x1+big_x2+big_x3)
summary(big_lm1)#WATCH! #t-value based choice is w alpha = 5%: USE x2+x3!!!

# ------------------------------------------------------------------------------
# backward selection, start with all regressors
# ------------------------------------------------------------------------------

backward_lm1 = regsubsets(y~x1+x2+x3, data = df1, method=c("backward"))
summary(backward_lm1)
#it can be seen that the best 1-variables model contains only x2, as in INSIGHT
#it can be seen that the best 2-variables model contains only x2 and x3, again false conclusion
#HENCE: Use domain knowledge and many charts
#and look forward to MLP where functional form is specified during training

# ------------------------------------------------------------------------------
# forward selection, start with all regressors
# ------------------------------------------------------------------------------

forward_lm1 = regsubsets(y~x1+x2+x3, data = df1, method=c("forward"))
summary(forward_lm1)

#alternatively as in Taddy Slides
null = glm(y ~ 1, data=df1)
full = glm(y~x1+x2+x3, data = df1)
fwd = step(null, scope=formula(full), dir="forward")
summary(fwd)#note the presence of x3

# ------------------------------------------------------------------------------
# LASSO selection, start with all regressors
# ------------------------------------------------------------------------------

summary(df1)
#hence we have to scale!, we do it inside function call (standardize = T)
lasso_lm1 = gamlr(cbind(x1,x2,x3), y, standardize = T) # lasso
x11()
plot(lasso_lm1)

lasso_lm1$beta[,10] #for high lambda
lasso_lm1$beta[,50]
lasso_lm1$beta[,100] #for lowest level of lambda

coef(lm1) #note that this is not the same in detail 
#(due to standardization and solving a different min-problem)
#but the main message is similar!

# ------------------------------------------------------------------------------
# LASSO selection, start with all regressors and use cross vali
# ------------------------------------------------------------------------------

lasso_lm2 = cv.gamlr(cbind(x1,x2,x3), y, standardize = T, nfold=5) 
# lasso, nfold = 50 i.e.each esti uses 40 elemnts (=sample) and 
#looks at error of left out set of ten elements (OOS)
x11()
plot(lasso_lm2) #see that the lowest penalty leads t lowest error out of sample
coef(lasso_lm2, select = "min")

# ------------------------------------------------------------------------------
# wrap up of econometrics
# ------------------------------------------------------------------------------

#starting point
summary(lm1)
# LASSO et al: drop x3, Visual: use x1 squared
x1sq = x1^2
lm3 = lm(y~x1sq+x2, data = df1)
summary(lm3)

# ------------------------------------------------------------------------------
# apply your first tree
# ------------------------------------------------------------------------------

#piecewise linear fct, see Schlittgen, p. 168 ff
cart1 = rpart(y~x1+x2+x3, data = df1, maxdepth = 1)
summary(cart1)
#https://gormanalysis.com/decision-trees-in-r-using-rpart/

#use fcts
round(cor(lm1$fitted.values,y)^2,4)
round(cor(rpart.predict(cart1,as.data.frame(cbind(x1,x2,x3)),rules=F),y)^2,4)#decrease

x11()
#compare this to CART tree result in Larose page 324, Fig. 11.4
rpart.plot(cart1, main = "CART 1")

x11()
#the full monty in one chart
plot(x2,y, main = "Linear Regression & CART 1")
abline(lm(y~x2))
abline(h=0.1339119)
abline(h=0.6749837)
abline(v=0.4278535)

#deep dive-------------------------------------------------------

#MSE by no splitting, i.e. forecast mean(y)
sum((y-mean(y))^2)/length(y)#compare this to MSE in cart summary

#now use split, use chart & summary data and build function
cart1Forecast = function(x){
  if (x < 0.4278535) (0.1339119)
  else (0.6749837)
}

#cross check first forecast
yx2_pairs = Matrix(c(y,x2),50,2)

m2 = 0
j=0
for (i in (1:50)){
  if(yx2_pairs[i,2]<0.4278535){
    m2 = m2 + yx2_pairs[i,1]
    j=j+1
    
      } 
}
m2=m2/j
m2 #hence, average y is forecasted

#test
cart1Forecast(0.8)
y_hat_cart1 = apply(as.matrix(x2), MARGIN = 1, FUN = cart1Forecast)
sum((y-y_hat_cart1)^2)/length(y)#compare this to MSE in cart summary
23/50*0.01691391 + 27/50*0.03071012
#R²
round(cor(lm1$fitted.values,y)^2,4)
round(cor(y_hat_cart1,y)^2,4)#decrease

# ------------------------------------------------------------------------------
# apply your 2nd tree
# ------------------------------------------------------------------------------

cart2 = rpart(y~x1+x2+x3, data = df1, maxdepth = 2)
round(cor(lm1$fitted.values,y)^2,4)
round(cor(rpart.predict(cart2,as.data.frame(cbind(x1,x2,x3)),rules=F),y)^2,4)#improve

#but more honest to compare
lm_single = lm(y~x2)
round(cor(lm_single$fitted.values,y)^2,4)
round(cor(rpart.predict(cart2,as.data.frame(cbind(x1,x2,x3)),rules=F),y)^2,4)#improve

x11()
#compare this to CART tree result in Larose page 324, Fig. 11.4
rpart.plot(cart2, main = "CART 2")

# ------------------------------------------------------------------------------
# apply your 3rd tree
# ------------------------------------------------------------------------------

cart3 = rpart(y~x1+x2+x3, data = df1, maxdepth = 3)
round(cor(lm1$fitted.values,y)^2,4)
round(cor(rpart.predict(cart3,as.data.frame(cbind(x1,x2,x3)),rules=F),y)^2,4)#improve

x11()
#compare this to CART tree result in Larose page 324, Fig. 11.4
rpart.plot(cart3, main = "CART 3")

# ------------------------------------------------------------------------------
# apply your first MLP
# ------------------------------------------------------------------------------

set.seed(SEED)
#model as formula
modelform = y~x1+x2+x3

net0 = neuralnet(modelform, data = df1, hidden = 1, err.fct = "sse", lifesign = "full", act.fct = "logistic", 
                 linear.output = F)
#use mostly default values for starters and only one hidden neuron, hence net0
#can set algorithm = "rprop+", threshold = 0.01) to make default explicit
net0_fc = predict(net0, as.matrix(cbind(x1,x2,x3)), rep = 1, all.units = FALSE)
#Return output for all units instead of final output only

#Plot the neural network, rough and dirty - old school
x11()
plot(net0)

#nicer is
x11()
plotnet(net0, y_names = "y", circle_cex = 3, node_labs = T, var_labs = T)#default settings
#NOTE positive black, grey = ng., this confuses a bit, see that x3 gets kind of canceled by pos/neg weights

x11()#function y(xi) shown with all other xj fixed at their resp 10/90 percent quantile
lekprofile(net0, group_show = F, group_vals = c(0.4,0.6)) 
#NOTE: Net0 has found out that x1 has a quadratic impact! And that x3 is of no interest

x11()#same at median values
lekprofile(net0, group_show = F, group_vals = c(0.5))
#padding on the x-axis to make it readable in full

#compare R² of lm1 and net0
round(cor(lm1$fitted.values,y)^2,4)
round(cor(net0_fc,y)^2,4)#clear cause net has got more parameters!

# ------------------------------------------------------------------------------
# Deep dive into your first MLP with ony one hidden
# ------------------------------------------------------------------------------
#understand output from red letter reporting, see that SSE/2 is reported!
net0$result.matrix[1,1]
round(sum((y-net0_fc)^2)/2,5)#note that target fct is 1/2 SSE 
#cause by first derivation get 2*SSE which cancels then against 1/2

#typical activation fct
logistic = function(x){
  1/(1+exp(-x))
}
x11()
plot(seq(-10,10,1), logistic(seq(-10,10,1)), type = 'l', main = "Logistic / Sigmoid Function")

#cross check first forecast
net0_fc[1,1]
#we need the follwoing weights
net0$result.matrix[4,1]#Intercept.to.1layhid1
net0$result.matrix[5,1]#x1.to.1layhid1
net0$result.matrix[6,1]#x2.to.1layhid1
net0$result.matrix[7,1]#x3.to.1layhid1

#from input into neuron in first hidden layer
single_hidden_neuron = 
  logistic(x1[1]*net0$result.matrix[5,1]+x2[1]*net0$result.matrix[6,1]+x3[1]*net0$result.matrix[7,1]
           +net0$result.matrix[4,1])#bX+a
#from first hidden layer to output neuron
logistic(net0$result.matrix[9,1]*single_hidden_neuron+net0$result.matrix[8,1])
#compare
net0_fc[1,1]

# ------------------------------------------------------------------------------
# apply your 2nd MLP with 2 hidden neurons and increase control
# ------------------------------------------------------------------------------
#NOW take back control and also change output neuron from linear to signoid, i.e. MOVE to net1
set.seed(SEED)

#give the net some fctal flexibility to discover that x1 matters squared!!!
#pyramid rule 3 in + 1 out * 2/3 = 8/3 = Ockham! 2 hiddens due to Ockhams razor
net1 = neuralnet(modelform, data = df1, hidden = 2, stepmax = n_epochs,
                 lifesign = "full", err.fct = "sse", linear.output = F,
                 act.fct = "logistic")
net1_fc = predict(net1, as.matrix(cbind(x1,x2,x3)), rep = 1, all.units = FALSE)

x11()#controlled setting if a big architecture comes
plotnet(net1, cex_val = 0.5, alpha_val = 1, pad_x = 0.6,#padding on the x-axis to make it readable in full
        y_names = "y", circle_cex = 3, node_labs = T, var_labs = T)
#NOTE positive black, grey = neg., this confuses a bit, see that x3 gets kind of canceled by pos/neg weights

#show the clusters
x11()#function y(xi) shown with all other xj fixed at their resp 10/90 percent quantile
lekprofile(net1, group_show = F, group_vals = c(0.4,0.6)) 
#NOTE: Net has found out that x1 has a quadratic impact! And that x3 is of no value
#compare R² of lm1 and net1
round(cor(lm1$fitted.values,y)^2,4)
round(cor(net1_fc,y)^2,4)#slight increase

#plot winning net
x11()
plot(net1)

#apply diagnostics
net1_resi = y-net1_fc

##Hetero----------------------------------------------------
plot(net1_resi, type='l')

x11()
plot(net1_fc,net1_resi)

# ------------------------------------------------------------------------------
# apply richer MLP 
# ------------------------------------------------------------------------------

set.seed(SEED)
net2 = neuralnet(modelform, data = df1, hidden = c(3,2), #pyramidal
                 
                 stepmax = n_epochs,# has been shown above as sufficient 
                 lifesign = "full", 
                 err.fct = "sse", #sse’ and ’ce’ which stand for the sum of squared errors and the cross-entropy can be used
                 
                 linear.output = F, #replace linear output neuron by logistic activation, note the chnage in x2 fct form
                 act.fct = "logistic")#this is the real change to frist try
net2_fc = predict(net2, as.matrix(cbind(x1,x2,x3)), rep = 1, all.units = FALSE)#Return output for all units instead of final output only

x11()#controlled setting if big net comes
plotnet(net2, cex_val = 0.5, alpha_val = 1, pad_x = 0.6,#padding on the x-axis to make it readable in full
        y_names = "y", circle_cex = 3, node_labs = T, var_labs = T)
#NOTE positive black, grey = ng., this confuses a bit, see that x3 gets kind of canceled by pos/neg weights
x11()#controlled setting if big net comes
plotnet(net2, y_names = "y")

#compare R² of lm1 and net2: NOTE bigger not better
round(cor(lm1$fitted.values,y)^2,4)
round(cor(net2_fc,y)^2,4)#less than linear model!
#need math to understand this! one hidden layer is enough!!!

# ------------------------------------------------------------------------------
# MLP with 2 hidden neurons by using insights from lekprofile that x3 does not matter
# ------------------------------------------------------------------------------

set.seed(SEED)
#model as formula
modelform_selection = y~x1+x2

#give the net some fctal flexibility to discover that x1 matters squared!!!
#pyramid rule 3 in + 1 out * 2/3 = 8/3 = Ockham! 2 hiddens due to Ockhams razor
net3 = neuralnet(modelform_selection, data = df1, hidden = 2, stepmax = n_epochs,
                 lifesign = "full", err.fct = "sse", linear.output = F,
                 act.fct = "logistic")
net3_fc = predict(net3, as.matrix(cbind(x1,x2)), rep = 1, all.units = FALSE)

x11()#controlled setting if a big architecture comes
plotnet(net3, cex_val = 0.5, alpha_val = 1, pad_x = 0.6,#padding on the x-axis to make it readable in full
        y_names = "y", circle_cex = 3, node_labs = T, var_labs = T)
#NOTE positive black, grey = neg., this confuses a bit, see that x3 gets kind of canceled by pos/neg weights

#compare R² of lm1 and net3
round(cor(lm1$fitted.values,y)^2,4)
round(cor(net3_fc,y)^2,4)#slight decrease

#plot winning net
x11()
plot(net3)

#nicer is
x11()
plotnet(net3, y_names = "y")#default settings
#NOTE positive black, grey = ng., this confuses a bit, see that x3 gets kin

x11()
olden(net3)

#apply diagnostics
net3_resi = y-net3_fc

##Hetero----------------------------------------------------
plot(net3_resi, type='l')

#end