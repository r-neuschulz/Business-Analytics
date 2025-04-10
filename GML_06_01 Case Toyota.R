# Author Frank Lehrbass, May 2022
# Ein R Skript für das FOM Zertifikat 
# nur zum Zwecke der Lehre
# KEINE GARANTIE bzgl Korrektheit o.a. 
# This R coding is WITHOUT ANY WARRANTY

#Zuvor verwendete Variablen leeren

# ------------------------------------------------------------------------------
# Clean previous variables
# ------------------------------------------------------------------------------

rm(list=ls(all=TRUE))

# ------------------------------------------------------------------------------
# Load packages
# ------------------------------------------------------------------------------

library(lmtest) # Regressionsdiagnostik
library(sandwich) #HAC SE
library(car) #VIF und Co
library(sjPlot) #only needed for nice output/fomatting of tables see viewer
#if THIS PACKAGE is not available it is no big issue, content stays the same, just layout less nice

Daten = read.table("GML_06_01 Case Toyota.csv", header=TRUE, sep=",",dec=".")

summary(Daten)
dim(Daten)

Daten = Daten[,-c(1:2,5:6)] #remove irrelevant info, all cars of same type etc
Daten = Daten[,-c(11)] #remove static data like cylinders, all = 4

dim(Daten)
head(Daten)


insample = data.frame(Daten[1:1000,])

#in sample regression
reg = lm(Price~., data = insample)
summary(reg, digits = 4)
tab_model(reg) #look in Viewer tab to the far right side

# ------------------------------------------------------------------------------
# Diagnostics of one specific regression
# ------------------------------------------------------------------------------

##Fehlspezifikation------------------------------------------
resettest(reg)#H0 No spec error

##Hetero---------------------------------------------------
plot(fitted(reg),residuals(reg)) #, type='l')
#Breusch-Pagan-Test auf Heteroskedastizit?t
bptest(reg)#H0: Homoscedasticity! but assumes normal resis

#Normality-------------------------------------------------
kernel_den = density(residuals(reg))
plot(kernel_den, main="Smoothed Residual Distribution", xlab = " ")
x = seq(min(kernel_den$x),max(kernel_den$x),length = 1000)
#compare to normal
std_den = dnorm(x,mean = mean(residuals(reg)),sd = sd(residuals(reg)))
lines(x,std_den,type="l",col="red")

#as test
shapiro.test(residuals(reg))#H0: Is normal
#NOT nORMALLLLLLLLLLLL

##conclusion-----------------------------------------------
#Hence Newey-West again!!!
summary(reg)
#75% is idiosnycratic risk!
coeftest(reg,vcov=NeweyWest(reg))#now use heteroskedasticity and autocorrelation
#consistent (HAC) covariance matrix estimators. Note the decreased t value!

vif(reg)[,1]#could reduce X dim ...

#end