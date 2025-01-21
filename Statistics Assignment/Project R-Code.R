library(ggplot2)
library(e1071) 
library(readxl)
library(fitdistrplus)

# Load data
data10 <- read_excel("data10.xlsx")
attach(data10)


##################################################################################################
# Descriptive Analysis
##################################################################################################
# age distribution
hist(age, freq=F, main="Histogram of Age distribution")

# weight/height

hist(weight, breaks = 15)
hist(height, breaks = 15)

scatter_plot <- ggplot(data10, aes(x = height, y = weight, color = sex)) +
  geom_point() +
  labs(x = "Height", y = "Weight", color = "Sex") #+
plot(scatter_plot)

# add line for avg BMI to plot
weight_for_avg_bmi <- mean(bmi)*(height/100)^2
height_for_avg_bmi <- sqrt(weight/(mean(bmi)))*100

filterd_wei <- weight_for_avg_bmi[weight_for_avg_bmi >= 41 & weight_for_avg_bmi <= 104]
filterd_hei <- height_for_avg_bmi[height_for_avg_bmi >= 150 & height_for_avg_bmi <= 200 ]

hei <- seq(min(filterd_hei), max(filterd_hei), length.out = 1000)
wei <- seq(min(filterd_wei), max(filterd_wei), length.out = 1000)
line_avg_bmi <- data.frame(x = hei, y = wei)

combined_plot <- scatter_plot +
  geom_line(data = line_avg_bmi, aes(x = x, y = y), color = "blue")
print(combined_plot)


#----------------------------------------------------------------------------------------
# BMI
x_range <- c(15,35)
hist(bmi, breaks = 15, freq = T, xlim = x_range, main = "Histogram BMI", xlab = "BMI")

boxplot(bmi, main = 'Boxplot BMI')
summary(bmi)

# Variance, sd
var(bmi)
sd(bmi)

# Coefficient of variation
sd(bmi)/mean(bmi)

# skewness
skewness(bmi) # left skew

# quartiles
q1 = quantile(bmi,probs = 0.25)
q2 = median(bmi)
q3 = quantile(bmi,probs = 0.75)

func_ecdf <- ecdf(bmi)
func_ecdf(25)
plot(ecdf(bmi))


#----------------------------------------------------------------------------------------
# BMI by gender
table(sex)
table(sex)/168

summary(bmi[sex == 'Female'])
summary(bmi[sex == 'Male'])

hist(bmi[sex == 'Female'], breaks = 20, freq = FALSE, xlim = x_range, main = "Female", xlab = "BMI")
hist(bmi[sex == 'Male'], breaks = 20, freq = FALSE, xlim = x_range, main = "Male", xlab = "BMI")

ggplot(data10, aes(x = sex, y = bmi, fill = sex)) +
  geom_boxplot() +
  labs(title = "") +
  ylab("BMI") +
  xlab("Sex")

#----------------------------------------------------------------------------------------
# Physiscal activity
table(physical_activity)
sample_size <- length(physical_activity)
rf_activity <- table(physical_activity)/sample_size*100
rf_activity
pie(rf_activity,labels=paste(names(rf_activity)
                             ,(paste0(round(as.vector(rf_activity),2),'%'))),
    edges =200, radius=1, col=c('green','red','yellow','blue','orange'),
    main="Relative frequency of activity in population")

#----------------------------------------------------------------------------------------
# Sugar preference
table(sugar_preference)

sugar_preference_table <- table(sugar_preference)/sample_size*100
sugar_preference_table
pie(sugar_preference_table,labels=paste(names(sugar_preference_table)
                             ,(paste0(round(as.vector(sugar_preference_table),2),'%'))),
    edges =200, radius=1, col=c('red','green','yellow','orange'))


#----------------------------------------------------------------------------------------
# Physical Activty + Sugar frequency
table(activity_frequency)
table(sugar_frequency)

rf_sugar_freq <- table(sugar_frequency)/sample_size*100
rf_sugar_freq


rf_activity <- table(physical_activity)/sample_size*100
rf_activity


table_SP_SF = table(sugar_frequency, sugar_preference)
table_SP_SF
# Stacked bar plot

cross_table <- table(activity_frequency, sugar_frequency)
cross_table
c1 <- as.vector(cross_table[,1])
c2 <- as.vector(cross_table[,2])
c3 <- as.vector(cross_table[,3])

Sugar_Frequency <- c(rep("0-1 times/week", 3) , rep("2-3 times/week" , 3) , rep("4+ times/week", 3) )
categories <- rep(c("0-1 times/week" , "2-3 times/week" , "4+ times/week"),3)
value <- c(c1,c2,c3)
data <- data.frame(categories,Sugar_Frequency,value)

ggplot(data, aes(fill=Sugar_Frequency, y=value, x=categories)) + 
  geom_bar(position="fill", stat="identity")+
  xlab("Physical activity frequency") +
  ylab("Sugar frequencies")


#freq of consumption / freq of training
table(sugar_frequency,activity_frequency)
table(sugar_frequency,activity_frequency)/sample_size*100
ggplot(data10, aes(fill=sugar_frequency, x=activity_frequency))+geom_bar()+labs(title = "Activity/sugar consumption frequency  per week (abs. values)", x = "Activity frequency per week", y = "Sample size")+scale_fill_manual(values = c('green','yellow','red'))+theme_minimal()


##################################################################################################
# Model fitting
##################################################################################################


#-----------------------------------------------------------------------------------
# Method of Moments
hist(bmi, freq = F, breaks = 20, main = 'Method of Moments')
grid=seq(0,40,0.01)

# Normal
mme_norm <- fitdist(bmi, 'norm', method ='mme')
mme_norm$estimate
lines(grid, dnorm(grid, mme_norm$estimate[1], mme_norm$estimate[2]), col='orange')

# Lognormal
mme_lnorm <- fitdist(bmi, "lnorm", method="mme")
mme_lnorm$estimate
lines(grid, dlnorm(grid, mme_lnorm$estimate[1], mme_lnorm$estimate[2]), col='blue')

# Gamma
mme_gamma <- fitdist(bmi,"gamma",method="mme")
mme_gamma
lines(grid, dgamma(grid, mme_gamma$estimate[1], mme_gamma$estimate[2]),col="red")

legend("topleft",c("Normal","Lognormal","Gamma"),col=c("orange", "blue","red"),lty=c(1,1))

# Beta 
scaled_bmi <- (bmi-min(bmi))/(max(bmi)-min(bmi))
scaled_bmi
grid1<- seq(0, 1 , 0.01)

mme_beta <- fitdist(scaled_bmi, 'beta', method = 'mme')
mme_beta

hist(scaled_bmi, freq = F)
lines(grid1, dbeta(grid1, mme_beta$estimate[1], mme_beta$estimate[2]), col="pink")


# Compare ecdf
plot(ecdf(bmi))
lines(grid,pnorm(grid, mme_norm$estimate[1], mme_norm$estimate[2]), col = 'orange')
lines(grid,plnorm(grid, mme_lnorm$estimate[1], mme_lnorm$estimate[2]), col = 'blue')
lines(grid,pgamma(grid, mme_gamma$estimate[1], mme_gamma$estimate[2]),col="red")
legend("topleft",c("Normal","Lognormal","Gamma"),col=c("orange", "blue","red"),lty=c(1,1))

# -----------------------------------------------------------------------------------
# Maximum Likelihood Estimation
hist(bmi, freq = F, breaks = 20, main = 'Maximum Likelihood Estimation')
grid=seq(0,40,0.01)

# normal distribution
mle_norm <- fitdist(bmi, 'norm', method = 'mle')
mle_norm$estimate
lines(grid, dnorm(grid, mle_norm$estimate[1], mle_norm$estimate[2]), col = 'orange')

# Lognormal distribution:
mle_lnorm <- fitdist(bmi,"lnorm", method="mle")
mle_lnorm$estimate
lines(grid, dlnorm(grid,mle_lnorm$estimate[1],mle_lnorm$estimate[2]), col='blue')


# Gamma:
mle_gamma <- fitdist(bmi, "gamma", method="mle")
mle_gamma$estimate
lines(grid, dgamma(grid,mle_gamma$estimate[1],mle_gamma$estimate[2]),col="red")

legend("topleft",c("Normal","Lognormal","Gamma"),col=c("orange", "blue","red"),lty=c(1,1))

# Beta 
scaled_bmi2 <- (bmi - min(bmi) + 0.001) / (max(bmi) - min(bmi) + 0.002)  # so that <> 0 or 1
scaled_bmi2
mle_beta <- fitdist(scaled_bmi2, 'beta', method = 'mle')
mle_beta

hist(scaled_bmi, freq = F)
lines(grid1, dbeta(grid1, mle_beta$estimate[1], mle_beta$estimate[2]), col="pink")


# Compare ecdf
plot(ecdf(bmi))
lines(grid,pnorm(grid, mle_norm$estimate[1], mle_norm$estimate[2]), col = 'orange')
lines(grid,plnorm(grid, mle_lnorm$estimate[1], mle_lnorm$estimate[2]), col = 'blue')
lines(grid,pgamma(grid, mle_gamma$estimate[1], mle_gamma$estimate[2]),col="red")
legend("topleft",c("Normal","Lognormal","Gamma"),col=c("orange", "blue","red"),lty=c(1,1))


# -----------------------------------------------------------------------------------
# Model comparison using AIC
# MME
mme_norm$aic        # normal distribution has lowest AIC -> best model
mme_lnorm$aic
mme_gamma$aic

mme_beta$aic

# MLE
mle_norm$aic        # normal distribution has lowest AIC -> best model
mle_lnorm$aic
mle_gamma$aic

mle_beta$aic

# Statistical Inference
library(readxl)
library(nortest)
library(fitdistrplus)

# Function for checking normality
check_normal <- function(data) {
  normal <- FALSE
  
  # Fit distributions using MLE
  aic_results <- c()
  distributions <- c("norm", "gamma", "lnorm")
  for (distribution in distributions) {
    fit <- fitdist(data, distribution, method="mle")
    aic_results <- c(fit$aic)
  }
  
  # Perform Lilliefors (K-S) test
  p_val <- lillie.test(data)$p.value
  
  if (p_val > 0.05 && aic_results[1] == min(aic_results)) {
    normal <- TRUE
  }
  
  return(normal)
}

# Function for plotting data
plot_data <- function(data) {
  fit <- fitdist(data, "norm", method = "mle")
  hist(data, freq = FALSE)
  grid <- seq(15, 30, 0.01)
  lines(grid, dnorm(grid, fit$estimate[1], fit$estimate[2]), col = "red")
}

# Hypothesis 1: Men have higher average BMI values than Women
names(data10)
men <- bmi[sex == 'Male']
women <- bmi[sex == 'Female']

# Check first if both variables are normal
isnormal_men <- check_normal(men)
isnormal_women <- check_normal(women)
if (isnormal_men && isnormal_women) {
  plot_data(men)
  plot_data(women)
}

# Test with normal test-statistic
res <- t.test(men, women, alternative = "greater", var.equal = FALSE)
if (res$p.value < 0.05) {
  print(res$p.value)
  print("Men have higher average BMI than women")
} else {
  print("Cannot reject that samples have different average BMI")
}

# Healthy BMI among men and women
n_men <- length(men)
n_women <- length(women)
healthy_men <- length(men[men >= 18.5 & men <= 24.9])
healthy_women <- length(women[women >= 18.5 & women <= 24.9])
prop.test(
  c(healthy_men, healthy_women), 
  c(n_men, n_women), 
  alternative = "less"
)

# Bayesian test
# Prior parameters
alpha <- 0.1
beta <- 0.1

# Posterior parameters
n <- length(bmi[bmi >= 18.5 & bmi <= 24.9])
pbeta(0.5, alpha + healthy_men, beta + n - healthy_men)

# ANOVA test
other <- bmi[physical_activity == 'Other']
walk <- bmi[physical_activity == 'Walk']
none <- bmi[physical_activity == 'None']
gym <- bmi[physical_activity == 'Gym']
running <- bmi[physical_activity == 'Running']
total <- c(other, walk, none, gym, running)
lillie.test(other)$p.value
lillie.test(walk)$p.value
lillie.test(none)$p.value
lillie.test(gym)$p.value
lillie.test(running)$p.value

block <- c(
  rep(1, length(other)), 
  rep(2, length(walk)), 
  rep(3, length(none)), 
  rep(4, length(gym)),
  rep(5, length(running))
)
Block <- factor(block)
boxplot(total ~ Block)
anova.fit <- aov(total ~ Block)
plot(TukeyHSD(anova.fit))

# Chi-squared + Cramer V
chi_res <- chisq.test(sugar_frequency, activity_frequency)
chi_res
k <- length(unique(activity_frequency))
r <- length(unique(sugar_frequency))
cramer <- sqrt(chi_res$statistic / (length(bmi) * (min(k, r) - 1)))
cramer


