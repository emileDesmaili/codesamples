
library(readxl)

sample_data <- read_excel("Data Assignement 2/25_Portfolios_5x5_CSV/25_Portfolios_5x5.xlsx")

T=NROW(sample_data);N=NCOL(sample_data)-1
mkt=as.matrix(sample_data[,N+1]) #vector of Rm; b=0

for (i in (1:N)) { #run 25 time series regression RPortfolio=a+B*Rm and store the betas in b
  my_ols=lm(as.matrix(sample_data[,i]) ~ mkt)
b[i]=summary(my_ols)$coefficients[2,1]

}
b_vec=as.matrix(b)

n=matrix(0,T,1)
gama=matrix(0,T,1)
alpha_mat=matrix(0,T,N)
sample_data_2=t(as.matrix(sample_data[,-N-1])) #transpose the vector of portoflio returns and remove Rm


for (j in (1:T)) { #Run 669 cross-section regressions RPortfolio=gamma+lambda*beta+alpha and store lambda gamma and alpha
  my_ols2=lm(sample_data_2[,j] ~ b_vec)
n[j]=summary(my_ols2)$coefficients[2,1]
gama[j]=summary(my_ols2)$coefficients[1,1]
alpha_mat[j,]=summary(my_ols2)$residuals
}
lambda_vec=as.matrix(n)
gamma_vec=as.matrix(gama)

lambda_bar=mean(lambda_vec)
gamma_bar=mean(gamma_vec)

stderr_lambda=sd(lambda_vec)/sqrt(T)
stderr_gamma=sd(gamma_vec)/sqrt(T)
my_tstat_gamma=gamma_bar/stderr_gamma
my_tstat_lambda=lambda_bar/stderr_lambda

plot.ts(gamma_vec)
plot.ts(lambda_vec)

########  QUESTION 3 ################
r_bar=matrix(0,N,1)

for (c in (1:N)){
  r_bar[c]=mean(as.matrix(sample_data_2[c,]))
}

newalpha_bar=matrix(0,N,1)
my_ols3=lm(r_bar ~ b_vec)
new_lambda=summary(my_ols3)$coefficients[2,1]
new_gamma=summary(my_ols3)$coefficients[1,1]
new_se_lambda=summary(my_ols3)$coefficients[2,2]
new_se_gamma=summary(my_ols3)$coefficients[1,2]
newalpha_bar=summary(my_ols3)$residuals

my_quad=lm(r_bar ~ poly(b_vec,2))
plot(b_vec,r_bar,col='blue')
lines(lowess(b_vec,fitted(my_quad)))
abline(my_ols3,col='red')
summary(my_quad)


########### Manual ############# do the  chi-square test manually

sigma=cov(alpha_mat)/(T)
alpha_bar=matrix(0,N,1)
for (o in (1:N)){
  alpha_bar[o]=mean(alpha_mat[,o])
}

chi2=t(alpha_bar)%*%solve(sigma,tol=-1e-5)%*%(alpha_bar)
my_pvalue=2*pchisq(chi2, df=N-1,lower.tail=FALSE)






