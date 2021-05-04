import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import invgamma
from numpy import random
import random as rd
from pandas import DataFrame
import pandas as pd

BATCH_SIZE=20
SAMPLE_SIZE=1000
ITER=5000
sigma1_2,sigma2_2,sigmax_2=10,1,2
theta_0=[0,1]
def norm(mu,sigma_2,x):
  return 1/np.sqrt(2*np.pi*sigma_2)*np.exp(-(x-mu)*(x-mu)/2/sigma_2)
  
def normall(theta,X,sigmax_2):
  px=norm(0,sigma1_2,theta[0])*norm(0,sigma2_2,theta[1])
  for i in range(len(X)):
    px*=(1/2*(norm(theta[0],sigmax_2,X[i])+norm(theta[0]+theta[1],sigmax_2,X[i])))
  return px
  
def grad_predict(xi,theta,sigmax_2):
  pxi=1/2*(norm(theta[0],sigmax_2,xi)+norm(theta[0]+theta[1],sigmax_2,xi))
  dp1=1/2*(xi-theta[0])/sigmax_2*norm(theta[0],sigmax_2,xi)
  dp2=1/2*(xi-theta[0]-theta[1])/sigmax_2*norm(theta[0]+theta[1],sigmax_2,xi)
  #pxi=1/2/np.sqrt(2*np.pi*sigmax_2)*(np.exp(-(xi-theta[0])*(xi-theta[0])/2/sigmax_2)+
  #np.exp(-(xi-theta[0]-theta[1])*(xi-theta[0]-theta[1])/2/sigmax_2))
  #dp1=1/2/np.sqrt(2*np.pi*sigmax_2*sigmax_2*sigmax_2)*(xi-theta[0])*np.exp(-(xi-theta[0])*(xi-theta[0])/2/sigmax_2)
  #dp2=1/2/np.sqrt(2*np.pi*sigmax_2*sigmax_2*sigmax_2)*(xi-theta[0]-theta[1])*np.exp(-(xi-theta[0]-theta[1])*(xi-theta[0]-theta[1])/2/sigmax_2)
  dp=[(dp1+dp2)/pxi,dp2/pxi]
  return np.array(dp)
  
def grad_prior(theta,sigma1_2,sigma2_2):
  return np.array([-theta[0]/sigma1_2,-theta[1]/sigma2_2])
  
def syn_data(N,theta,sigmax_2):
  n1=int(N/2)
  x1=random.normal(theta[0],np.sqrt(sigmax_2),size=n1)
  x2=random.normal(theta[0]+theta[1],np.sqrt(sigmax_2),size=N-n1)
  mix=np.concatenate((x1,x2))
  return mix

def MCMC(T,X,sigma1_2,sigma2_2,sigmax_2):
  theta=random.uniform(-1,1,size=2)
  param_rec=np.zeros((T,2))
  px=norm(0,sigma1_2,theta[0])*norm(0,sigma2_2,theta[1])*normall(theta,X,sigmax_2)
  for t in range(T):
    th1=random.normal(theta[0],10)
    px_=norm(0,sigma1_2,th1)*norm(0,sigma2_2,theta[1])*normall([th1,theta[1]],X,sigmax_2)
    if(px*random.uniform(0,1)<min(px,px_)):
      theta[0]=th1
      px=px_
    th2=random.normal(theta[1],10)
    px_=norm(0,sigma1_2,theta[0])*norm(0,sigma2_2,th2)*normall([theta[0],th2],X,sigmax_2)
    if(px*random.uniform(0,1)<min(px,px_)):
      theta[1]=th2
      px=px_
    param_rec[t,:]=theta
  param_rec = DataFrame(param_rec)
  param_rec.columns=['theta1','theta2']
  return param_rec

#def Gibbs(T,X,sigma1_2,sigma2_2,sigmax_2):
  
  
def SGLD(Bsize,T,X,sigma1_2,sigma2_2,sigmax_2):
  theta=random.uniform(-1,1,size=2)
  param_rec=np.zeros((T,2))
  for t in range(T):
    ind=rd.sample(range(len(X)),Bsize)
    g=np.array([0.0,0.0])
    for i in ind:
      g=g+grad_predict(X[i],theta,sigmax_2)
    g=g/Bsize*len(X)
    g+=grad_prior(theta,sigma1_2,sigma2_2)
    et=0.01/np.sqrt(1+t)
    mu=et/2*g
    d1=random.normal(mu[0],et)
    d2=random.normal(mu[1],et)
    theta+=np.array([d1,d2])
    param_rec[t,:]=theta
  param_rec = DataFrame(param_rec)
  param_rec.columns=['theta1','theta2']
  return param_rec
  
X=syn_data(SAMPLE_SIZE,theta_0,sigmax_2)
plt.figure(2)
plt.hist(X)
param_rec=SGLD(BATCH_SIZE,ITER,X,sigma1_2,sigma2_2,sigmax_2)
#param_rec2=MCMC(ITER*5,X,sigma1_2,sigma2_2,sigmax_2)
theta1=param_rec['theta1'].values
theta2=param_rec['theta2'].values
#th1=param_rec2['theta1'].values
#th2=param_rec2['theta2'].values
it=[*range(1, ITER+1)]
#it2=[*range(1, ITER*5+1)]
plt.figure(0)
plt.plot(it,theta1,'-.',color='b',linewidth=1, label='theta1')
plt.plot(it,theta2,'-.',color='r',linewidth=1, label='theta2')
plt.grid(True)
plt.legend(loc='best')
plt.figure(1)
plt.scatter(theta1,theta2)
plt.grid(True)
plt.xlabel('theta1')
plt.ylabel('theta2')
"""
plt.figure(3)
plt.plot(it2,th1,'-.',color='b',linewidth=1, label='theta1')
plt.plot(it2,th2,'-.',color='r',linewidth=1, label='theta2')
plt.grid(True)
plt.legend(loc='best')
"""
plt.show()