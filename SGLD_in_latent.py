import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import invgamma
from numpy import random
import random as rd
from pandas import DataFrame
import pandas as pd
import math

J_0=100
rho_0=0.8
mu_0=0.0
tau2_0=1.0
sigma2_0=1.0

BATCH_SIZE=2
SAMPLE_SIZE=20
ITER=1500

def synthetic_data(SAMPLE_SIZE,J,rho,mu,tau2,sigma2):
  phi=np.zeros(J)
  Y=np.zeros((SAMPLE_SIZE,J))
  phi[0]=random.normal(mu,np.sqrt(sigma2/(1-rho*rho)))
  for i in range(1,J):
    mean_i=mu+rho*(phi[i-1]-mu)
    phi[i]=random.normal(loc=mean_i,scale=np.sqrt(sigma2))
  for i in range(J):
    Y[:,i]=random.normal(phi[i],np.sqrt(tau2),SAMPLE_SIZE)
  return Y


def grad_mu(J,rho,mu,sigma2,phi):
  return 1/sigma2*(-(1-rho)*((1-rho)*J+2*rho)*mu+(1-rho)*(phi[0]+phi[J-1]+(1-rho)*sum(phi[1:J-1])))
  
def grad_sigma2(J,rho,mu,sigma2,phi):
  v0=(1-rho*rho)*(phi[0]-mu)*(phi[0]-mu)
  for i in range(J-1):
    tmp=phi[i+1]-rho*phi[i]-(1-rho)*mu
    v0=v0+tmp*tmp
  g=-J/2/sigma2+1/(2*sigma2*sigma2)*v0
  return g

def grad_tau2(N,M,J,rho,tau2,phi,y):
  #return -(N/2*J+2)/tau2+(1+N/M*sum(sum(y))-N*sum(phi))/(2*tau2*tau2)
  tmp=0
  for i in range(M):
    for j in range(J):
      tmp+=(y[i,j]-phi[j])*(y[i,j]-phi[j])
  g=-(N/2*J+2)/tau2+(1+N/M*tmp)/(2*tau2*tau2)
  return g

def grad_phi(ind,N,M,J,mu,sigma2,rho,tau2,phi,y):
  if(ind==0):
    return -1/tau2*(N*phi[0]-N/M*sum(y[:,0]))-1/sigma2*((phi[0]-mu)-rho*(phi[1]-mu))
  elif(ind==J-1):
    return -1/tau2*(N*phi[J-1]-N/M*sum(y[:,J-1]))-1/sigma2*((phi[J-1]-mu)-rho*(phi[J-2]-mu))
  else:
    return -1/tau2*(N*phi[ind]-N/M*sum(y[:,ind]))-1/sigma2*((1+rho*rho)*phi[ind]-rho*(phi[ind-1]+phi[ind+1])-mu*(1-rho)*(1-rho))

def sample_phi(J,rho,ind,mu,sigma2,tau2,y,phi):
  if ind==0:
    var=1/(1/tau2+1/sigma2)
    mean=np.mean(y[:,0])/tau2+(mu+rho*(phi[1]-mu))/sigma2
    mean=mean*var
    nphi=random.normal(mean,np.sqrt(var))
  elif ind == J-1 :
    var=1/(1/tau2+1/sigma2)
    mean=np.mean(y[:,J-1])/tau2+(mu+rho*(phi[J-2]-mu))/sigma2
    mean=mean*var
    nphi=random.normal(mean,np.sqrt(var))
  else:
    var=1/(1/tau2+(1+rho*rho)/sigma2)
    mean=np.mean(y[:,ind])/tau2+(1-rho)*(1-rho)*mu/sigma2+rho*(phi[ind-1]+phi[ind+1])/sigma2
    mean=mean*var
    nphi=random.normal(mean,np.sqrt(var))
  return nphi

def SGLD(N,M,J,T,Y,rho):
  mu,sigma2,tau2=random.uniform(-2,2),random.uniform(0.5,2),random.uniform(0.5,2)
  phi=random.uniform(-1,1,J)
  param_rec=np.zeros((T,3))
  dphi=np.zeros(J)
  for t in range(T):
    bind=rd.sample(range(N),M)
    y=Y[bind,:]
    dmu=grad_mu(J,rho,mu,sigma2,phi)
    dsigma2=grad_sigma2(J,rho,mu,sigma2,phi)
    dtau2=grad_tau2(N,M,J,rho,tau2,phi,y)
    for i in range(J):
      dphi[i]=grad_phi(i,N,M,J,mu,sigma2,rho,tau2,phi,y)
    et=0.001/math.pow(t+1,0.05)
    mu+=random.normal(et/2*dmu,et)
    sigma2+=random.normal(et/2*dsigma2,et)
    tau2+=random.normal(et/2*dtau2,et)
    phi+=random.normal(et/2*dphi,et)
    #for i in range(J):
    #phi[i]=sample_phi(J,rho,i,mu,sigma2,tau2,y,phi)
    param_rec[t,:]=np.array([mu,sigma2,tau2])
  param_rec = DataFrame(param_rec)
  param_rec.columns=['mu','sigma2','tau2']
  return param_rec
  
def BBLD(N,M,J,T,Y,rho):
  mu,sigma2,tau2=random.uniform(-2,2),random.uniform(0.5,2),random.uniform(0.5,2)
  phi=random.uniform(-1,1,J)
  param_rec=np.zeros((T,3))
  dphi=np.zeros(J)
  pmu,psigma2,ptau2=mu,sigma2,tau2
  for t in range(T):
    bind=rd.sample(range(N),M)
    y=Y[bind,:]
    dmu=grad_mu(J,rho,mu,sigma2,phi)
    dsigma2=grad_sigma2(J,rho,mu,sigma2,phi)
    dtau2=grad_tau2(N,M,J,rho,tau2,phi,y)
    for i in range(J):
      dphi[i]=grad_phi(i,N,M,J,mu,sigma2,rho,tau2,phi,y)
    et=0.001/math.pow(t+1,0.03)
    phi+=random.normal(et/2*dphi,et)
    if t==0:
      mu+=random.normal(et/2*dmu,et)
      sigma2+=random.normal(et/2*dsigma2,et)
      tau2+=random.normal(et/2*dtau2,et)
      pdmu,pdsigma2,pdtau2=dmu,dsigma2,dtau2
    else:
      s=np.array([mu-pmu,sigma2-psigma2,tau2-ptau2])
      y=np.array([dmu-pdmu,dsigma2-pdsigma2,dtau2-pdtau2])
      a=np.matmul(s,s)/np.matmul(s,y)
      pmu,psigma2,ptau2=mu,sigma2,tau2
      pdmu,pdsigma2,pdtau2=dmu,dsigma2,dtau2
      mu+=random.normal(et/2*dmu,et)
      sigma2+=random.normal(et/2*dsigma2,et)
      tau2+=random.normal(et/2*dtau2,et)
      
    #for i in range(J):
    #phi[i]=sample_phi(J,rho,i,mu,sigma2,tau2,y,phi)
    param_rec[t,:]=np.array([mu,sigma2,tau2])
  param_rec = DataFrame(param_rec)
  param_rec.columns=['mu','sigma2','tau2']
  return param_rec

Y=synthetic_data(SAMPLE_SIZE,J_0,rho_0,mu_0,tau2_0,sigma2_0)
param_rec=SGLD(SAMPLE_SIZE,BATCH_SIZE,J_0,ITER,Y,rho_0)
mu=param_rec['mu'].values
sigma2=param_rec['sigma2'].values
tau2=param_rec['tau2'].values
it=[*range(1, ITER+1)]
plt.figure(0)
plt.plot(it,mu,'-.',color='b',linewidth=1, label='mu')
plt.plot(it,sigma2,'-.',color='r',linewidth=1, label='sigma2')
plt.plot(it,tau2,'-.',color='g',linewidth=1, label='tau2')
plt.xlabel('iteration time')
plt.legend()
plt.show()