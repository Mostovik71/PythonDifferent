import numpy as np
import pandas as pd
import scipy.stats as sps
import seaborn as sns
import matplotlib.pyplot as plt
import math
sumx=0
sumxsqr=0
sumy=0
sumysqr=0
sumkorel=0
r=0
for x in dfx:
   sumx+=x
   sumxsqr+=x*x
for y in dfy:
    sumy += y
    sumysqr += y * y

sredneex=sumx/len(dfx)
sredneey=sumy/len(dfy)
dispx=(sumxsqr/len(dfx))-sredneex*sredneex#Выборочная дисперсия(Смещённая)
dispy=(sumysqr/len(dfy))-sredneey*sredneey#Выборочная дисперсия(Смещённая)
standartotklx=math.sqrt(dispx)
standartotkly=math.sqrt(dispy)
sumkorel=0
for x,y in zip(dfx,dfy):
    sumkorel+=(x-sredneex)*(y-sredneey)
r=(sumkorel/num_samples)/(standartotklx*standartotkly)
rs.append(r)
rs=[]
#in one cycle
for i in range(1,1000):
  num_samples=200
  means=np.array([0.0,0.0])
  r=np.array([[1,0.7],
              [0.7, 1]])
  y=np.random.multivariate_normal(means, r, num_samples)
  df=pd.DataFrame(y)
  df.columns=['norm1', 'norm2']
  dfx=df['norm1']
  dfy=df['norm2'] 
  sumx=0
  sumxsqr=0
  sumy=0 
  sumysqr=0
  sumkorel=0
  r=0
  for x in dfx:
   sumx+=x
   sumxsqr+=x*x
  for y in dfy:
    sumy += y
    sumysqr += y * y
    
  sredneex=sumx/len(dfx)
  sredneey=sumy/len(dfy)
  dispx=(sumxsqr/len(dfx))-sredneex*sredneex#Выборочная дисперсия(Смещённая)
  dispy=(sumysqr/len(dfy))-sredneey*sredneey#Выборочная дисперсия(Смещённая)
  standartotklx=math.sqrt(dispx)
  standartotkly=math.sqrt(dispy)
  sumkorel=0
  for x,y in zip(dfx,dfy):
    sumkorel+=(x-sredneex)*(y-sredneey)
  r=(sumkorel/num_samples)/(standartotklx*standartotkly)
  rs.append(r)
  
mu = 0.7
variance = (1/200)*(1-mu**2)**2
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x,sps.norm.pdf(x, mu, sigma))
#sns.kdeplot(rs)

plt.hist(rs,density=True, histtype='stepfilled')
