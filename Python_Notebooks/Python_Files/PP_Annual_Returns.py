import os,sys
import pandas as pd
import pandas_datareader as web
import numpy as np
from datetime import datetime as dt
import matplotlib
import seaborn as sns
#%matplotlib inline
import pylab as py
py.ion()

import scipy as sp
import scipy.optimize as scopt
import scipy.stats as spstats

file_path = os.path.dirname(os.path.realpath('__file__'))
sys_path='%s/../../'%file_path
sys.path.append(sys_path)
from investmenttools import PortfolioBuilder as PB
reload(PB)

def get_data_from_csv(filename,ticker):
	data=pd.read_csv(filename)
	data.set_index('Date',inplace=True)
	data.rename(columns={'Close':ticker},inplace=True)
	data=data[ticker]
	return data

# Get portfolio data
partial_data=PB.get_historical_closes(['shy','iau'],dt(2005,1,1),dt(2017,8,26))
shy=get_data_from_csv('%s/Data/SHY.csv'%file_path,'shy')
iau=get_data_from_csv('%s/Data/IAU.csv'%file_path,'iau')
itot=get_data_from_csv('%s/Data/ITOT.csv'%file_path,'itot')
tlt=get_data_from_csv('%s/Data/TLT.csv'%file_path,'tlt')
portfolio=pd.concat([shy,iau,itot,tlt],axis=1)
portfolio.set_index(pd.to_datetime(portfolio.index),inplace=True) 
portfolio.dropna(inplace=True)
portfolio.rename(columns={'Close':'itot'},inplace=True)
portfolio.dropna(inplace=True)
portfolio_list=list(portfolio.columns)

## Compute Annualized Returns for 1-Year and 2-Year holding period
Holding_Period=[1,2]
Years=np.arange(2005,2018)
portfolio_list=list(portfolio.columns)

Results=pd.DataFrame(columns=['HP1','HP2'],index=np.arange(len(Years)))

for duration in Holding_Period:
    Returns=[]
    for i in range(len(Years)-duration+1):
        val=PB.Backtest_Annual_Rebalance(portfolio_list,[.25,.25,.25,.25],\
                                         dt(Years[i],1,1),dt(Years[i]+duration-1,12,31),\
                                         initial=10000,stk_data=portfolio)
        rr=100*(((1+val['Total'].pct_change()).cumprod().iloc[-1])**(1./duration)-1)
        Returns.append(rr)
        #print Years[i]+duration-1,':',rr
    if duration==1:
        Results['HP1']=np.array(Returns)
    if duration==2:
        Returns.append(None)
        Results['HP2']=np.array(Returns)

Results.dropna(inplace=True)
Results.set_index(Years[0:-1],inplace=True)
print Results
_=Results.plot(kind='bar')
py.xticks(fontsize=15,fontweight='bold')
py.yticks(fontsize=15,fontweight='bold')
py.ylabel('Annualized Rate-of-Return',fontsize=14,fontweight='bold')
py.savefig('/home/sachin/Work/Applied_Value_Investor//Figures/PP_Annualized_Returns.png')
py.figure()

## Compare performance for PP and Buffet-Portfolio
bp_list=['shy','spy']
bp_weight=[.1,.9]
BP=PB.get_historical_closes(['shy','spy'],dt(2005,1,1),dt(2017,8,26))
BP.dropna(inplace=True)

PP_Growth=PB.Backtest_Annual_Rebalance(portfolio_list,weights,dt(2005,1,1),dt(2017,8,26),initial=10000,stk_data=portfolio)
BP_Growth=PB.Backtest_Annual_Rebalance(bp_list,bp_weight,dt(2005,1,1),dt(2017,8,26),initial=10000,stk_data=BP)

py.figure()
py.hold('on')
PP_Growth[['Total','Cash']].sum(axis=1).plot()
BP_Growth[['Total','Cash']].sum(axis=1).plot()
py.xticks(fontsize=15,fontweight='bold')
py.yticks(fontsize=15,fontweight='bold')
py.xlabel('')
py.legend(['PP','BP'],loc='best')
#py.savefig('/home/sachin/Work/Applied_Value_Investor//Figures/Compare_Performance_PPvsBP.png')
py.figure()

##### additional piece of code to compute annualized returns of portfolio with all equally weights etfs
def normalized_returns(data,fundname):
    dp=data.pct_change()
    dp.dropna(inplace=True)
    cp=(1+dp).cumprod()
    cp.dropna(inplace=True)
    num_funds=cp.shape[1]
    cp['total_%s'%fundname]=cp.sum(axis=1)/num_funds
    return cp

cp_712=normalized_returns(portfolio_712,'712')
cp_711=normalized_returns(portfolio_711,'711')
cp_PP=normalized_returns(portfolio_PP,'PP')
cp_BP=normalized_returns(portfolio_BP,'BP')

combined_returns=pd.concat([cp_712['total_712'],cp_711['total_711'],cp_PP['total_PP'],cp_BP['total_BP']],axis=1)
combined_returns.dropna(inplace=True)
combined_returns.index=pd.to_datetime(combined_returns.index)
dp=PB.calc_daily_returns(combined_returns)
ap=100*PB.calc_annual_returns(dp)
print ap

print 'Average Annual Returns over last 5-years:'
print ap.mean(axis=0)

## Plot Annual Returns Chart
## Plot Normalized returns
py.figure(figsize=(10,10))
ax1=py.subplot(211)
combined_returns.plot(ax=ax1)
py.xticks(fontsize=15,fontweight='bold')
py.yticks(fontsize=15,fontweight='bold')
py.ylabel('Cumulative Return',fontsize=15,fontweight='bold')
py.xlabel('',fontsize=15,fontweight='bold')
ax2=py.subplot(212)
ap.plot(kind='bar',ax=ax2)
py.xticks(fontsize=15,fontweight='bold')
py.yticks(fontsize=15,fontweight='bold')
py.ylabel('Annualized % Return',fontsize=15,fontweight='bold')