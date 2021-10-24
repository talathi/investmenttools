from datetime import date,datetime
import sys,os
import optparse
import pickle
import numpy as np
import scipy as sc
import pandas as pd

from investmenttools import Intrinsic_Value as IVS
reload(IVS)


### Assume Directory with pkl file exists...
### Assume indx object exists...
## For above run GetIndexData code base


Dir='/home/sachin/Work/DataSets/IndexData/MorningStarData_SP_2017_7_16_19_6_44'
df=pd.DataFrame(index=indx.List,columns=['Earnings_Slope','PE_Slope','PE','PB','SV'])

for stk in indx.List:
	data=IVS.GetStockDataFromPickle(Dir,stk)
	if len(data)!=0:
		
		if type(data['PE']) is not int:

			x=np.arange(len(data['PE']))
			y=np.array(data['PE'])
			pe_slope=sc.polyfit(x,y,1)[0]

			x=np.arange(len(data['Earnings Per Share USD']))
			y=np.array(data['Earnings Per Share USD'])
			ear_slope=sc.polyfit(x,y,1)[0]
		else:
			ear_slope=0;pe_slope=0
		SV=data['SV']
		PE=0;PB=0
		if type(data['PE'])!=int:
			PE=data['PE'][-1]
		if type(data['PB'])!=int:
			PB=data['PB'][-1]
	else:
		ear_slope=0
		pe_slope=0
		SV=0;PE=0;PB=0
	df.loc[stk]=[ear_slope,pe_slope,PE,PB,SV]


df=df.loc[~(df==0).all(axis=1)] ## remove rows with zero values 
df.where((df['Earnings_Slope']>0.1) & (df['PE_Slope']<-10)).dropna()


### Items from GetData_MarketWatch that I use
def InitiateSession():
  session=requests.session()
  return session

def CloseSession(session):
  session.close()


def GetStockPriceAndMarketCap(session,stock):
  bull=False
  StockPrice=GetDataFromGoogle(session,stock,'stock')
  
  while not bull:  
    try:
      stk=yp.Share(stock)
      bull=True
    except urllib2.HTTPError:
      print 'Try Again for: %s'%stock
      bull=False

  txt=stk.get_market_cap()
  if txt is None:
    URL='http://www.marketwatch.com/investing/stock/%s'%stock
    r=session.get(URL)
    s=r.content
    if s.find('Market Cap')!=-1:
      ind=s.find('Market Cap')
      txt=s[ind+60:ind+100]

  if txt is not None:
    patt=re.findall("\d+.\d+", txt)
    if len(patt)==0:
      patt=re.findall("\d+", txt)

    if 'B' in txt:
      MarketCap=map(float,patt)[0]*1e9
    if 'M' in txt:
      MarketCap=map(float,patt)[0]*1e6
  else:
    MarketCap=0
  return StockPrice,MarketCap

 def GetDataFromGoogle(session,stock,securitytype):
  # Time format conversion: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(1158955200))
  #if 'Mutual' in securitytype:
  URL='https://www.google.com/finance/getprices?q=%s&i=86400&p=2d&f=d,c,h,l,o,v'%stock.upper()
  #else:
  #  URL='https://www.google.com/finance/getprices?q=%s&i=3600&p=1d&f=d,c,h,l,o,v'%stock.upper()
  #print URL
  r=session.get(URL)
  d=r.content
  Data=[]
  tmp_Data=d.split('\n')
  Data=[]
  for d in tmp_Data:
    if ',' in d and 'DATE' not in d and len(d)!=0:
      Data.append(map(float,d.split(',')[1:]))
  #print Data
  if len(Data)!=0:
    return Data[-1][0]
  else:
    return 0