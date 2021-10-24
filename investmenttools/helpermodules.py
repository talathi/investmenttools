import requests
import re
from numpy import *
import pandas as pd
import pandas_datareader as web
import time
import datetime
from datetime import date,datetime
import os
from pandas_datareader._utils import RemoteDataError
HOME=os.environ['HOME']

def InitiateSession():
  session=requests.session()
  return session

def CloseSession(session):
  session.close()

def GetStockPriceAndMarketCap(session,stock):
  MarketCap=0
  try:
    StockPrice=GetDataFromGoogle(session,stock,'stock')
  except ValueError:
      dt=datetime.now()
      try:
          StockPrice = web.DataReader(stock.upper(), "yahoo", datetime(dt.year,dt.month,dt.day))['Close'].values[0]
          #StockPrice=web.DataReader(stock.upper(), "yahoo", datetime(dt.year,dt.month,dt.day))['Close'].values[0]
      except RemoteDataError as exp:
          try:
            StockPrice = web.DataReader(stock.upper(), "yahoo", datetime(dt.year, dt.month, dt.day-1))['Close'].values[0]
          except RemoteDataError as exp:
            try:
                StockPrice = web.DataReader(stock.upper(), "yahoo", datetime(dt.year, dt.month, dt.day-2))['Close'].values[0]
            except RemoteDataError as exp:
                StockPrice = \
                web.DataReader(stock.upper(), "yahoo", datetime(dt.year, dt.month, dt.day - 3))['Close'].values[0]
  txt=None
  '''
  while not bull:  
    try:
      stk=yp.Share(stock)
      bull=True
    except urllib2.HTTPError:
      print 'Try Again for: %s'%stock
      bull=False
    except KeyError:
    	break
  '''
  #txt=stk.get_market_cap()
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

def GetAnnualDataFromGoogle(session,stock,ref_year=None,ref_month=None):
  # Time format conversion: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(1158955200))
	curr_Year=time.gmtime()[0]
	curr_Month=time.gmtime()[1]

	URL='https://www.google.com/finance/getprices?q=%s&i=432000&p=10Y'%stock.upper()
	#print URL
	r=session.get(URL)
	d=r.content
	Data=[]
	tmp_Data=d.split('\n')

	Data=[]
	for d in tmp_Data:
		if ',' in d and len(d)!=0 and 'a' in d.split(',')[0] and '=' not in d:
      #gm_time=time.gmtime(int(d.split(',')[0].split('a')[1]))
      #print gm_time[0],gm_time[1],gm_time[2]
			Data.append([int(d.split(',')[0].split('a')[1]),map(float,d.split(',')[1:])])
  
	Ann_Data=[]
	if ref_year is None:
		_Yr=curr_Year-10
		ref_year=curr_Year-10
		if _Yr<time.gmtime(Data[0][0])[0]:
			_Yr=time.gmtime(Data[0][0])[0]
			ref_year=_Yr
			ref_month=time.gmtime(Data[0][0])[1]
			curr_Month=ref_month

	if ref_year is not None:
		_Yr=ref_year

	if ref_month is not None:
		curr_Month=ref_month

	for i in range(len(Data)):
		gm_time=time.gmtime(Data[i][0])
		if gm_time[0]==_Yr and gm_time[1]==curr_Month:
			Ann_Data.append([Data[i][1][0]])
			_Yr+=1
  
	return Ann_Data,ref_year,ref_month

def GetDataFromGoogle(session,stock,securitytype):
 	URL='https://www.google.com/finance/getprices?q=%s&i=86400&p=2d&f=d,c,h,l,o,v'%stock.upper()
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

def GetHistoricalData(session,stock,start_date,end_date,securitytype):
  #Get Historical data for stock for given date range at a frequency of period={d,w,m}
  #Example: GetHistoricalData('aapl',(2014,5,1),(2014,5,10),'m'))
  
  Start_Year,Start_Month,Start_Date=start_date
  End_Year,End_Month,End_Date=end_date
  sd=int(time.mktime((Start_Year,Start_Month,Start_Date,0,0,0,0,0,0)))
  ed=int(time.mktime((End_Year,End_Month,End_Date,0,0,0,0,0,0)))
  URL='https://www.google.com/finance/getprices?q=%s&i=432000&p=10Y&p=2d'%stock.upper()
  #print URL
  r=session.get(URL)
  d=r.content
  Data=[]
  tmp_Data=d.split('\n')
  Data=[]
  for d in tmp_Data:
    if ',' in d and len(d)!=0 and 'a' in d.split(',')[0] and '=' not in d:
      #gm_time=time.gmtime(int(d.split(',')[0].split('a')[1]))
      #print gm_time[0],gm_time[1],gm_time[2]
      Data.append([int(d.split(',')[0].split('a')[1]),map(float,d.split(',')[1:])])

  Historic_Data=[]

  for d in Data:
    if d[0]>=sd and d[0]<=ed:
      Historic_Data.append([d[0],d[1][0]])
  return Historic_Data


def GetDataForPortfolio(session,MF_List,sd,ed,securitytype):
  MFData={}
  bul=1
  count=0;
  for m,s in zip(MF_List,securitytype):
    print m
    Data=GetHistoricalData(session,m,sd,ed,s)
    MFData[count]=Data
    count=count+1  
  return MFData


def convertIVSDatatoPandas(Data):
	## This module converts the IVS data on stock fundamentals into Pandas DataFrame	
	dp=pd.DataFrame.from_dict(Data)
	ref_date=dp[''].iloc[0]
	dp.set_index([''],inplace=True)
	val=list(dp.columns[(dp.iloc[0]==ref_date)])
	dp.drop(val,1,inplace=True)
	return dp