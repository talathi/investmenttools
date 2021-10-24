#Copyright (c) 2016, Sachin Talathi (sst7up@gmail.com)
#All rights reserved
# Redistribution and use in source, with or without modification,
# are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import requests
import sys
import json
import re
from numpy import *
import pylab as py
import time
from datetime import date
import dateutil.parser as dparser
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from datetime import date,datetime
import xlrd
import csv,glob
import os
import yahoo_finance as yp
import urllib2
HOME=os.environ['HOME']

regexf='[-+]?[0-9]*\.\d+' #for finding all floating point numbers
regexi='[-+]?[0-9]+'
Mnth=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

def WaitForDownload(downloaddir):
  File='%s/table.csv'%(downloaddir)
  bul=0
  while not(bul):
    if len(glob.glob(File))>0:
      bul=1
    else:
      bul=0
  

def MnthCheck(tmp):
  boolmnth=0
  for m in Mnth:
    if m in tmp:
      boolmnth=1
  return boolmnth
  
def InitiateSession():
  session=requests.session()
  return session

def CloseSession(session):
  session.close()

def GetSymbolList(Type):
  if Type=='SP':
    TxtFile='S&P-500-symbols.txt'
  if Type=='NYSE':
    TxtFile='NYSE-symbols.txt'
  if Type=='DOW':
    TxtFile='DJIA-symbols.txt'
  if Type=='My':
    TxtFile='MyStocks.txt'
  SymbolList=loadtxt(TxtFile,dtype='S40')
  return SymbolList


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

def ReadCSVFile(File):
  StockData={}
  o=open(File) 
  R=csv.reader(o)
  Dates=[];Value=[]
  for r in R:
    if 'Date' not in r:
      Dates.append(r[0])
      Value.append(map(float,r[1:]))
  Dates=array(Dates)
  Value=array(Value)
  StockData['Dates']=Dates[::-1]
  StockData['Value']=Value[::-1,:]
  o.close()
  return StockData
  
def GetTreasuryData(Period,sd,ed,downloaddir,savedir):
  #sd='2001-07-31'
  #ed='2014-11-20'
  if Period=='M':
    URL='http://research.stlouisfed.org/fred2/series/DGS1MO/downloaddata'
  if Period=='Y':
    URL='http://research.stlouisfed.org/fred2/series/DGS10/downloaddata'
    
  driver=webdriver.Chrome()
  driver.get(URL)
  e1=driver.find_element_by_id('form_frequency')
  e1.send_keys('Monthly')
  e2=driver.find_element_by_id('form_aggregation')
  e2.send_keys('End of Period')
  e3=driver.find_element_by_id('form_obs_start_date')
  e3.clear()
  e3.send_keys(sd)
  e4=driver.find_element_by_id('form_obs_end_date')
  e4.clear()
  e4.send_keys(ed)
  e5=driver.find_element_by_id('form_file_format')
  e5.send_keys('Excel')
  e6=driver.find_element_by_id('form_download_data_2')
  e6.submit()
  time.sleep(10)
  driver.quit()
  if Period=='M':
    mvfile='%s/DSG1MO_%s_%s.xls'%(savedir,sd,ed)
    strmv='mv %s/DGS1MO.xls %s'%(downloaddir,mvfile)
  if Period=='Y':
    mvfile='%s/DSG10_%s_%s.xls'%(savedir,sd,ed)
    strmv='mv %s/DGS10.xls %s'%(downloaddir,mvfile)
  os.system(strmv)
  w=xlrd.open_workbook(mvfile)
  sheet=w.sheet_by_index(0)
  nrows=sheet.nrows; ncols=sheet.ncols
  for i in range(nrows):
    if sheet.cell_value(i,1)=='VALUE':
      index=i
      break
  Data={}
  Data['Value']=[]
  Data['Dates']=[]
  for i in range(index+1,nrows):
    datetuple=xlrd.xldate_as_tuple(sheet.cell_value(i,0),w.datemode)
    if datetuple[1]<10:
      strmonth='0%d'%datetuple[1]
    else:
      strmonth='%d'%datetuple[1]
    if datetuple[2]<10:
      strday='0%d'%datetuple[2]
    else:
      strday='%d'%datetuple[2]
    stryear=str(datetuple[0])
    Data['Dates'].append('-'.join([stryear,strmonth,strday]))
    Data['Value'].append(sheet.cell_value(i,1))
    #Data.append(['-'.join([stryear,strmonth,strday]),sheet.cell_value(i,1)])
    
  return Data

def GetDataFromYahooAPI(session,stocklist=[]):
  print 'TO DO'

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
  if len(Data)!=0:
    return Data[-1][0]
  else:
    return None

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
 

def GetFinancialsQuarterly(session,stock):
  Data={}
  URL='http://www.marketwatch.com/investing/stock/%s/financials/income/quarter'%stock
  r=session.get(URL)
  s=r.content
  if s.find("data-chart")!=-1:
    Data['Revenue']= GetData(s,"Sales/Revenue","I")
    Data["GI"]=GetData(s,"Gross Income","I")
    Data['SG']=GetData(s,"SG&","I")
    Data['DA']=GetData(s,"Depreciation &","I")
    Data['NI']=GetData(s,"Net Income","I")
    Data['EBITDA']=GetData(s,"EBITDA","I")
    Data['EPSD']=GetData(s,"EPS (Diluted)","F")
  
    URL='http://www.marketwatch.com/investing/stock/%s/financials/balance-sheet/quarter'%stock
    r=session.get(URL)
    s=r.content
  
    Data['CA']=GetData(s,"Total Current Assets","I") 
    Data['CL']=GetData(s,"Total Current Liabilities","I")
    Data['LTD']=GetData(s,"Long-Term Debt","I")
    Data['SE']=GetData(s,"Total Shareholders' Equity","I")
  
    URL='http://www.marketwatch.com/investing/stock/%s/financials/cash-flow/quarter'%stock
    r=session.get(URL)
    s=r.content
    Data['FCF']=GetData(s,"Free Cash Flow","I")
  
    URL='http://www.marketwatch.com/investing/stock/%s'%stock
    r=session.get(URL)
    s=r.content
    txt= s[s.find("Market cap")+50:s.find("Market cap")+58]
    if txt.find('B')!=-1:
      Data['MC']=array(map(float32,re.findall("\d+.\d+", txt)))*10**9
    elif (txt.find('M'))!=-1:
      Data['MC']=array(map(float32,re.findall("\d+.\d+", txt)))*10**6
    else:
      Data['MC']=0 #Market cap of less than a million is not interesting to me
    
    txt= s[s.find("Div yield")+40:s.find("Div yield")+58]
    if txt.find('%')!=-1:
      Data['DY']=array(map(float32,re.findall("\d+.\d+", txt)))
  
    txt= s[s.find("Dividend")+40:s.find("Dividend")+58]
    if txt.find("N/A")==-1:
      Data['Div']=4*array(map(float32,re.findall("\d+.\d+", txt))) #estimating annual dividend
    else:
      Data['Div']=array([0.])
  
    txt= s[s.find("EPS")+40:s.find("EPS")+58]
    Data["EPSTTM"]=array(map(float32,re.findall("\d+.\d+", txt)))
  
    if s.find("Previous close")!=-1:
      txt=s[s.find("Previous close")+40:s.find("Previous close")+150]
    if s.find("Today's close")!=-1:
      txt=s[s.find("Today's close")+40:s.find("Today's close")+150]

    if txt.find(",")!=-1:
      txt=txt.replace(",","")
    Data["SP"]=array(map(float32,re.findall("\d+.\d+", txt)))
  
  return Data  

def GetFinancials(session,stock):
  Data={}
  URL='http://www.marketwatch.com/investing/stock/%s/financials'%stock
  r=session.get(URL)
  s=r.content
  if s.find("data-chart")!=-1:
    Data['Revenue']= GetData(s,"Sales/Revenue","I")
    Data["GI"]=GetData(s,"Gross Income","I")
    Data['SG']=GetData(s,"SG&","I")
    Data['DA']=GetData(s,"Depreciation &","I")
    Data['NI']=GetData(s,"Net Income","I")
    Data['EBITDA']=GetData(s,"EBITDA","I")
    Data['EPSD']=GetData(s,"EPS (Diluted)","F")
  
    URL='http://www.marketwatch.com/investing/stock/%s/financials/balance-sheet'%stock
    r=session.get(URL)
    s=r.content
  
    Data['CA']=GetData(s,"Total Current Assets","I") 
    Data['CL']=GetData(s,"Total Current Liabilities","I")
    Data['LTD']=GetData(s,"Long-Term Debt","I")
    Data['SE']=GetData(s,"Total Shareholders' Equity","I")
  
    URL='http://www.marketwatch.com/investing/stock/%s/financials/cash-flow'%stock
    r=session.get(URL)
    s=r.content
    Data['FCF']=GetData(s,"Free Cash Flow","I")
  
    URL='http://www.marketwatch.com/investing/stock/%s'%stock
    r=session.get(URL)
    s=r.content
    txt= s[s.find("Market cap")+50:s.find("Market cap")+58]
    if txt.find('B')!=-1:
      Data['MC']=array(map(float32,re.findall("\d+.\d+", txt)))*10**9
    elif (txt.find('M'))!=-1:
      Data['MC']=array(map(float32,re.findall("\d+.\d+", txt)))*10**6
    else:
      Data['MC']=0 #Market cap of less than a million is not interesting to me
    
    txt= s[s.find("Div yield")+40:s.find("Div yield")+58]
    if txt.find('%')!=-1:
      Data['DY']=array(map(float32,re.findall("\d+.\d+", txt)))
  
    txt= s[s.find("Dividend")+40:s.find("Dividend")+58]
    if txt.find("N/A")==-1:
      Data['Div']=4*array(map(float32,re.findall("\d+.\d+", txt))) #estimating annual dividend
    else:
      Data['Div']=array([0.])
  
    txt= s[s.find("EPS")+40:s.find("EPS")+58]
    Data["EPSTTM"]=array(map(float32,re.findall("\d+.\d+", txt)))
  
    if s.find("Previous close")!=-1:
      txt=s[s.find("Previous close")+40:s.find("Previous close")+150]
    if s.find("Today's close")!=-1:
      txt=s[s.find("Today's close")+40:s.find("Today's close")+150]

    if txt.find(",")!=-1:
      txt=txt.replace(",","")
    Data["SP"]=array(map(float32,re.findall("\d+.\d+", txt)))
  
  return Data  

def GetStockPriceAndMarketCap(session,stock):
  bull=False
  StockPrice=GetDataFromGoogle(session,stock,'stock')
  
  while not bull and StockPrice is not None:  
    try:
      stk=yp.Share(stock)
      bull=True
    except urllib2.HTTPError:
      print 'Try Again for: %s'%stock
      bull=False
  if StockPrice is not None:
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
  else:
    StockPrice=None;MarketCap=None
  return StockPrice,MarketCap
  
def GetFundamentals(Data,tim):
  Data["Fund"]={}
  Data["Fund"]["CAGR"]=[];Data["Fund"]["CR"]=[];Data["Fund"]["DPO"]=[];
  Data["Fund"]["EPS"]=[];Data["Fund"]["FCF"]=[];Data["Fund"]["FCFY"]=[];
  Data["Fund"]["LTDE"]=[];Data["Fund"]["NPM"]=[];Data["Fund"]["OM"]=[];
  Data["Fund"]["PE"]=[];Data["Fund"]["ROE"]=[];Data["Fund"]["SP"]=[];
  if len(Data["GI"])!=0:
    Data["Fund"]["OM"]=100.0*(array(Data["GI"])-array(Data["SG"]))/(array(Data["Revenue"]))
  if len(Data["NI"])!=0 and len(Data["Revenue"])!=0:
    Data["Fund"]["NPM"]=100.0*(array(Data["NI"]))/(array(Data["Revenue"]))
  if len(Data["NI"])!=0 and len(Data["SE"])!=0:
      Data["Fund"]["ROE"]=100.0*(array(Data["NI"]))/(array(Data["SE"]))
  if len(Data["CA"])!=0:
    Data["Fund"]["CR"]=1.0*(array(Data["CA"]))/(array(Data["CL"]))
  if len(Data["LTD"])!=0:
    Data["Fund"]["LTDE"]=100.0*(array(Data["LTD"]))/(array(Data["SE"]))
  if len(Data["FCF"])!=0:
    Data["Fund"]["FCF"]=array(Data["FCF"])
    if "MC" in Data.keys():
      Data["Fund"]["FCFY"]=100.0*Data["FCF"]/Data["MC"]
      
  if len(Data["EPSD"])!=0:
    if tim=='Y':
      Data["Fund"]["CAGR"]=100*((array(Data["EPSD"][-1])/array(Data["EPSD"][0]))**(1./4.)-1)
    else:
      Data["Fund"]["CAGR"]=100*(array(Data["EPSD"][-1])/array(Data["EPSD"][0])-1)
    
  Data["Fund"]["EPS"]=array(Data["EPSTTM"])
  Data["Fund"]["SP"]=Data["SP"]#100*Data["Div"]/Data["DY"]
  Data["Fund"]["DPO"]=100*Data["Div"]/Data["EPSTTM"]
  Data["Fund"]["PE"]=array(Data["Fund"]["SP"])/array(Data["EPSTTM"])
  return Data
    
def GetIntrinsicValue(Data):
  ############### Method 1 ######################
  #Need: Price, EPS, PE, Div, DPO, CAGR,
  Expected_Avg_Return=10 ## Assumption on Expected Avg. Return
  DFund=Data["Fund"]
  if DFund["CAGR"]>15: ## Assumption on Compound Annual Growth Rate
    CAGR=15.
  else:
    CAGR=10.
  
  if DFund["PE"]>=20: ## Assumption on PE
    PE=17.
  else:
    PE=12.
  
  if len(DFund["EPS"])!=0:
    
    EPS_Project=(DFund["EPS"])*(1+CAGR/100.)**5.
    SR_Project=PE*EPS_Project
  
  
    TE_Project=0
    for i in range(5):
      TE_Project+=(DFund["EPS"])*(1+CAGR/100.)**(i+1)
  
    Div_Project=TE_Project*DFund["DPO"]/100.
    
    SRD_Project=SR_Project+Div_Project
  
    SRD_Current=SRD_Project/((1+Expected_Avg_Return/100.)**5.)
    #print len(SRD_Current)
  
    #################### Method 2 #################################
    #Need FCF, MC, SP, CAGR and some additional assumptions on FCF growth rate; CAGR=Discount Rate; Terminal Rate;
    AvgFCF=mean(Data['FCF'][2:])
    TR=2
    DR=8
    GR=[10,8] 
    SO= int(Data['MC']/Data['SP'])
    Debt=Data['LTD'][-1]
  
    Future_FCF=[]; PV=[]
    f=AvgFCF*(1+GR[0]/100.0)
    Future_FCF.append(f)
    p=Future_FCF[-1]/(1+DR/100.)
    PV.append(p)
    for i in range(1,10):
      if i<=4:
        f=Future_FCF[-1]*(1+GR[0]/100.0)
      else:
        f=Future_FCF[-1]*(1+GR[1]/100.0)
      Future_FCF.append(f)
      p=Future_FCF[-1]/(1+DR/100.)**(i+1)
      PV.append(p)
  
    TYCF=Future_FCF[-1]*(1+TR/100.)
    TV=TYCF*100.0/((DR-TR)*(1+DR/100.0)**10.0)
    Future_FCF=array(Future_FCF); PV=array(PV)
    PVSum=sum(PV)    
    TPV=PVSum+TV
    IV=(TPV-Debt)/SO
  
    #print TYCF,PVSum, TPV, TV
    #print Future_FCF
    #print PV
    
    ############### Method 3 #######################
    Graham_Number=(Data["SP"]*sqrt(22.5*Data["NI"][-1]*Data["SE"][-1]))/Data["MC"]
  
    Data["Intrinsic"]=array([SRD_Current[0],IV,Graham_Number[0]])
  else:
    Data["Intrinsic"]=array([0.0,0.0,0.0])
  Data['MCFCF']=Data['MC']/Data['FCF'][-1] ##less than or equal to 10 good 
  
  #print DFund["CAGR"],EPS_Project
  return Data
  
  
def GetIndexData(Index,tim):
  Stock={}
  S=GetSymbolList(Index)
  session=InitiateSession()
  for s in S:
    print "Getting data for %s"%s
    if tim=='Y':
      Data=GetFinancials(session,s)
    else:
      Data=GetFinancialsQuaterly(session,s)
    if len(Data)>0:
      Stock[s]=Data
      Stock[s]=GetFundamentals(Stock[s],tim)
      Stock[s]=GetIntrinsicValue(Stock[s])
  return Stock

def GetStockData(session,s,tim):
  Stock={}
  print "Getting data for %s"%s
  if tim=='Y':
    Data=GetFinancials(session,s)
  else:
    Data=GetFinancialsQuarterly(session,s)
  if len(Data)>0:
    Stock=Data
    Stock=GetFundamentals(Stock,tim)
    Stock=GetIntrinsicValue(Stock)
  return Stock

def PrintStockData(session,s,tim):
  Data=GetStockData(session,s,tim)
  print 'Net Profit Margin=%s'%str(Data["Fund"]["NPM"])[1:-1]
  print 'Free Cash Flow Yield=%s'%str(Data["Fund"]["FCFY"])[1:-1]
  print 'Return on Equity=%s'%str(Data["Fund"]["ROE"])[1:-1]
  print 'Current Ratio=%s'%str(Data["Fund"]["CR"])[1:-1]
  print 'Long Term Debt-to-Equity=%s'%str(Data["Fund"]["LTDE"])[1:-1]
  print 'PE=%s'%str(Data["Fund"]["PE"])[1:-1]
  print 'EPS=%s'%str(Data["Fund"]["EPS"])[1:-1]
  print 'Compound Annual Growth Rate- 5 years=%s'%str(Data["Fund"]["CAGR"])[1:-1]
  print 'Share Price=%s'%str(Data["Fund"]["SP"])[1:-1]
  print 'Intrinsic Value: (Discount Cas Flow, Graham No.)=%f, %f'%(Data["Intrinsic"][0],Data["Intrinsic"][1])

def AnalyzeStockIndexData(Stock):
  for s in Stock.keys():
    CR=-1;lROE=0;lFCF=0;
    DFund=Stock[s]["Fund"]
    
    if "CR" in DFund.keys():
      if len(DFund["CR"])>0:
        CR=DFund["CR"][-1]
    if "ROE" in DFund.keys():
      lROE=len(where(Stock[s]["Fund"]["ROE"]<15)[0])
    if "FCF" in DFund.keys():
      lFCF=len(where(Stock[s]["Fund"]["FCF"]<0)[0])
    if CR>1 and lROE==0 and lFCF==0:
      IS=GetIntrinsicValue(Stock[s])
      #print IS["Intrinsic"][0]
      OV=100.*(Stock[s]["SP"][0]-IS["Intrinsic"][0])/(IS["Intrinsic"][0])
      if OV<15:
        print s,Stock[s]["Intrinsic"][0],Stock[s]["SP"][0],OV
    
if __name__=='__main__':
  
  if len(sys.argv)<2:
    print "Usage: python GetData_MatketWatch <Index>"
    sys.exit(0)
  Index=sys.argv[1]
  S=GetIndexData(Index)
  today=date.today()
  File='%s_%d_%d_%d.pkl'%(Index,today.month,today.day,today.year)
  o=open(File,'wb')
  pickle.dump(S,o)
  o.close()
