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
import sys,os
import re
from numpy import *
import pylab as py
from selenium import webdriver
import csv
import glob
import platform
import pickle
import helpermodules as GM
reload(GM)
import ReadMagicFormulaList as RF
reload(RF)
MF=RF.ReadMagicFormula('abc@gmail.com','abc')

HOME=os.environ['HOME']

def correct_for_comma(number_list):
  correct_num_list=[]
  for n in number_list:
    if ',' in n:
      nr=n.replace(',','')
      correct_num_list.append(nr)
    else:
      correct_num_list.append(n)
  return correct_num_list

def GetKeyRatios(Stock,profile_path=''):
  #display = Display(visible=0, size=(800, 600))
  #display.start(
  if 'Linux' in platform.platform():
    #profile = webdriver.FirefoxProfile(profile_path)
    if os.path.isdir('%s/.config/google-chrome'%HOME):
      strrm='rm -rf %s/.config/google-chrome'%HOME
      os.system(strrm)
    driver=webdriver.Chrome()
  if 'Darwin' in platform.platform():
    driver=webdriver.Safari()
  URL='http://financials.morningstar.com/valuation/price-ratio.html?t=%s'%Stock
  driver.get(URL)
  txt=driver.page_source
  Numstr=[m.start() for m in re.finditer('abbr="Price/Earnings',txt)]
  while len(Numstr)==0:
    driver.get(URL)
  PE_Ratio=map(float32,correct_for_comma(re.findall('\d+.\d+',txt[Numstr[0]:Numstr[1]])))
  Numstr=[m.start() for m in re.finditer('abbr="Price/Book',txt)]
  PB_Ratio=map(float32,correct_for_comma(re.findall('\d+.\d+',txt[Numstr[0]:Numstr[1]])))
  Numstr=[m.start() for m in re.finditer('abbr="Price/Sales',txt)]
  PS_Ratio=map(float32,correct_for_comma(re.findall('\d+.\d+',txt[Numstr[0]:Numstr[1]])))
  Numstr=[m.start() for m in re.finditer('abbr="Price/Cash',txt)]
  PC_Ratio=map(float32,correct_for_comma(re.findall('\d+.\d+',txt[Numstr[0]:Numstr[1]])))
  driver.quit()
  #display.stop()
  return array(PE_Ratio),array(PB_Ratio),array(PS_Ratio),array(PC_Ratio)

def GetStockDataFromWeb(Stock,profile_path='',type='Small_Cap'):
  Data={}
  Dir=MF.GetMorningStartData([Stock.upper()],type)
  #print Dir
  if os.path.isdir(Dir):
    Data_Stock=MF.ReadCSVFiles(Dir)
    Data=Data_Stock[Stock]
    strrm='rm -rf %s'%Dir
    #PE,PB,PS,PC=GetKeyRatios(Stock,profile_path)
    session=GM.InitiateSession()
    SV,MC=GM.GetStockPriceAndMarketCap(session,Stock)
    GM.CloseSession(session)
    if 'Earnings Per Share USD' in Data.keys():
        Data['PE']=array([SV/val for val in Data['Earnings Per Share USD'] if val!=0])
    else:
      Data['PE']=0
    if 'Book Value Per Share * USD' in Data.keys():
      Data['PB']=array([SV/val for val in Data['Book Value Per Share * USD'] if val!=0])
    else:
      Data['PB']=0
    Rev=[]
    if 'Revenue USD Mil' in Data.keys():
      for r in Data['Revenue USD Mil']:
        if type(r)==float:
          Rev.append(r)
        elif ',' in r:
          Rev.append(float(r.replace(',','')))
        else:
          Rev.append(1.)
      Data['PS']=MC*1e-6/Rev
    else:
      Data['PS']=0
    if 'Free Cash Flow Per Share * USD' in Data.keys():
      Data['PC']=array([SV/val for val in Data['Free Cash Flow Per Share * USD'] if val!=0])
    else:
      Data['PC']=0
    Data['SV']=SV
    strrm ='rm -rf %s'%Dir
    os.system(strrm)
  return Data

def GetStockDataFromFolder(Dir,Stock,profile_path=''):
  Data={}
  CVSList=os.listdir(Dir)
  stock_exists=0
  for c in CVSList:
    if Stock.upper()==c.split('Key')[0].replace(' ',''):
      stock_exists=1
      Ticker= c.split('Key')[0].replace(' ','')
      File='%s/%s'%(Dir,c)
      o=open(File,'rb')
      R=csv.reader(o)
      U=[]
      for r in R:
        U.append(r)
      o.close()
      for i in range(len(U)):
        if len(U[i])>2:
          if '2012' in ''.join(U[i][1:]):
            Data[U[i][0]]=array(U[i][1:])
          elif ',' in ''.join(U[i][1:]):
            X=array(U[i][1:])
            X[X=='']='-9999'
            Data[U[i][0]]=map(float,[x.replace(',','') for x in X])
          elif '2012' not in ''.join(U[i][1:]) or ',' not in ''.join(U[i][1:]):
            X=array(U[i][1:])
            X[X=='']='0'
            Data[U[i][0]]=map(float,X)
        else:
          continue
      #print "Get Stock Data For:",Stock
      #print profile_path
      session=GM.InitiateSession()
      #print c
      SV,MC=GM.GetStockPriceAndMarketCap(session,Stock)
      GM.CloseSession(session)
      Data['MC']=MC
      if 'Earnings Per Share USD' in Data.keys():
        Data['PE']=array([SV/val for val in Data['Earnings Per Share USD'] if val!=0])
      else:
        Data['PE']=0
      if 'Book Value Per Share * USD' in Data.keys():
        Data['PB']=array([SV/(float(val)+.0000001) for val in Data['Book Value Per Share * USD'] if val!=''])
      else:
        Data['PB']=0
      Rev=[]
      if 'Revenue USD Mil' in Data.keys():
        for r in Data['Revenue USD Mil']:
          if type(r)==float:
            Rev.append(r)
          elif ',' in r:
            Rev.append(float(r.replace(',','')))
          else:
            Rev.append(1.)
        Data['PS']=MC*1e-6/array(Rev)
      else:
        Data['PS']=0
      if 'Free Cash Flow Per Share * USD' in Data.keys():
        Data['PC']=array([SV/(float(val)+.0000001) for val in Data['Free Cash Flow Per Share * USD'] if val!=''])
      else:
        Data['PC']=0
      Data['SV']=SV
      
  if stock_exists==0:
    print '%s stock data not available in %s'%(Stock,Dir)
  return Data    


def GetStockDataFromPickle(Dir,Stock):
  Data={}
  file='%s/%s_Combined.pkl'%(Dir,Stock)
  print file
  if os.path.isfile(file):
    o=open(file)
    Data=pickle.load(o)
    o.close()
  return Data
  
def AppendStockDataFromFolder(Dir='',List=[],profile_path=''):
  Files=glob.glob('%s/*.*'%Dir)
  List_Stock=[]
  if len(List)==0:
    for f in Files:
      stock=f.split('/')[-1].split(' ')[0]
      List_Stock.append(stock)
  else:
    List_Stock=List
  #print List_Stock
  #sys.exit(1)
  for stock in List_Stock:
    print stock
    PklFile='%s/%s_Combined.pkl'%(Dir,stock)
    if not PklFile in Files:
      Data=GetStockDataFromFolder(Dir,stock,profile_path)
      o=open('%s/%s_Combined.pkl'%(Dir,stock),'wb')
      pickle.dump(Data,o)
      o.close()
    

def ComputeCAGR(Data,type):
  #Data is obtained in the form of MagicFormula Data from MorningStar
  seterr(divide='ignore', invalid='ignore')
  CAGR=0
  if 'Earnings Per Share USD' in Data.keys():
    EPSD=array(Data['Earnings Per Share USD'])
  else:
    EPSD=zeros((11,))
  if len(EPSD)<6:
    print 'Not enough historical data on %s available'%Stock
    print 'Remove the Stock from consideration for further analysis'
    sys.exit(0)
  
  if len(EPSD)<11:
    print 'Computing CAGR based on last 5 years of EPS '
    CAGR=100*((EPSD[-1]/EPSD[0])**(1./5.0)-1)
  
  if len(EPSD)==11:
    #print 'Historical Data for last 10 years is available'
    #print 'More confident about estimates for expected CAGR'
    cagr_ref=10000
    cagr_avg=0
    for i in range(5):
      cagr=100*((EPSD[-1]/EPSD[i])**(1./(10-i))-1)
      cagr_avg+=cagr
      #print cagr
      if cagr<cagr_ref and cagr>=0:
        cagr_ref=cagr
    if type.upper()=='MIN':
      CAGR=cagr_ref
    if type.upper()=='AVG':
      CAGR=cagr_avg/5.0
      
  return CAGR

def ComputeIntrinsicValue(Data,ER,type,Provided_CAGR=0.0):
  IV=0;CAGR=0;Out_CAGR=0
  if 'Earnings Per Share USD' in Data.keys():
    EPSDVec=array(Data['Earnings Per Share USD'])
  else:
    EPSDVec=zeros((11,))
  Out_CAGR=ComputeCAGR(Data,type)
  if isnan(Out_CAGR):
    CAGR=0
  if Out_CAGR<=15 and Out_CAGR>=5:
    CAGR=Out_CAGR
  if Out_CAGR>15:
    if Provided_CAGR!=0:
      CAGR=Provided_CAGR
    else:
      CAGR=15
  
  if Out_CAGR<5:
    if Out_CAGR>0:
      CAGR=Out_CAGR
    else:
      CAGR=Provided_CAGR
  EPSD=EPSDVec[-1]


  ind=where(Data['PE']>0)[0]
  #print ind

  if len(ind)>1:#type(Data['PE'])!=int#array(Data['PE']).any()!=0:
    PEVec = Data['PE'][ind]

    if type.upper()=='MIN':
      PE=min(PEVec)

    if type.upper()=='AVG':
      PE=median(PEVec)
    
    if 'Dividends USD' in Data.keys():
      DivVec=Data['Dividends USD']
    else:
      DivVec=zeros((11,))
    Div=DivVec[-1]
    DPO=Div/EPSD
    EPS_Project=EPSD*(1+CAGR/100.)**(10.)
    SV_Project=PE*EPS_Project
      
    TE_Project=0
    for i in range(10):
      TE_Project+=EPSD*(1+CAGR/100)**(i+1)
      
    Div_Project=DPO*TE_Project
    SVD_Project=SV_Project+Div_Project
      
    IV=SVD_Project/((1+ER/100.)**10.)
  else:
    IV=0
  #print IV
  return IV,CAGR,Out_CAGR

def PrintIntrinsicValue(Data,Provided_CAGR=8.0):
  ER=[5,10,15,5,10,15]
  Cstr=['Avg','Avg','Avg','Min','Min','Min']
  
  for i in range(6):
    IV,CAGR,Out_CAGR=ComputeIntrinsicValue(Data,ER[i],Cstr[i].upper(),Provided_CAGR=Provided_CAGR)
    print 'At {0} ER and CAGR-{1}: CAGR={2}; IV={3}'.format(ER[i],Cstr[i],CAGR,IV)
  
def IntrinsicValues(Dir='',Data={},ER=10,Provided_CAGR=8,Dollar_Available=100000.,Stock_File='',cagr_method='Min'):
  ### Read Virtual Stock Investment Data
  Virtual_Stock_List=[]
  Virtual_Stock_Count=[]
  if os.path.isfile(Stock_File):
    o=open(Stock_File)
    D=o.readlines()
    for d in D:
      Virtual_Stock_List.append(d.split(' ')[0])
      Virtual_Stock_Count.append(int(d.split(' ')[1]))
  #print Dir
  if len(Dir)!=0:
    Files=glob.glob('%s/*.pkl'%Dir)

    print 'At ER: {0} and Default CAGR: {1}'.format(ER,Provided_CAGR)
    printstr='Stock| Price   | Intrin Val| CAGR'
    print printstr
    print len(printstr)*'*'
    criterion_Data=[]
    for f in Files:
      #print f
      fstats=os.stat(f)
      if fstats.st_size!=0:
        o=open(f)
        Data=pickle.load(o)
        o.close()
        if len(Data)!=0:
          IV,CAGR,Out_CAGR=ComputeIntrinsicValue(Data,ER,cagr_method,Provided_CAGR=Provided_CAGR)
          if isnan(Out_CAGR):
            IV,CAGR,Out_CAGR=ComputeIntrinsicValue(Data,ER,'Min',Provided_CAGR=Provided_CAGR)
          if IV==IV:
            print '{0:^5}| {1:7.2f}| {2:8.2f}| {3:5.2f}'.format(f.split('/')[-1].split('_')[0],Data['SV'],IV,CAGR)
          if Data['SV']<=IV and Out_CAGR>0 and Out_CAGR<1000:
            criterion_Data.append([f.split('/')[-1].split('_')[0],Data['SV'],IV,CAGR])
            #print '{0:^5}| {1:^8}| {2:10.2f}| {3:5.2f}'.format(f.split('/')[-1].split('_')[0],Data['SV'],IV,CAGR)

    print '***** List of Stocks that meet CAGR>=%.1f and Expected-Return >=%.1f Criterion******'%(Provided_CAGR,ER)
    num_stocks_meeting_criterion=len(criterion_Data)

   # print num_stocks_meeting_criterion
    for i in range(len(criterion_Data)):
       #print criterion_Data[i]
       num_stocks_purchase=floor(Dollar_Available/num_stocks_meeting_criterion/criterion_Data[i][1])
       if criterion_Data[i][0] in Virtual_Stock_List:
         ind=Virtual_Stock_List.index(criterion_Data[i][0])
         num_exist_stock=Virtual_Stock_Count[ind]
         num_stocks_purchase=num_stocks_purchase-num_exist_stock
       if criterion_Data[i][2]>0.0:
        print '{0:^5}| {1:7.2f}| {2:8.2f}| {3:5.2f} |{4:4.0f}'.format(criterion_Data[i][0],criterion_Data[i][1],criterion_Data[i][2],criterion_Data[i][3],num_stocks_purchase)
       #print '%s: %d'%(criterion_Data[i][0],num_stocks_purchase)
    
  else:
    IV,CAGR,Out_CAGR=ComputeIntrinsicValue(Data,ER,'AVG',Provided_CAGR=Provided_CAGR)
    if isnan(Out_CAGR):
      IV,CAGR,Out_CAGR=ComputeIntrinsicValue(Data,ER,'Min',Provided_CAGR=Provided_CAGR)
    print 'At ER: {0} and Default CAGR: {1}'.format(ER,Provided_CAGR)
    print '{0:^8}| {1:10.2f}| {2:5.2f}'.format(Data['SV'],IV,CAGR)

def Plot_Ratios(Data,apr):
  class prettyfloat(float):
    def __repr__(self):
      return '%.2f'%self

  import pprint as pp
  #print Data.keys()
  set_printoptions(precision=2)
  print 'ROA:', map(prettyfloat,Data['Return on Assets %'])
  print 'ROE:', Data['Return on Equity %']
  print 'ROIC:', Data['Return on Invested Capital %']
  print 'Net Margin:', Data['Net Margin %']
  print 'Asset Turnover:', Data['Asset Turnover']
  print 'Inventory Turnover:', Data['Inventory Turnover']
  print 'Fin Lev:', Data['Financial Leverage']
  print 'DSO:', Data['Days Sales Outstanding']
  print 'D/E Ratio:', Data['Debt/Equity']
  print 'Interest Cov:', Data['Interest Coverage']
  print 'Quick R:', Data['Quick Ratio']
  print 'Free Cash Flow Growth %:', Data['Free Cash Flow Growth % YOY']
  print 'Dividend Yield:', 100*array(Data['Dividends USD'])/Data['SV']
  print 'Earnings per share',Data['Earnings Per Share USD']
  print 'PE:', map(prettyfloat,Data['PE'])
  print 'PB:', map(prettyfloat,Data['PB'])
  print 'PS:', map(prettyfloat,Data['PS'])
  print 'PC:', map(prettyfloat,Data['PC'])
  '''
  py.figure();
  py.subplot(541);py.plot(Data['Return on Assets %'])
  py.legend(['ROA'],loc='best')
  py.subplot(542);py.plot(Data['Return on Equity %'])
  py.legend(['ROE'],loc='best')
  py.subplot(543);py.plot(Data['Return on Invested Capital %'])
  py.legend(['ROIC'],loc='best')
  py.subplot(544);py.plot(Data['Net Margin %'])
  py.legend(['Profit Margin'],loc='best')
  py.subplot(545);py.plot(Data['Asset Turnover'])
  py.legend(['Asset Turnover'],loc='best')
  py.subplot(546);py.plot(Data['Financial Leverage'])
  py.legend(['Fin Lev'],loc='best')
  py.subplot(547);py.plot(Data['Gross Margin %'])
  py.legend(['Gross M'],loc='best')
  py.subplot(548);py.plot(Data['Fixed Assets Turnover'])
  py.legend(['Fixes Asset T'],loc='best')
  py.subplot(5,4,11);py.plot(Data['Inventory Turnover'])
  py.legend(['Inventory T'],loc='best')
  py.subplot(5,4,14);py.plot(Data['Days Sales Outstanding'])
  py.legend(['DSO'],loc='best')
  py.subplot(5,4,9);py.plot(Data['Debt/Equity'])
  py.legend(['Debt/Equity'],loc='best')
  py.subplot(5,4,12);py.plot(Data['Interest Coverage'])
  py.legend(['Interest Cov'],loc='best')
  py.subplot(5,4,10);py.plot(Data['PE'])
  py.legend(['PE'],loc='best')
  py.subplot(5,4,15);py.plot(Data['Quick Ratio'])
  py.legend(['Quick R'],loc='best')
  py.subplot(5,4,16);py.plot(100*array(Data['Dividends USD'])/Data['SV'])
  py.legend(['Dividend Yield'],loc='best')
  py.subplot(5,4,17);py.plot(Data['Free Cash Flow Growth % YOY'])
  py.legend(['FCF growth'],loc='best')

  if apr is not None:
    if len(apr)!=0:
      py.subplot(5,4,13);py.plot(apr)
      py.legend(['Ann Price Ratio'],loc='best')
  '''
if __name__=='__main__':
  if len(sys.argv)<2:
    print 'python Intrinsic_Value.py <Dir>'
  else:
    Dir=sys.argv[1]
  Files=os.listdir(Dir)
  
  for f in Files:
    Data=[]
    Ticker=f.split('Key')[0].replace(' ','')
    print Ticker
    DS=GetStockDataFromFolder(Dir,Ticker)
    if 'Earnings Per Share USD' in DS.keys():
      EPSD=DS['Earnings Per Share USD'][-1]
    else:
      EPSD=0.0
    (SV,EV,V1,V2,V3)=(DS['SV'],100*EPSD/DS['SV'],ComputeIntrinsicValue(DS,5),ComputeIntrinsicValue(DS,10),ComputeIntrinsicValue(DS,15))
    Data.append([Ticker,(SV,EV,V1,V2,V3)])
    