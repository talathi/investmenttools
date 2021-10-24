import requests
import sys,os,getopt
import json
import re
from numpy import *
import pylab as py
import time
from datetime import date,datetime
import dateutil.parser as dparser
from lmfit import minimize, Parameters, Parameter, report_fit
from GetData_MarketWatch import DownloadHistoricalData
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import csv
import pickle

from GetData_MarketWatch import DownloadHistoricalData,InitiateSession,CloseSession,GetDataIndex
import optparse

#Helper Functions###
def GetRebalanceDates(DayOne,DayEnd,RebalancePeriod):
  Start_Year,Start_Month,Start_Day=int(DayOne.split('-')[0]),int(DayOne.split('-')[1]),int(DayOne.split('-')[2])
  End_Year,End_Month,End_Day=int(DayEnd.split('-')[0]),int(DayEnd.split('-')[1]),int(DayEnd.split('-')[2])
  if Start_Day==1:
    Start_DayStr='0%d'%Start_Day
  elif Start_Day>1 and Start_Day<10:
    Start_DayStr='0%d'%(Start_Day-1)
  else:
    Start_DayStr='%d'%(Start_Day-1)
  r=RebalancePeriod
  RebalanceDates=[]
  
  cY=Start_Year
  cM=Start_Month
  
  if cY+r/12>End_Year:
    cY=End_Year
  
  while(cY<End_Year):
      rM=mod(cM+r,12)
      if rM==0:
        rM=12
      if rM<cM+r:
          cY=cY+1
      if cY<End_Year:
        if rM>9:
          RebalanceDates.append('%d-%d-%s'%(cY,rM,Start_DayStr))
        else:
          RebalanceDates.append('%d-0%d-%s'%(cY,rM,Start_DayStr)) 
        #print (cY,rM,Start_Day-1)
      if cY==End_Year and rM<=End_Month:
        if rM>9:
          RebalanceDates.append('%d-%d-%s'%(cY,rM,Start_DayStr))
        else:
          RebalanceDates.append('%d-0%d-%s'%(cY,rM,Start_DayStr))
        #print (cY,rM,Start_Day-1)
      cM=rM
      if (cY==End_Year):
        if cM+r>End_Month:
          continue
        while (cM<End_Month):
          rM=mod(cM+r,12)
          if rM==0:
            rM=12
          if rM<=End_Month:
            if rM>9:
              RebalanceDates.append('%d-%d-%s'%(cY,rM,Start_DayStr))
            else:
              RebalanceDates.append('%d-0%d-%s'%(cY,rM,Start_DayStr))
            cM=rM
          if cM+r>End_Month:
            break
      
  return RebalanceDates

def DateStr(sd):
  #Input sd: tuple (1,1,2014)
  Year=sd[2];Month=sd[1];Day=sd[0]
  if Day<10:
    DayStr='0%d'%Day
  else:
    DayStr='%d'%Day
  if Month<10:
    MonthStr='0%d'%Month
  else:
    MonthStr='%d'%Month
  return '-'.join([str(Year),MonthStr,DayStr])
  
def DatetoString(datelist):
  datestr=[]
  datestr.append(str(datelist[0]))
  if datelist[1]<10:
    datestr.append('%s%s'%(0,str(datelist[1])))
  else:
    datestr.append(str(datelist[1]))
  if datelist[2]<10:
    datestr.append('%s%s'%(0,str(datelist[2])))
  else:
    datestr.append(str(datelist[2]))
  return '-'.join(datestr)

## Main Object Class
class HistoricalDataAnalysis(object):
  def __init__(self,Portfoliofile,Directory,sd,ed):
    o=open(Portfoliofile)
    Data=o.readlines()
    o.close()
    Stocks=[]
    Stocks_Weight={}
    for d in Data:
      s=d.split('\n')[0].split(' ')[0]
      Stocks.append(s)
      Stocks_Weight[s]=(float(d.split('\n')[0].split(' ')[1])/100)
    self.Portfolio=Stocks
    self.Portfolio_Weight=Stocks_Weight
    
    self.Portfolio.append('^GSPC')
    self.Portfolio_Weight['^GSPC']=1.0
    
    #driver=webdriver.Firefox()
    #self.driver=driver
    self.start_date=sd #tuple: (1,1,2014)
    self.end_date=ed #tuple (2,1,2014)
    self.Directory=Directory
    
  def ObtainPortfolioData(self):
    ## In browser goto about:support to find the profile_path for FireFox
    profile_path='/Users/sachintalathi/Library/Application Support/Firefox/Profiles/3rmd0qv9.default'  
    profile = webdriver.FirefoxProfile(profile_path)
    driver=webdriver.Firefox(profile)
    for s in self.Portfolio:
      DownloadHistoricalData(driver,self.Directory,s.upper(),self.start_date,self.end_date,'d')
    driver.close()
  
  def ConvertPortfolioData(self):
    StockData={}
    for s in self.Portfolio:
      StockData[s.upper()]={}
      File='%s/%s.csv'%(self.Directory,s.upper())
      print File 
      if os.path.isfile(File):
        o=open(File) 
        R=csv.reader(o)
        Dates=[];Value=[]
        for r in R:
          if 'Date' not in r:
            Dates.append(r[0])
            Value.append(map(float,r[1:]))
        Dates=array(Dates)
        Value=array(Value)
        StockData[s.upper()]['Dates']=Dates[::-1]
        StockData[s.upper()]['Value']=Value[::-1,:]
        o.close()
    return StockData

  def PlotNormalizedPortfolio_StockData(self,S):
      DayOne=DateStr(self.start_date)
      DayEnd=DateStr(self.end_date)
      count=0
      
      for s in self.Portfolio:
        if count==0:
          DayEnd=S[s.upper()]['Dates'][-1]
        count=count+1
        l=len(S[s.upper()]['Dates'])
        S[s.upper()]['Portfolio_NumStocks']=zeros(l,)
        S[s.upper()]['Portfolio_Value']=S[s.upper()]['Portfolio_NumStocks']*S[s.upper()]['Value'][:,5]
        if S[s.upper()]['Dates'][0]>=DayOne:
            DayOne=S[s.upper()]['Dates'][0]
      print DayOne
      Stats={}
      py.figure()
      Stocklist=[]
      count=0
      
      AllStockData=[]
      AllStocklist=[]
      for s in self.Portfolio:
        if s!='^GSPC':
          Stocklist.append(s)
          AllStocklist.append(s)
        else:
          Stocklist.append('S&P500')
          
        ind=where(S[s.upper()]['Dates']>=DayOne)[0]
        SDates=S[s.upper()]['Dates'][ind]
        SData=S[s.upper()]['Value'][ind,5]
        SDataNorm=100*SData/SData[0]
        if count==0 and s!='^GSPC':
          SMean=self.Portfolio_Weight[s]*SDataNorm
        if count !=0 and s!='^GSPC':
          SMean+=self.Portfolio_Weight[s]*SDataNorm
        count+=1
        growth=(SDataNorm[-1]-SDataNorm[0])/SDataNorm[0]
        stdval=std(SDataNorm)
        Stats[s.upper()]=[growth,stdval] 
        if s!='^GSPC':
          AllStockData.append(SDataNorm)
          py.plot(SDataNorm,linewidth=.5)
        else:
          IndexNorm=SDataNorm
          py.plot(SDataNorm,'--r',linewidth=3)
        py.hold('on')
      py.plot(1.0*SMean,'--k',linewidth=3)
      py.xticks([0,len(SDataNorm)],[DayOne,DayEnd],rotation=45)
      Stocklist.append('Weighted-Mean')
      py.legend(Stocklist,loc='best')
      py.tight_layout() 
      py.savefig('%s/Timeseries.png'%self.Directory)
      Stats['beta']=cov(SMean,IndexNorm)/var(IndexNorm) ##Compute beta for the portfolio
      #Estimate Portfolio Correlation 
      corr=corrcoef(array(AllStockData))
      fig=py.figure();
      ax=fig.add_subplot(111)
      pax=ax.pcolor(corr,cmap=py.get_cmap('spectral'),vmin=-1, vmax=1)
      py.xticks(range(len(corr)),AllStocklist)
      py.yticks(range(len(corr)),AllStocklist)
      labels = ax.get_xticklabels() 
      py.tight_layout()
      py.savefig('%s/Correlation.png'%self.Directory)
      for label in labels: 
          label.set_rotation(90)
      cbar = fig.colorbar(pax,ticks=[-1,-.5,-.25,0,.25,.5,1])
      return SMean,IndexNorm,Stats
      
  def AnalyzePortfolioData_WithRebalancing(self,S,Investment,Weight,RebalancePeriod):
    DayOne='2010-01-15'
    count=0
    #Weight={'VTI':.25,'AGG':.25,'GLD':.25,'VCSH':.25}
    for s in self.Portfolio:
      if count==0:
        DayEnd=S[s.upper()]['Dates'][-1]
      count=count+1
      l=len(S[s.upper()]['Dates'])
      S[s.upper()]['Portfolio_NumStocks']=zeros(l,)
      S[s.upper()]['Portfolio_Value']=S[s.upper()]['Portfolio_NumStocks']*S[s.upper()]['Value'][:,5]  #Using adjusted close price
      if S[s.upper()]['Dates'][0]>=DayOne:
          DayOne=S[s.upper()]['Dates'][0]

    print DayOne
  
    Start_Year,Start_Month,Start_Day=int(DayOne.split('-')[0]),int(DayOne.split('-')[1]),int(DayOne.split('-')[2])
    End_Year,End_Month,End_Day=int(DayEnd.split('-')[0]),int(DayEnd.split('-')[1]),int(DayEnd.split('-')[2])
    
    ReBalanceDates=GetRebalanceDates(DayOne,DayEnd,RebalancePeriod)
    ReBalanceDates=array(ReBalanceDates)
    print ReBalanceDates
    
    Old=DayOne;
    for i in range(len(ReBalanceDates)+1):
      if i<len(ReBalanceDates):
        New=ReBalanceDates[i]
      if i==len(ReBalanceDates):
        New=DayEnd
      PVal=0
  
      for s in self.Portfolio:
    
        ind_old=where(S[s.upper()]['Dates']==Old)[0]
    
        while((len(ind_old)==0)):
          print 'Mod Old:',Old, s.upper()
          Data=Old.split('-')
          if int(Data[2])<10:
            Old='-'.join([Data[0],Data[1],'%s%s'%('0',str(int(Data[2])-1))])
          else:
            Old='-'.join([Data[0],Data[1],str(int(Data[2])-1)])
          ind_old=where(S[s.upper()]['Dates']==Old)[0]
    
        ind_new=where(S[s.upper()]['Dates']==New)[0]
    
        Data=[1,1,1]
        while ((len(ind_new))==0 or (Data[2]==0)):
          print 'Mod New:',New, s.upper()
          Data=New.split('-')
          if int(Data[2])<10:
            Day=int(Data[2])-1
            Month=int(Data[1])
            if Day==0:
              Day=30
              Month=Month-1
              if Month==0:
                Month=12
                Year=str(int(Data[0])-1)
              else:
                Year=str(Data[0])
              if Month<10:
                Monthstr='0%d'%Month
              else:
                Monthstr='%d'%Month
            New='-'.join([Year,Monthstr,'%s%s'%('0',str(int(Day)-1))])
          else:
            New='-'.join([Data[0],Data[1],str(int(Data[2])-1)])
          ind_new=where(S[s.upper()]['Dates']==New)[0]
    
    
        if len(ind_new)>0 and len(ind_old)>0:
          if ind_new[0]>ind_old[0]:
            Ref_Price=(S[s.upper()]['Value'][ind_old[0],5])
            Ref_Stocks=(Weight[s.upper()]*Investment/Ref_Price)
            print Old,':',New,':',ind_old[0],':',ind_new[0]
            S[s.upper()]['DayOneIndex']=ind_old[0]
            S[s.upper()]['Portfolio_NumStocks'][ind_old[0]:ind_new[0]]=Ref_Stocks
            S[s.upper()]['Portfolio_Value']=S[s.upper()]['Portfolio_NumStocks']*S[s.upper()]['Value'][:,5]
            print Investment,Ref_Stocks,Ref_Price, S[s.upper()]['Portfolio_Value'][ind_new[0]-1], s.upper()
            PVal=PVal+S[s.upper()]['Portfolio_Value'][ind_new[0]-1]
        
      Old=New
      Investment=PVal
    
    return S

if __name__=='__main__':
  #Example: run HistoricalDataAnalysis.py --file 'QComRoth.txt' --start-date 1,1,2015 --end-date 12,21,2015
  parser = optparse.OptionParser()
  # Data options
  parser.add_option('--file',help='Porfolio File', dest='portfolio_file', type=str, default='QComRoth.txt')
  parser.add_option('--save-path',help='Path to save Portfolio data', type='str',dest='save_path', default='./HistoricalData')
  parser.add_option('--start-date',help='Start date for collecting Portfolio Data', dest='start_date', action='append')
  parser.add_option('--end-date',help='End date for collecting Portfolio Data', dest='end_date', action='append')
  
  (opts, args) = parser.parse_args()
  sd=tuple(map(int,opts.start_date[0].split(',')))
  ed=tuple(map(int,opts.end_date[0].split(',')))
  
  #Get time string
  t_s=time.localtime()
  t_str=str(t_s.tm_year)+'_'+str(t_s.tm_mon)+'_'+str(t_s.tm_mday)
  #Save_Path_Folder
  SubFolder=opts.portfolio_file.split('.')[0]+'_'+t_str
  save_path='./%s/%s'%(opts.save_path,SubFolder)
  if not os.path.isdir(save_path):
    os.mkdir(save_path)
    
  P=HistoricalDataAnalysis(opts.portfolio_file,save_path,sd,ed)
  P.ObtainPortfolioData()
  S=P.ConvertPortfolioData()
  SMean,IndexNorm,Stats=P.PlotNormalizedPortfolio_StockData(S)
  o=open('%s/Data.pkl'%save_path,'wb')
  pickle.dump([SMean,IndexNorm,Stats],o)
  o.close()
### Play Code
'''
datestr=[]
datestr.append(str(datelist[0]))
if datelist[1]<10:
  datestr.append('%s%s'%(0,str(datelist[1])))
else:
  datestr.append(str(datelist[1]))
if datelist[2]<10:
    datestr.append('%s%s'%(0,str(datelist[2])))
else:
  datestr.append(str(datelist[2]))

cY=Start_Year
cM=Start_Month
while(cY<End_Year):
    rM=mod(cM+r,12)
    if rM<cM+r:
        cY=cY+1
    if cY<End_Year:
      print (cY,rM,Start_Day-1)
    if cY==End_Year and rM<=End_Month:
      print (cY,rM,Start_Day-1)
    cM=rM
    if (cY==End_Year):
      while (cM<End_Month):
        rM=mod(cM+r,12)
        if rM<End_Month:
          print (cY,rM,Start_Day-1)
        cM=rM
    
''' 
