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
import sys,getopt,os
import numpy as np
from numpy import *
import matplotlib
matplotlib.use('TKAgg')
import pylab as py
from operator import itemgetter
import datetime
import pandas as pd
import pandas_datareader as web
import scipy as sp
import scipy.optimize as scopt
import seaborn as sns
from helpermodules import GetStockPriceAndMarketCap,InitiateSession,CloseSession,GetHistoricalData,GetDataForPortfolio
from xvfbwrapper import Xvfb
from time import mktime
from selenium import webdriver
from gsheets import Sheets
import pandas as pd

HOME=os.environ['HOME']
file_path = os.path.dirname(os.path.realpath('__file__'))
#print file_path
#sys.exit(0)

def WaitForDownload(datafile):
    bul=0
    while not bul:
        if os.path.isfile(datafile):
            bul=1
        else:
            bul=0

def download_ticker(ticker,sd,ed):
    vdisplay=Xvfb()
    sd_unix=mktime(sd.timetuple())
    ed_unix=mktime(ed.timetuple())
    URL='https://finance.yahoo.com/quote/%s/history?period1=%d&period2=%d&interval=1d&filter=history&frequency=1d'%(ticker.upper(),sd_unix,ed_unix)
    print URL
    #vdisplay.start() ## uncomment to prevent browser window from opening 
    driver=webdriver.Chrome()
    driver.get(URL)
    driver.implicitly_wait(100)
    elem=driver.find_element_by_link_text('Download Data')
    elem.click()
    datafile='%s/Downloads/%s.csv'%(HOME,ticker.upper())
    WaitForDownload(datafile)
    mv_cmd='mv %s/Downloads/%s.csv %s/Data'%(HOME,ticker.upper(),file_path)
    os.system(mv_cmd)
    #vdisplay.stop() ## uncomment to prevent browser window from opening 
    driver.close()    
    
## Define python module to read ticker data from csv file downloaed from Yahoo-finance
def get_data_from_csv(filename,ticker):
  data=pd.read_csv(filename)
  data.set_index('Date',inplace=True)
  data.rename(columns={'Close':ticker},inplace=True)
  data=data[ticker]
  return data

def get_relevant_date():
  #Function to get last possible valid day in any given year which is not in the future
  current_year=datetime.datetime.now().year
  current_month=datetime.datetime.now().month
  current_day=datetime.datetime.now().day

  if current_day==1:
    if current_month==3:
      current_day=28
      current_month=2
    elif current_month==1:
      current_day=30
      current_year=current_year-1
      current_month=12
    else:
      current_day=30
      current_month=current_month-1
  else:
    current_day=current_day-1

  return (current_year,current_month,current_day)

### Pandas Functions
def get_historical_closes(tickers,start,end,ref='google'):
    ## start is a datetime.datetime variable
    ## end is datetime.datetime variable
    ## example: start=datetime.datetime(2001,12,1)

    def get_data(ticker):
      try:
        return web.data.DataReader(ticker, ref, start, end)
      except:
        print 'error in data download for ticker: %s'%ticker
    datas = map(get_data, tickers) ## this is a nice concept.. map applies function "data" to the items of sequence "ticker"
    all_data=pd.concat(datas, keys=tickers, names=['Ticker','Date'])
    all_data_reset = all_data[['Close']].reset_index()
    # pivot = all_data_reset.pivot('Date', 'Ticker','Close')
    return all_data_reset

def calc_daily_returns(closes):
    #Log of Pt/Pt-1
    return log(closes/closes.shift(1))

def calc_month_year_returns(daily_returns):
    per=daily_returns.index.to_period("M")
    return daily_returns.groupby(per).sum()

def calc_quaterly_returns(daily_returns):
    per = daily_returns.index.to_period("Q")
    return daily_returns.groupby(per).sum()

def calc_annual_returns(daily_returns):
    grouped = exp(daily_returns.groupby(lambda date: date.year).sum())-1
    return grouped

def calc_portfolio_var(returns,weights=None):
    if weights is None:
        weights=ones(returns.columns.size)/returns.columns.size ### returns is of type Data-Frame
    var=returns.cov().dot(weights).dot(weights)
    return var
    
def sharpe_ratio(returns, weights = None, risk_free_rate = 0.015):
    n = returns.columns.size
    if weights is None: weights = ones(n)/n
    var = calc_portfolio_var(returns, weights) 
    means = returns.mean()
    return (means.dot(weights) - risk_free_rate)/var**0.5

def negative_sharpe_ratio_n_minus_1_stock(weights,returns,risk_free_rate):
    """
    Given n-1 weights, return a negative sharpe ratio
    """
    weights2 = sp.append(weights, 1-sum(weights))
    #print weights2
    return -sharpe_ratio(returns, weights2, risk_free_rate)

def optimize_portfolio(returns, risk_free_rate):
    w0 = ones(returns.columns.size-1,dtype=float) * 1.0 / returns.columns.size
    #w0=np.array([.2,0,0])
    opt= scopt.minimize(negative_sharpe_ratio_n_minus_1_stock,w0, args=(returns, risk_free_rate),bounds=[(0,1)]*len(w0))
    w1=opt.x
    final_w = sp.append(w1, 1 - sum(w1))
    final_sharpe = sharpe_ratio(returns, final_w, risk_free_rate)
    print final_sharpe
    return (final_w, final_sharpe)

def bond_constraint(W,P):
  indx=[i for i,s in enumerate(P.SecurityType) if 'Bond' in s]
  return np.sum(W[indx])-0.3

def neg_sharp(weights,returns,risk_free_rate=0.01):
  return -np.log(sharpe_ratio(returns,weights,risk_free_rate))*1e-2
 
def direct_portfolio_optimize(returns,securitytype,risk_free_rate=0.01):
  w0 = np.ones(returns.columns.size,dtype=float) * 1.0 / returns.columns.size
  bounds = [(0,1) for i in np.arange(returns.columns.size)]
  c1={'type': 'eq','fun': lambda W: np.sum(W) - 1}
  c2={'type': 'ineq','fun':lambda W,ST: np.sum(W[[i for i,s in enumerate(ST) if 'bond' or 'Bond' in s]])- .3,'args':(securitytype,)}
  c3={'type': 'ineq','fun':lambda W,ST: -np.sum(W[[i for i,s in enumerate(ST) if 'stock' or 'Stock' in s]])+ .1,'args':(securitytype,)}
  constraints = (c1)
  constraints = (c1,c2)
  constraints = (c1,c2,c3)
  #print securitytype
  #constraints=({'type': 'ineq','fun':lambda W,ST: np.sum(W[[i for i,s in enumerate(ST) if 'bond' or 'Bond' in s]])- .3,'args':(securitytype,)})
  #constraints=({'type': 'eq','fun': lambda W: np.sum(W) - 1})
  #results = scopt.minimize(neg_sharp, w0, (returns,risk_free_rate),method='SLSQP',constraints=constraints,bounds=bounds,options={'maxiter': 1000})
  msg=''
  tol=0.000001
  while 'successfully' not in msg:
    print tol
    results = scopt.minimize(neg_sharp, w0, (returns,risk_free_rate),method='SLSQP',constraints=constraints,bounds=bounds,tol=tol,options={'maxiter':1000,'disp': False})
    msg=results.message
    tol=tol*5
  print results.message
  return results.x


class Portfolio(object):
  def __init__(self,portfolio='',risk_free_rate=0.015):
    '''
    Portfolio Object Class to Read, Rebalance and Analyze Stock ETF portfolio
    :param portfolio: Portfolio file name locally stored as .txt file
    :param risk_free_rate:
    :param gsheet: Boolean indicator if True will try to download and read portfolio from google drive
    '''
    self.Portfolio=portfolio
    self.Stocks=None
    self.Weight=None
    self.risk_free_rate=risk_free_rate
    self.CurrentAlloc=None
    self.SecurityType=None
    self.Index='.INX'

  def ReadPortfolio(self,client_secret=None,sheet_url=None):
    '''
    Read Portfolio data
    :return: Portfolio object with portfolio information
    '''
    self.client_secret = client_secret
    self.sheet_url = sheet_url

    Stocks=[]; Weight=[]; CurrentAlloc=[];SecurityType=[]
    if not os.path.exists(self.Portfolio):
      print "Portfolio File %s does not exist"%self.Portfolio
      print "trying to access portfolio from google drive using client information"
      if os.path.isfile(self.client_secret):
        tmp_folder='%s/Temp'%HOME
        if not os.path.isdir(tmp_folder):
          os.mkdir(tmp_folder)
        tmp_jsonfile='%s/storage.json'%tmp_folder
        sheets=Sheets.from_files(self.client_secret,tmp_jsonfile)
        s=sheets.get(self.sheet_url)
        if s is not None:
          for i in range(s[0].nrows-1):
            Stocks.append(str(s[0].at(row=i+1,col=0)))
            Weight.append(s[0].at(row=i+1,col=1))
            CurrentAlloc.append(s[0].at(row=i+1,col=2))
            SecurityType.append(s[0].at(row=i+1,col=3))
      else:
        print "Client Secret Information not correct, exiting....."
        sys.exit(0)
    else:
      o=open(self.Portfolio)
      Data=o.readlines()
      o.close()
      for d in Data:
        Stocks.append(d.split(' ')[0])
        Weight.append(float(d.split(' ')[1]))
        CurrentAlloc.append(int(d.split(' ')[2]))
        SecurityType.append(d.split(' ')[3])
    self.Stocks=Stocks
    self.Weight=Weight
    self.CurrentAlloc=CurrentAlloc
    self.SecurityType=SecurityType
    
  def PrintPortfolio(self):
    print "Stock Weight Percent_Alloc Type "
    for i in range(len(self.Stocks)):
      print '%s'%self.Stocks[i]+(6-len(self.Stocks[i]))*' '+'%.1f'%self.Weight[i]+6*' '+'%d'%self.CurrentAlloc[i]+8*' '+self.SecurityType[i]
      #print '%s    %.1f   %d  %s'%(self.Stocks[i],self.Weight[i],self.CurrentAlloc[i],self.SecurityType[i])
        
  def Rebalance(self,Fundsize):  
    session=InitiateSession()
    period='d'
    for i in range(len(self.Stocks)):
      session=InitiateSession()
      stockprice,marketcap=GetStockPriceAndMarketCap(session,self.Stocks[i])
      CloseSession(session)
      #stockprice=GetDataFromGoogle(session,self.Stocks[i],self.SecurityType[i])
      #stockprice=GetHistoricalData(session,self.Stocks[i],start_date,end_date,period)[0][-1]
      NumStock=floor((Fundsize*self.Weight[i])/100./stockprice)
      if self.CurrentAlloc[i]==NumStock:
        print "Do nothing on %s; Stock Price is %f; Current Weigh is %.2f"%(self.Stocks[i],stockprice,100.*stockprice*self.CurrentAlloc[i]/Fundsize)
      if self.CurrentAlloc[i]<NumStock:
        print "Purchase %d of %s; Stock Price is %f; Current Weight is %.2f"%(int(NumStock-self.CurrentAlloc[i]),self.Stocks[i],stockprice,100.*stockprice*self.CurrentAlloc[i]/Fundsize)
      if self.CurrentAlloc[i]>NumStock:
        print "Sell %d of %s; Stock Price is %f; Current Weight is %.2f"%(int(-NumStock+self.CurrentAlloc[i]),self.Stocks[i],stockprice,100.*stockprice*self.CurrentAlloc[i]/Fundsize)

  
  def AnalyzePortfolio(self,start_date,end_date):
    session=InitiateSession()
    self.ReadPortfolio()
    #sys.exit(0)
    stocksData=GetDataForPortfolio(session,self.Stocks,start_date,end_date,self.SecurityType)
    indexData=array(GetHistoricalData(session,self.Index,start_date,end_date,'index'))
    
    ## Match the dimensionality of Data (inconsistencies when data for mutual fund is downloaded )
    minlen=100000;
    stocksDatalen=[]
    for k in stocksData.keys():
      if len(stocksData[k])<minlen:
        minlen=len(stocksData[k])
        ref_key=k
      stocksDatalen.append(len(stocksData[k]))
    ref_data=array(stocksData[ref_key])
    reflen=minlen

    stocksArray=[]
    for k in stocksData.keys():
      tmp_data=array(stocksData[k])
      if k!=ref_key:
        ind=[i for i, item in enumerate(tmp_data[:,0]) if item in ref_data[:,0]]
        tmp_data=tmp_data[ind,:]
        if len(ind)<minlen:
          reflen=len(ind)
          for i in range(minlen-len(ind)):
            tmp_data=vstack([tmp_data,array([0,0])])
        stocksArray.append(tmp_data[:,1])
      else:
        stocksArray.append(ref_data[:,1])
    stocksArray=array(stocksArray).T[0:reflen,:]
  
  ## Adjust indexData (for inconsistensies resulting from missing mutual fund data)
    if len(indexData)!=minlen:
        ind=[i for i, item in enumerate(indexData[:,0]) if item in ref_data[:,0]]
        indexData=indexData[ind]
    indexData=indexData[0:reflen,1]

    #Correlation analysis
    sortedsecurity=array(sorted(enumerate(self.SecurityType),key=itemgetter(1)))
    sortedindex=map(int,sortedsecurity[:,0])
    sortedsecuritylist=array(self.Stocks)[sortedindex]
    sortedsecuritylist=[d.upper() for d in sortedsecuritylist]
    sortedstocksArray=stocksArray[:,sortedindex]
    corr=corrcoef(sortedstocksArray.T)
    fig=py.figure();
    ax=fig.add_subplot(111)
    pax=ax.pcolor(corr,cmap=py.get_cmap('spectral'),vmin=-1, vmax=1)
    py.xticks(range(len(corr)),sortedsecuritylist)
    py.yticks(range(len(corr)),sortedsecuritylist)
    labels = ax.get_xticklabels() 
    py.tight_layout()
    for label in labels: 
        label.set_rotation(90)
    cbar = fig.colorbar(pax,ticks=[-1,-.5,-.25,0,.25,.5,1])
    #cbar.ax.set_yticklabels([str(-1),str(-.5),str(-.25), '0',str(.25),str(.5),str(1)])
    corrFileName='%s_Correlation.png'%self.Portfolio.split('.')[0]
    py.savefig(corrFileName)
    
    weight=array(self.Weight)
    weight.shape=(len(self.Stocks),1)
    w=tile(weight,(shape(stocksArray)[0],))
    stocksPriceData=100*sum(stocksArray*w.T,1)/(sum(weight))

    #Normalized Data
    indexNorm=array(indexData)/array(indexData[0])
    stocksPriceDataNorm=stocksPriceData/stocksPriceData[0]
    
    #Plot Data and Save Output
    py.figure();py.plot(indexNorm)
    py.hold('on');py.plot(stocksPriceDataNorm)
    py.xticks([0,len(stocksPriceDataNorm)],[start_date,end_date],rotation=45)
    py.legend(['S&P','%s'%self.Portfolio],loc='best')
    py.tight_layout() 
    FileName='%s_timeseries.png'%self.Portfolio.split('.')[0]
    py.savefig(FileName)
    
    return stocksArray
    CloseSession(session)
  
  def getData(self,start_date,end_date):
    stocks=[s.replace('.','-') for s in self.Stocks]
    #sd=str(start_date[0])+'-'+str(start_date[1])+'-'+str(start_date[2])
    #ed=str(end_date[0])+'-'+str(end_date[1])+'-'+str(end_date[2])
    sd=datetime.datetime(start_date[0],start_date[1],start_date[2])
    ed=datetime.datetime(end_date[0],end_date[1],end_date[2])
    closes=get_historical_closes(stocks,sd,ed)
    closes.fillna(0,inplace=True) ### replace NaN values with 0s
    return closes

  def Optimize(self,start_date,end_date,plot_bool=False):
    closes=self.getData(start_date,end_date) ## result is pandas data frame: Date x Tickers
    daily_returns=calc_daily_returns(closes)
    daily_returns.fillna(0,inplace=True)
    annual_returns=calc_annual_returns(daily_returns)
    annual_returns=annual_returns.replace(inf, 0)
    w=direct_portfolio_optimize(annual_returns,self.SecurityType, self.risk_free_rate)
    return w

  def Analyze(self,start_date,end_date,weights=None,plot_bool=False):
    ## First get index data
    sd=str(start_date[0])+'-'+str(start_date[1])+'-'+str(start_date[2])
    ed=str(end_date[0])+'-'+str(end_date[1])+'-'+str(end_date[2])
    sp_500 = web.DataReader("^GSPC", "yahoo", sd, ed)
    sp_500_dpc = sp_500['Adj Close'].pct_change().fillna(0)

    if weights is None:
      weights=ones(len(self.Stocks))/len(self.Stocks)
    closes=self.getData(start_date,end_date)
    portfolio_close=closes.dot(weights)
    combined_data=pd.concat([closes,portfolio_close],axis=1)
    combined_data.rename(columns={0:'Portfolio'},inplace=True) ### Add the optimized portfolio data to individual ticker data
    daily_pct_change=combined_data.pct_change()
    dpc_all = pd.concat([sp_500_dpc, daily_pct_change], axis=1) ## concate sp500 pct change data with other stocks
    dpc_all.rename(columns={'Adj Close': 'SP500'}, inplace=True)
    cdr_all = (1 + dpc_all).cumprod()
    if plot_bool:
      _=cdr_all.plot()
      py.figure();
      sns.heatmap(combined_data.corr())
    return cdr_all


def next_weekday(d,weekday):
  days_ahead=weekday-d.weekday()
  if days_ahead<=0:
    days_ahead+=7
  return d+datetime.timedelta(days_ahead)


def Backtest_Annual_Rebalance(stocklist,weight,sd,ed,initial=10000,stk_data=None):
  #sd and ed of type datetime.datetime
  ## generate weight dictionary
  def get_total(stk,wdict,stocklist,initial):
    stk['Total']=0
    for u in stocklist:
      var='N_%s'%u
      stk[var]=np.floor(initial*wdict[u]/stk[u].iloc[0])
      stk['Total']+=stk[var]*stk[u]
    stk['Cash']=initial-stk.Total.iloc[0]
    return stk

  wdict={}
  for i in range(len(stocklist)):
    wdict[stocklist[i]]=weight[i]

  if not type(stocklist)==list:
    stocklist=list(stocklist)
  #print stocklist,sd,ed

  if stk_data is None: 
    stk=get_historical_closes(stocklist,sd,ed)
  else:
    cp_stk_data=stk_data.copy()
    stk=cp_stk_data[cp_stk_data.index>=sd]
    stk=stk[stk.index<ed]
  stk=get_total(stk,wdict,stocklist,initial)
  Years=range(sd.year+1,ed.year+1)
  for yr in Years:
    start_day=next_weekday(datetime.datetime(yr,1,1),2)
    start_day_var=datetime.datetime(start_day.year,start_day.month,start_day.day)
    
    stk_l=stk[stk.index<start_day_var]
    stk_u=stk[stk.index>=start_day_var]
    vals=stk[stk.index==start_day_var]
    
    ini=stk_l['Total'][-1]+stk_l['Cash'][-1]
    stk_u=get_total(stk_u,wdict,stocklist,ini)
    stk=stk_l.append(stk_u)
  
  return stk

def main(argv):
  inputfile=''
  g_optimize = 0
  g_rebalance = 0
  g_analyze=0
  g_period='w'
  start_date=(2013,7,1)
  end_date=(2014,6,30)
  fundsize=20000.0
  try:
    opts, args = getopt.getopt(argv,"hf:orapdes",["file=","optimize","rebalance","analyze","period=","start-date=","end-date=","fundsize="])
  except getopt.GetoptError:
    print '''PortfolioBuilder.py 
      -f <inputfile>
      --r (rebalance) [Default: 0] 
      --a (analyze)[Default: 0] 
      --p (period)[Default: 'w']
      --d (start-date) [Default: (2013,7,1)]
      --e (end-date) [Default: (2014,6,30)]
      --s (fundsize) [Default: 20000.]'''
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
      print '''PortfolioBuilder.py 
         -f <inputfile>
         --r (rebalance) [Default: 0] 
         --a (analyze)[Default: 0] 
         --p (period)[Default: 'w']
         --d (start-date) [Default: (2013,7,1)]
         --e (end-date) [Default: (2014,6,30)]
         --s (fundsize) [Default: 20000.]'''
      sys.exit()
    elif opt in ("-f", "--file"):
      inputfile = arg
    elif opt in ("-r", "--rebalance"):
      g_rebalance = 1
    elif opt in ("-a", "--analyze"):
      g_analyze = 1
    elif opt in ("-p", "--period"):
      period = arg
    elif opt in ("-d", "--start-date"):
      start_date = eval(arg)
    elif opt in ("-e", "--end-date"):
      end_date = tuple(eval(arg))
    elif opt in ("-s", "--fundsize"):
      fundsize = eval(arg)
  print 'Input file is %s'%(inputfile)
  print 'Boolean Flags are optimize=%d, rebalance=%d, analyze=%d'%(g_optimize,g_rebalance,g_analyze)
  print 'Period is: %s'%g_period
  print "Dates are:",start_date,end_date
  print "Fund size:",fundsize
  return inputfile,g_optimize,g_rebalance,g_analyze,g_period,start_date,end_date,fundsize
   
if __name__=='__main__':
  inputfile,g_optimize,g_rebalance,g_analyze,g_period,start_date,end_date,fundsize=main(sys.argv[1:])
  P=Portfolio(inputfile)
  P.ReadPortfolio()
  P.PrintPortfolio()

  if g_rebalance:
    P.Rebalance(fundsize,start_date,end_date)
  if g_analyze:
    SA=P.AnalyzePortfolio(start_date,end_date,g_period)
  
