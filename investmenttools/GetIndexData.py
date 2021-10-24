import re
import numpy as np
from datetime import datetime
import optparse
import pickle
import pandas as pd
import sys,os
import glob
import urllib2
import time
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from xvfbwrapper import Xvfb
import quandl
HOME=os.environ['HOME']
quandl.ApiConfig.api_key = "hV1kr3pQW8ofRGZZjy6s"
ALPHA_VANTAGE_API="Z9G0WY4M4F0X8QSB"

vdisplay=Xvfb()
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('%s/..'%file_path)
from investmenttools import Compute_Fundamentals as CF
reload(CF)

def WaitForDownload(datafile):
    bul=0
    while not bul:
        if os.path.isfile(datafile):
            bul=1
        else:
            bul=0

def augment_Timeseries(Timeseries,portposfile):
  '''
  Function that augments the Timeseries data with portfolio position data at the end of the year
  :param Timeseries:
  :param portposfile:
  :return:
  '''
  print portposfile
  portpos=pd.read_excel(portposfile)
  portpos.set_index('Ticker:Symbol', inplace=True, drop=True)

  for stk in Timeseries.columns:
    Nstk = 'num_%s' % stk
    Timeseries[Nstk] = np.zeros((Timeseries.shape[0],))
    cstk = 'cost_%s' % stk
    Timeseries[cstk] = np.zeros((Timeseries.shape[0],))
    if stk in portpos.index:
      Timeseries[Nstk]=np.ones((Timeseries.shape[0],))*portpos.loc[stk]['Shares']
      Timeseries[cstk]=np.ones((Timeseries.shape[0],))*portpos.loc[stk]['Cost Basis']

  return Timeseries

def insert_transactions(Timeseries,data):
  '''

  :param Timeseries:
  :param data:
  :return:
  '''
  for stk in Timeseries.columns:
    if stk in data.columns:
      for i in range(len(data[stk])):
        if not pd.isnull(data[stk].iloc[i]):

          dt=data[stk].iloc[i].split(';')[0].replace(',','-')
          nstk=int(data[stk].iloc[i].split(';')[1])
          pricestk=float(data[stk].iloc[i].split(';')[2])
          indx=np.where(Timeseries.index==dt)[0][0]
          new_stock_count=nstk+Timeseries['num_%s'%stk].iloc[indx]
          if new_stock_count!=0:
            new_costbasis=(nstk*pricestk+Timeseries['cost_%s'%stk].iloc[indx]*Timeseries['num_%s'%stk].iloc[indx])/new_stock_count
          else:
            new_costbasis=0
          if stk=='NWL':
            print (i,dt, stk,dt,nstk,pricestk,Timeseries['cost_%s'%stk].iloc[indx],Timeseries['num_%s'%stk].iloc[indx])
          Timeseries['num_%s'%stk].iloc[indx:]=new_stock_count
          Timeseries['cost_%s' % stk].iloc[indx:] = new_costbasis

  return Timeseries


class Index(object):
  def __init__(self,indx=None,filepath=None,save_dir=None,memo=None,stock=None):
    '''

    :param indx: Index: SP, DOW or RUS
    :param filepath: path to index file
    :param save_dir: path to save directory
    :param memo: string variable to annotate data collection
    :param stock: list of stocks if no index file provided
    '''
    self.Index=indx
    self.File=filepath
    self.List=stock
    self.save_dir=save_dir
    self.memo=memo

    if self.File is not None:
      if os.path.isfile(self.File):
        inpFile=pd.read_csv(self.File)
        self.List=list(inpFile['Ticker Symbol'].dropna())
      else:
        print 'Index File does not exist...'
        sys.exit(0)
    elif stock is not None:
      if type(stock)==str:
        self.List=[]
        self.List.append(stock)
      else:
        self.List=stock
    else:
      print 'Index File does not exist and no stock list provided.. check filepath'
      sys.exit(0)

  def getquandlData(self,start=(2018,1,1),end=(2018,12,31)):
    self.TimeSeries=pd.DataFrame(columns=[])
    if type(start) is not datetime:
      start=datetime(start[0],start[1],start[2])
    if type(end) is not datetime:
      end=datetime(end[0],end[1],end[2])

    for s in self.List:
      print s
      stk=quandl.get("WIKI/" + s, start_date=start, end_date=end)
      self.TimeSeries[s]=stk['Adj. Close']


  def getAlphaVantageData(self,start_date='2018-1-1',vdisplay_bool=True):
    if vdisplay_bool:
      vdisplay.start()
    self.TimeSeries = pd.DataFrame(columns=[])
    driver = webdriver.Chrome()
    count_download=0
    for s in self.List:
      print s
      download_file = '%s/Downloads/daily_adjusted_%s.csv' % (HOME, s)
      if not os.path.isfile(download_file):
        URL='https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=%s&apikey=%s&datatype=csv&outputsize=full'%(s,ALPHA_VANTAGE_API)
        print URL
        if count_download>5:
          time.sleep(62) ## because of restriction from AlphaVantage to download 5 stock data per minute
        driver.get(URL)
        WaitForDownload(download_file)
        count_download+=1

      if os.path.isfile(download_file):
        vals=pd.read_csv(download_file)
        vals.set_index('timestamp',inplace=True,drop=True)
        vals.index=pd.to_datetime(vals.index)
        self.TimeSeries[s]=vals[vals.index>start_date]['adjusted_close'].iloc[::-1]
    driver.close()
    if vdisplay_bool:
      vdisplay.stop()



  def getYahooData(self):
    ## get Data in pandas DataFrame Format
    passon_list=[]
    stk=yp.Share(self.List[0])
    data=stk.data_set
    StockData=pd.DataFrame(index=data.keys())
    for i in range(len(self.List)):
      time.sleep(2)
      bull=False
      while not bull:  
        try:
          stk=yp.Share(self.List[i])
          data=stk.data_set
          StockData[self.List[i]]=pd.DataFrame.from_dict(data,orient='index')
          bull=True
        except urllib2.HTTPError:
          print 'Try Again for: %s'%self.List[i]
          bull=False
    return StockData

  def get_stockrowData(self,type='Growth',vdisplay_bool=True):
    '''
    :param: type: 'Growth' or 'Metrics'
    :param vdisplay_bool
    :return: Populated directory of stock quaterly key metrics
    '''
    def get_data(stk,filename,type='Growth'):
      if sys.platform=='darwin':
        driver=webdriver.Safari()
      else:
        driver = webdriver.Chrome()
      if type=='Metrics':
        URL = 'https://stockrow.com/%s/financials/metrics/quarterly' % stk
        selector = '#root > div > div > section > div > div.main-content > div:nth-child(1) > div:nth-child(2) > section.grid-x.align-center.company-financials > div > div.grid-x.align-center.grid-margin-x.control-buttons > div.cell.medium-7 > a'
      if type=='Growth':
        URL='https://stockrow.com/%s/financials/growth/quarterly' % stk
        selector='#root > div > div > section > div > div.main-content > div:nth-child(1) > div:nth-child(2) > section.grid-x.align-center.company-financials > div > div.grid-x.align-center.grid-margin-x.control-buttons > div.cell.medium-7 > a'
      print URL
      driver.get(URL)
      time.sleep(3)
      driver.find_element_by_css_selector(selector).click()
      datafile = '%s/Downloads/financials.xlsx' % HOME
      WaitForDownload(datafile)
      try:
        data = pd.read_excel(datafile).T
        data.to_csv(filename)
        os.remove(datafile)
      except KeyError as e:
        print 'Datafile for stock:%s is empty'%stk
        os.remove(datafile)
      driver.close()

    if vdisplay_bool:
      vdisplay.start()
    if not os.path.isdir(self.save_dir):
      os.mkdir(self.save_dir)
    subdir = '%s/%s' % (self.save_dir, self.memo)
    if not os.path.isdir(subdir):
      os.mkdir(subdir)
    print subdir
    for stk in self.List:
      filename = '%s/%s_%s.csv' % (subdir, stk,type)
      if not os.path.isfile(filename):
        try:
          get_data(stk, filename,type)
        except NoSuchElementException:
          print 'Quaterly data for %s not found'%stk
          del_files=glob.glob('%s/Downloads/financials*')
          if len(del_files)!=0:
            for f in del_files:
              os.remove(f)
          continue
    if vdisplay_bool:
      vdisplay.stop()

  def get_MorningstarFinancials(self,vdisplay_bool=True):
    if vdisplay_bool:
      vdisplay.start()
    for stk in self.List:
      CF.MorningStarFinancialsData(stk,downloaddir='%s/Downloads'%HOME,savedir=self.save_dir)
    if vdisplay_bool:
      vdisplay.stop()

#stk_data[stk_data.columns[(stk_data.loc['PriceBook'].values.astype(float)<1) & (stk_data.loc['PriceSales'].values.astype(float)<1)]]
#print stk_data.loc[['PERatio','PriceSales','PriceBook','PercebtChangeFromYearHigh']][stk_data.columns[np.array([np.nan if u is None else map(float,re.findall('[-+]?\d+.\d+',u))[0] for u in x])<-25]]

if __name__=='__main__':
  parser = optparse.OptionParser()
  # Data options
  parser.add_option('--index',help='[SP, Dow, Rusell]', dest='index', type=str, default='SP')
  parser.add_option("--get-YP-data",action="store_true",dest="get_YP_bool",default=False,help="Get Data from yahoo Finance")
  (opts, args) = parser.parse_args()
  
  if 'SP' in opts.index.upper():
    index_file='%s/Work/Python/Git_Folder/investmenttools/Reference_Data/SP500_ticker.txt'%HOME
  elif 'DOW' in opts.index.upper():
    index_file='%s/Work/Python/Git_Folder/investmenttools/Reference_Data/dow-jones-industrial-average-components.csv'%HOME
  elif 'RUS' in opts.index.upper():
    index_file='%s/Work/Python/Git_Folder/investmenttools/Reference_Data/Russell-1000-Stock-Tickers-List.csv'%HOME
  else:
    print 'index file path not correct.. '
    sys.exit(0)

  indx=Index(opts.index,index_file)
  if opts.get_YP_bool:
    stk_data=indx.getYahooData()
    data_folder='%s/Work/DataSets/IndexData'%HOME
    ts=datetime.now()
    if os.path.isdir(data_folder):
      data_file='%s/%s_YF_%d_%d_%d_%d_%d.pkl'%(data_folder,opts.index,ts.year,ts.month,ts.day,ts.hour,ts.minute)
      o=open(data_file,'wb')
      pickle.dump(stk_data,o)
      o.close()
    else:
      print 'Path to data dump does not exist..'
      sys.exit(0)

    x=stk_data.loc['PercebtChangeFromYearHigh'].values
    print stk_data.loc[['PERatio','PriceSales','PriceBook','PercebtChangeFromYearHigh']][stk_data.columns[np.array([np.nan if u is None else map(float,re.findall('[-+]?\d+.\d+',u))[0] for u in x])<-25]]

  #stk_data# .loc[['LastTradePriceOnly','DividendYield','PERatio','PriceBook','PercebtChangeFromYearHigh','PriceSales']][stk_data.columns[(stk_data.loc['LastTradePriceOnly'].astype('float32')>50) & (stk_data.loc['DividendYield'].astype('float32')>4)]].T.dropna()
  #URL='https://stockrow.com/<TickerName>/financials/metrics/quarterly'
  #from selenium import webdriver
  #driver=webdriver.Chrome()
  #selector= '#root > div > div > section > div > div.main-content > div:nth-child(1) > div:nth-child(2) > section.grid-x.align-center.company-financials > div > div.grid-x.align-center.grid-margin-x.control-buttons > div.cell.medium-7 > a'
  #driver.find_element_by_css_selector(selector).click()