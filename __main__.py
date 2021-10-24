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

## Command to compute portfolio rebalance
#python __main__.py -P -r --portfolio-file <full path to portfolio file> --portfolio-size 69260
from datetime import date,datetime
import sys,os
import optparse
import pickle
import numpy as np
HOME=os.environ['HOME']
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)
#To see the available options run:
#$ python investmenttools -h

def get_index_file(index=None):
  if index is None:
    index='random'
  if index.upper()=='SP':
    mf_file='%s/Reference_Data/SP500_ticker.txt'%file_path
  elif index.upper()=='DOW':
    mf_file='%s/Reference_Data/dow-jones-industrial-average-components.csv'%file_path
  elif 'RUS' in index.upper():
    mf_file='%s/Reference_Data/Russell-1000-Stock-Tickers-List.csv'%file_path
  else:
    mf_file=None
  return mf_file

if __name__=='__main__':
  parser = optparse.OptionParser()
  parser.add_option('--index-yf',help='Get Index Data from Yahoo Finance Choice: [Dow,SP]', dest='index_yf', type=str, default=None)
  parser.add_option('--index-mf',help='Get Fundamentals for Index Data Choice: [Dow,SP]', dest='index_mf', type=str, default=None)
  parser.add_option("-P", "--portfolio",action="store_true",dest="port_build",default=False,help="Run Portfolio Builder")
  parser.add_option('--portfolio-file',help='Portfolio File', dest='port_file', type=str, default='./test_portfolio.txt')
  parser.add_option("-r", "--portfolio-rebalance",action="store_true",dest="port_rebalance",default=False,help="Perform Portfolio Rebalance")
  parser.add_option("-a", "--portfolio-analyze",action="store_true",dest="port_anal",default=False,help="Perform Portfolio Analysis")
  parser.add_option('--start-date',help='Start date: Year Month Day', type=int, dest='start_date',nargs=3)
  parser.add_option('--end-date',help='End date: Year Month Day', type=int, dest='end_date',nargs=3)
  parser.add_option('--portfolio-size',help='Value of Portfolio', dest='port_size', type=int, default=10000)
  parser.add_option("-R", "--compile-stock-list",action="store_true",dest="compile_stocks",default=False,help="Shortlist Stocks for Analysis")
  parser.add_option('--login',help='Login Name', dest='login', type=str,default=None)
  parser.add_option('--password',help='Password', dest='password', type=str,default=None)
  parser.add_option('--marketcap',help='Market Cap in Millions', dest='marketcap', type=int,default=100)
  parser.add_option('--temp-dir',help='Temp Download Dir', dest='temp_dir', type=str, default='%s/Downloads'%HOME)
  parser.add_option('--data-dir',help='Directory where data is saved', dest='data_dir', type=str, default='')
  parser.add_option('--profile-path',help='Firefox Profile path', dest='profile_path', type=str, default='')
  parser.add_option("-i", "--intrisic-value",action="store_true",dest="intrinsic_value",default=False,help="Perform Intrinsic Value Analysis")
  parser.add_option('--morningstar-filename',help='Folder Name of Morning Star Data', dest='msfile', type=str, default=None)
  parser.add_option('--index-filename',help='Folder Name of Index Data', dest='indexfile', type=str, default=None)
  parser.add_option('--expected-return',help='Expected Return for Intrinsic Value Analysis', dest='expected_return', type=float,default=10)
  parser.add_option('--cagr',help='Average CAGR for Intrinsic Value Analysis', dest='cagr', type=float,default=8)
  parser.add_option('--dollar-available',help='Numeric value of total dollars for investment', dest='dollar_available', type=float,default=100000.)
  parser.add_option('--stock-file',help='File containing list of stocks in stock-portfolio', dest='stock_file', type=str,default='')
  parser.add_option('--stock',help='Analysis for Individual Stock', dest='stock', type=str, default='')
  parser.add_option('--cagr-method',help='Method for computing CAGR:Min,Avg', dest='cagr_method', type=str, default='Avg')
  parser.add_option("-v", "--virtual-display",action="store_true",dest="virtual_display",default=False,help="Set Virtual Display")
  parser.add_option("-f", "--plot",action="store_true",dest="plot_ivs",default=False,help="Plot")
  
  (opts, args) = parser.parse_args()
  
  import investmenttools
  import investmenttools.GetIndexData as GI
  reload(GI)
  import investmenttools.ReadMagicFormulaList as RF
  reload(RF)
  import investmenttools.Intrinsic_Value as IVS
  reload(IVS)

  mf_file=get_index_file(index=opts.index_mf)
  yf_file=get_index_file(index=opts.index_yf)

  
  if yf_file is not None:
    indx=GI.Index(opts.index_yf,yf_file)
    stk_data=indx.getYahooData()
    data_folder='%s/Work/DataSets/IndexData'%HOME
    ts=datetime.now()
    if os.path.isdir(data_folder):
      data_file='%s/%s_YF_%d_%d_%d_%d_%d.pkl'%(data_folder,opts.index_yf,ts.year,ts.month,ts.day,ts.hour,ts.minute)
      o=open(data_file,'wb')
      pickle.dump(stk_data,o)
      o.close()
    else:
      print 'Path to data dump does not exist..'
      sys.exit(0)

  if mf_file is not None:
    indx=GI.Index(opts.index_mf,mf_file)
    data_folder='%s/Work/DataSets/IndexData'%HOME
    #data_file='%s/%s_MF_%d_%d_%d_%d_%d.pkl'%(data_folder,opts.index,ts.year,ts.month,ts.day,ts.hour,ts.minute)
    MF=RF.ReadMagicFormula(opts.login,opts.password,opts.profile_path)
    Dir=MF.GetMorningStartData(indx.List,opts.index_mf,downloaddir=opts.temp_dir,savedir=data_folder,virtual_display=opts.virtual_display)
    IVS.AppendStockDataFromFolder(Dir,profile_path=opts.profile_path)
  
  ### Run Portfolio Builder Tool for Rebalancing
  if opts.port_build:
    import investmenttools.PortfolioBuilder as PB
    reload(PB)   
    P=PB.Portfolio(opts.port_file)
    P.ReadPortfolio()
    P.PrintPortfolio()
    if opts.port_rebalance:
      P.Rebalance(opts.port_size)
    if opts.port_anal:
      if opts.start_date==None:
        print 'No start date provided'
        sys.exit(1)
      if opts.end_date==None:
        print 'No end date provided'
        sys.exit(1)
      SA=P.AnalyzePortfolio(opts.start_date,opts.end_date)
  
  ## Read Magic Formular StockList Shortlist
  if opts.compile_stocks:
    if opts.login==None and opts.password==None:
      print 'Enter User Login and Password for the MagicFormula website'
      sys.exit(1)
    MF=RF.ReadMagicFormula(opts.login,opts.password,opts.profile_path)
    if opts.marketcap<1000:
      data_type='Small_Cap'
    if opts.marketcap>=1000 and opts.marketcap<5000:
      data_type='Mid_Cap'
    if opts.marketcap>=5000:
      data_type='Large_Cap'
    Stocklist1=MF.GetStocklist(opts.marketcap,virtual_display=opts.virtual_display)
    if opts.marketcap<5000:
      Stocklist2=MF.GetStocklist(opts.marketcap*10,virtual_display=opts.virtual_display)
      Stocklist=list(set(Stocklist1) | set(Stocklist2))
    else:
      Stocklist=Stocklist1
    print Stocklist
    Dir=MF.GetMorningStartData(Stocklist,data_type,downloaddir=opts.temp_dir,savedir=opts.data_dir,virtual_display=opts.virtual_display)
    IVS.AppendStockDataFromFolder(Dir,profile_path=opts.profile_path)
    
  if opts.intrinsic_value:
    import investmenttools.Intrinsic_Value as IVS
    reload(IVS)
    
    if opts.msfile is not None:
      Dir='%s/%s'%(opts.data_dir,opts.msfile)
      stk_dir=opts.msfile

    if opts.indexfile is not None:
      Dir='%s/%s'%(opts.data_dir,opts.indexfile)
      stk_dir=opts.indexfile
    
    if not os.path.isdir(Dir):
      print 'Data directory For %s does not exist'%stk_dir
      sys.exit(1)
    #IVS.IntrinsicValues(Dir=Dir,ER=opts.expected_return,Provided_CAGR=opts.cagr,Dollar_Available=opts.dollar_available,Stock_File=opts.stock_file)
    IVS.IntrinsicValues(Dir=Dir,ER=opts.expected_return,Provided_CAGR=opts.cagr,cagr_method=opts.cagr_method)
  
  if len(opts.stock)!=0: 
    import investmenttools.ReadMagicFormulaList as RF
    reload(RF)
    import investmenttools.Intrinsic_Value as IVS
    reload(IVS)
    MF=RF.ReadMagicFormula(opts.login,opts.password,opts.profile_path)
    #import investmenttools.GetData_MarketWatch as GM
    import investmenttools.helpermodules as GM
    reload(GM)
    
    Dir='%s/%s'%(opts.data_dir,opts.msfile)
    Stock_File='%s/%s_Combined.pkl'%(Dir,opts.stock.upper())
  
    if os.path.isfile(Stock_File):
      Data=IVS.GetStockDataFromPickle(Dir,opts.stock.upper())
      IVS.IntrinsicValues(Data=Data,ER=opts.expected_return,Provided_CAGR=opts.cagr,cagr_method=opts.cagr_method)
    else:
      Stocklist=[opts.stock.upper()]
      data_type='tmp'
      Dir=MF.GetMorningStartData(Stocklist,data_type,downloaddir=opts.temp_dir,savedir=opts.data_dir,virtual_display=opts.virtual_display)
      #Dir=MF.GetMorningStartData(Stocklist,data_type,downloaddir=opts.temp_dir,savedir=opts.data_dir,virtual_display=opts.virtual_display)
      #print Dir
      IVS.AppendStockDataFromFolder(Dir,profile_path=opts.profile_path)    
      Data=IVS.GetStockDataFromPickle(Dir,opts.stock.upper())
      IVS.IntrinsicValues(Dir=Dir,ER=opts.expected_return,Provided_CAGR=opts.cagr,cagr_method=opts.cagr_method)      
      rm_str='rm -rf %s'%Dir
      os.system(rm_str)  
      
    for k in Data:
      if type(Data)!=list and k!='SV' and k!='MC':
        Data[k]=list(Data[k])

    #import pprint
    #pprint.pprint(Data,width=200)
    
    #Get Annual Stock Data Relative to S&P Index
    '''
    session=GM.InitiateSession()
    annual_price_stock,ref_year,ref_month=GM.GetAnnualDataFromGoogle(session,opts.stock)
    annual_price_SP,ref_year,ref_month=GM.GetAnnualDataFromGoogle(session,'SPY',ref_year,ref_month)
    annual_price_ratio=100*np.array(annual_price_stock)/np.array(annual_price_SP)
    #print annual_price_ratio
    '''
    #plot ratios
    if opts.plot_ivs:
      IVS.Plot_Ratios(Data, None)
      #IVS.Plot_Ratios(Data,annual_price_ratio)
      #raw_input('Enter ^C')
