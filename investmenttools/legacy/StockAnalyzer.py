import requests
import sys,getopt
import json
import re
from numpy import *
import pylab as py
import time
from operator import itemgetter
from datetime import date
import dateutil.parser as dparser
from lmfit import minimize, Parameters, Parameter, report_fit
from GetData_MarketWatch import *

class StockAnalyzer(object):
  def __init__(self,stock):
    self.stock=stock
    self.data={}
    
  def GetData(self):
    session=InitiateSession()
    self.Data=GetFinancials(session,self.stock)
    if len(self.Data)>0:
      self.Data=GetFundamentals(self.Data,'Y')
      self.Data=GetIntrinsicValue(self.Data)
    CloseSession(session)
  
  def PrintData(self):
    EY=100*self.Data['Fund']['EPS']/self.Data['SP']
    print '''
    Data for stock %s:
    Current Share Price: %.2f
                Revenue: %s
             Net Income: %s
         Free Cash Flow: %s
   Free Cash Flow Yeild: %s
      Net Profit Margin: %s
       Return on Equity: %s
  Long Term Debt/Equity: %s
     Com. Ann. Grth. Rate: %.3f
     Earnings Per Share: %s
     Earning Yield: %f
               PE Ratio: %.3f
  Dividend Payout Ratio: %.3f
        Intrinsic Value: %s
  
          '''%(stock.upper(),self.Data['SP'],', '.join("%g"%x for x in self.Data['Revenue']),', '.join("%g"%x for x in self.Data['NI']),
          ', '.join("%g"%x for x in self.Data['FCF']),', '.join("%g"%x for x in self.Data['Fund']['FCFY']),
          ', '.join(format(x, ".3f") for x in self.Data['Fund']['NPM']),', '.join(format(x, ".3f") for x in self.Data['Fund']['ROE']),
          ', '.join(format(x, ".3f") for x in self.Data['Fund']['LTDE']),self.Data["Fund"]["CAGR"],
          ', '.join("%.3f"%x for x in self.Data['Fund']['EPS']),EY,self.Data["Fund"]["PE"],self.Data["Fund"]["DPO"],', '.join("%.3f"%x for x in self.Data['Intrinsic']))
    
def main(argv):
  stock='msft'
  try:
    opts, args = getopt.getopt(argv,"hs:",["stock="])
  except getopt.GetoptError:
    print "See StockAnalyzer -h"
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
      print '''StockAnalyzer.py 
               -s (--stock)
            '''
      sys.exit()
    elif opt in ("-s", "--stock"):
      stock = arg
  return stock


if __name__=='__main__':
  stock=main(sys.argv[1:])
  S=StockAnalyzer(stock.upper())
  S.GetData()
  S.PrintData()