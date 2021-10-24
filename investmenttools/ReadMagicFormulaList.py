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
import sys,os,getopt
import json
import re
from numpy import *
import matplotlib.pylab as py
import time
from datetime import date,datetime
import dateutil.parser as dparser
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from pyvirtualdisplay import Display
import csv
import glob
import platform
from xvfbwrapper import Xvfb
#import GetData_MarketWatch as GM
#reload(GM)
HOME=os.environ['HOME']
vdisplay=Xvfb()

##firefox profile set up
def getfirefoxdriver():
  profile = webdriver.FirefoxProfile()
  profile.set_preference('browser.download.folderList', 2)
  profile.set_preference('browser.download.manager.showWhenStarting', False)
  profile.set_preference('browser.download.dir', '%s/Downloads'%HOME)
  profile.set_preference('browser.helperApps.neverAsk.saveToDisk', ('application/vnd.ms-excel'))
  profile.set_preference('browser.helperApps.neverAsk.saveToDisk', 'text/html')
  driver = webdriver.Firefox(profile)
  return driver

def wait_for_load(driver,Val):
  driver.wait = WebDriverWait(driver, 5)
  try:
    box = driver.wait.until(EC.presence_of_element_located((By.ID, "MinimumMarketCap")))
    box.clear()
    box.send_keys(str(Val))
  except TimeoutException:
    print("Button not found")

def WaitForDownload(downloaddir,S):
  if not os.path.isdir(downloaddir):
    print 'Download Dir Does Not Exist'
    print 'Check your browser download path'
    sys.exit(1)
  File1='%s/%s Key Ratios.csv'%(downloaddir,S)
  File2='%s/%s Key Ratios.csv.html'%(downloaddir,S)
  bul=0
  while not(bul):
    if len(glob.glob(File2))>0:
      F1='%s/%s\ Key\ Ratios.csv.html'%(downloaddir,S)
      F2='%s/%s\ Key\ Ratios.csv'%(downloaddir,S)
      rename_cmd='mv %s %s'%(F1,F2)
      #print rename_cmd
      os.system(rename_cmd)
    if len(glob.glob(File1))>0:
      bul=1
      Dir=downloaddir
    else:
      bul=0
  #print Dir
  return Dir
  
class ReadMagicFormula(object):
  def __init__(self,login,passwd,profile_path=''):
    self.URL='http://www.magicformulainvesting.com/Account/LogOn'
    self.login=login
    self.password=passwd
    self.profile_path=profile_path
    
  # def GetData(self,S):
  #   session=GM.InitiateSession()
  #   SData=[]
  #   for s in S:
  #     #print s,"\r"
  #     Data={}
  #     Data=GM.GetFinancials(session,str(s))
  #     if len(Data)>0:
  #       Data=GM.GetFundamentals(Data,'Y')
  #       Data=GM.GetIntrinsicValue(Data)
  #       if len(Data['Fund']['EPS']):
  #         EY=100*Data['Fund']['EPS']/Data['SP']
  #       else:
  #         EY=[0.0]  
  #       SData.append([str(s),float(Data['SP'][0]),float(Data['Intrinsic'][0]),float(EY[0])])  
  #   GM.CloseSession(session)
  #   return array(SData)
  
  
  def GetStocklist(self,Val,virtual_display=False):
    print '**** Get Stock List ******'
    if virtual_display:
      vdisplay.start()
    if 'Linux' in platform.platform():
      #driver = getfirefoxdriver()
      driver=webdriver.Chrome()
    if 'Darwin' in platform.platform():
      driver = webdriver.Safari()
    driver.get(self.URL)
    driver.find_element_by_id("Email").send_keys(self.login)
    driver.find_element_by_id("Password").send_keys(self.password)
    driver.find_element_by_id("login").click()
    wait_for_load(driver,Val)
    driver.find_element_by_name("Select30").submit()
    time.sleep(2)
    txt=driver.page_source
    driver.quit()
    if virtual_display:
      vdisplay.stop()
    Numstr1=[m.start() for m in re.finditer('tr class=""',txt)]
    Numstr2=[m.start() for m in re.finditer('altrow',txt)]
    Numstr=Numstr1+Numstr2
    Numstr=sorted(Numstr)
    S=[]
    for i in range(len(Numstr)):
      S.append(str(txt[Numstr[i]:Numstr[i]+400].split('center">')[1].split('<')[0]))
    Stock=[s.encode('UTF8') for s in S]
    return Stock
  
  def GetMorningStartData(self,S,data_type,downloaddir='%s/Downloads'%HOME,\
    savedir='',virtual_display=False):
      print '***** Get Financial Data from Morningstar *********'
      time_stamp=datetime.now()
      hr_data=time_stamp.time().isoformat().split(':')
      hr_data_int=[int(float(x)) for x in hr_data]
      hr_data_str='_'.join(str(x) for x in hr_data_int)
      
      time_data=[time_stamp.year,time_stamp.month,time_stamp.day]
      time_data_str='_'.join(str(x) for x in time_data)
      time_stampstr=time_data_str+'_'+hr_data_str
      DirStr='%s/MorningStarData_%s_%s'%(savedir,data_type,time_stampstr)
      if not os.path.isdir(DirStr):
        os.mkdir(DirStr)
      if virtual_display:
        vdisplay.start()
      if 'Linux' in platform.platform():
        #driver=getfirefoxdriver()
        driver=webdriver.Chrome()
      if 'Darwin' in platform.platform():
        driver=webdriver.Safari()
      for stock in S:
        '''
        URL='http://financials.morningstar.com/ratios/r.html?t=%s&region=usa&culture=en-US'%stock
        driver.get(URL)
        try:
          driver.find_element_by_xpath('//*[@id="financials"]/div[2]/div/a/div')
        except NoSuchElementException, e:
          continue
        driver.find_element_by_xpath('//*[@id="financials"]/div[2]/div/a/div').click()
        DwnloadDir=WaitForDownload(downloaddir,stock)
        driver.implicitly_wait(2) # seconds
        '''
        print stock
        URL='http://financials.morningstar.com/ajax/exportKR2CSV.html?t=%s'%stock.upper()
        #print URL
        driver.get(URL)
        time.sleep(4)
        DwnloadDir=WaitForDownload(downloaddir,stock)

        if DwnloadDir!=0:
          strmv='mv %s/*.csv %s'%(DwnloadDir,DirStr)
          os.system(strmv)
        else:
          if len(os.listdir('%s'%DirStr))==0:
            strrm='rm -rf %s'%DirStr
            os.system(strrm)
      driver.quit()
      if virtual_display:
        vdisplay.stop()
      return DirStr
      
  def ParseStocklist(self,S):
    SData=self.GetData(S)
    Data=SData[:,1:]
    Data=Data.astype(float)
    ind=where(Data[:,0]<=Data[:,1])[0]
    return SData,ind

  def ReadCSVFiles(self,Dir):
    Data={}
    CSVList=os.listdir(Dir)
    for c in CSVList:
      Ticker= c.split('Key')[0].replace(' ','')
      File='%s/%s'%(Dir,c)
      o=open(File,'rb')
      R=csv.reader(o)
      U=[]
      for r in R:
        U.append(r)
      o.close()
      Data[Ticker]={}
      for i in range(len(U)):
        if len(U[i])>2:
          if '2012' in ''.join(U[i][1:]):
            Data[Ticker][U[i][0]]=array(U[i][1:])
          elif ',' in ''.join(U[i][1:]):
            X=array(U[i][1:])
            X[X=='']='-9999'
            Data[Ticker][U[i][0]]=map(float,[x.replace(',','') for x in X])
          elif '2012' not in ''.join(U[i][1:]) or ',' not in ''.join(U[i][1:]):
            X=array(U[i][1:])
            X[X=='']='0'
            Data[Ticker][U[i][0]]=map(float,X)
        else:
          continue
    return Data
    
def main(argv):
  Val=1000
  virtual_display=False
  login='abc'
  passwd='abc'
  try:
    opts, args = getopt.getopt(argv,"hvlpd:",["Value=","Login=","Password=","virtual_display="])
  except getopt.GetoptError:
    print "See ReadMagicFormulalist -h"
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
      print '''ReadMagicFormulalist.py 
               -v (--Value)
               -l (--Login)
               -p (--Password)
            '''
      sys.exit()
    elif opt in ("-v", "--Value"):
      Val = eval(arg)
    elif opt in ("-l", "--Login"):
      login = arg
    elif opt in ("-p", "--Password"):
      passwd = arg
    elif opt in ("-d", "--virtual-display"):
      virtual_display = arg
  return Val,login,passwd,virtual_display


if __name__=='__main__':
  Val,Login,Password,virtual_display=main(sys.argv[1:])
  MF=ReadMagicFormula(Login,Password)
  Stocklist1=MF.GetStocklist(Val,virtual_display=virtual_display)
  if Val<1000:
    data_type='Small_Cap'
  if Val>=1000 and Val<5000:
    data_type='Mid_Cap'
  if Val>=5000:
    data_type='Large_Cap'
  if Val<5000:
    Stocklist2=MF.GetStocklist(Val*10)
    Stocklist=list(set(Stocklist1) | set(Stocklist2))
  else:
    Stocklist=Stocklist1
  Dir=MF.GetMorningStartData(Stocklist,data_type,virtual_display=virtual_display)
  