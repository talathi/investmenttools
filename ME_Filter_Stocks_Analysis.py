import pandas as pd
import sys,os
import datetime

HOME=os.environ['HOME']
file_path = os.path.dirname(os.path.realpath('__file__'))

sys_path='%s/..'%file_path
sys.path.append(sys_path)
from investmenttools import PortfolioBuilder as PB
reload(PB)
from investmenttools import ReadMagicFormulaList as RF
reload(RF)
from investmenttools import Intrinsic_Value as IV
reload(IV)

def get_stocklist(csvfile,oneyeardrop=None):
    stklist = pd.read_csv(csvfile)
    stklist.head(5)
    if oneyeardrop is not None:
        stklist['1 Year Change'].replace('--', 0.0, inplace=True)
        stklist.dropna(inplace=True)
        shortList = stklist[stklist['1 Year Change'].astype(float) < oneyeardrop]
    else:
        shortlist=stklist.copy()
    return shortlist
