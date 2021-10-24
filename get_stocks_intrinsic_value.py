import glob
import datetime
import sys,os
Datadir='/home/sachin/Work/DataSets/MorningstarData/MorningStarData'
tm=datetime.datetime.now()
LargeCapFile=glob.glob('%s_Large_Cap_%d_%d_%d_21*'%(Datadir,tm.year,tm.month,tm.day-1))
SmallCapFile=glob.glob('%s_Small_Cap_%d_%d_%d_21*'%(Datadir,tm.year,tm.month,tm.day-1))
print SmallCapFile

if len(LargeCapFile)!=0:
	lg_File=LargeCapFile[0].split('/')[-1]
	str_cmd='python /home/sachin/Work/Python/Git_Folder/investmenttools -i --morningstar-filename %s --data-dir /home/sachin/Work/DataSets/MorningstarData --cagr-method min'%lg_File
	print '*'*10,'Results for Large Cap Stocks','*'*10
	os.system(str_cmd)

if len(SmallCapFile)!=0:
	sc_File=SmallCapFile[0].split('/')[-1]
	str_cmd='python /home/sachin/Work/Python/Git_Folder/investmenttools -i --morningstar-filename %s --data-dir /home/sachin/Work/DataSets/MorningstarData --cagr-method min'%sc_File
	print '*'*10,'Results for Small Cap Stocks','*'*10
	os.system(str_cmd)	
