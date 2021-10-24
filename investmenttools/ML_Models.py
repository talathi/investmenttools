import xgboost as xgb
import numpy as np
import pandas as pd
import os,sys
import fastparquet
import glob
import datetime
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from scipy import stats
from xgboost.sklearn import XGBClassifier
import warnings
import matplotlib
matplotlib.use('TKAgg')
import pylab as py
import seaborn as sns
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('%s/..'%file_path)
from investmenttools import PortfolioBuilder as PB;reload(PB)
from investmenttools import GetIndexData as GI;reload(GI)
print file_path
py.ion()
warnings.filterwarnings("ignore")

def label(x):
    return 1 if x > 0 else 0

def maptime(x):
    return x.split(' ')[0]

def combine_stockrowdata(stockrowfile,stockrow_growthfile):
    data = None
    gr_data = None
    combined_data = None
    if os.path.isfile(stockrowfile):
        data = pd.read_csv(stockrowfile)
        data['Date'] = data['Unnamed: 0'].apply(maptime)
        for c in data.columns:
            if data[c].dtypes != 'object':
                data['%s Growth' % c] = (data[c] / data[c].shift(-1))-1 ## date is set in reverse order
    if os.path.isfile(stockrow_growthfile):
        gr_data = pd.read_csv(stockrow_growthfile)
        gr_data.drop(['Book Value per Share Growth'],1,inplace=True)
        gr_data['Date'] = gr_data['Unnamed: 0'].apply(maptime)

    if data is not None and gr_data is not None:
        combined_data = pd.merge(data, gr_data, on=['Date'])
    if data is None:
        combined_data = gr_data
    if gr_data is None:
        combined_data = data

    return combined_data

class IndexDataReader(object):
    def __init__(self,stockrow_datapath,stockprices_datapath,\
                 indexdatafile='%s/../Reference_Data/SP500_ticker.txt'%file_path):
        self.stockrow_folder=stockrow_datapath
        self.prices_file=stockprices_datapath
        if os.path.isfile(indexdatafile):
            self.indexdatafile=indexdatafile
        else:
            print 'Index data file is missing...'


    def read_data(self):
        index_list=pd.read_csv(self.indexdatafile,sep=',')
        self.tickerlist=index_list['Ticker Symbol']
        closes=None
        try:
            closes=pd.read_parquet(self.prices_file)
        except:
            print 'Check the stock price data file path and format...'

        ## read stockrow data
        stockrow_data=None
        for stk in index_list['Ticker Symbol']:
            stockrowfile='%s/%s.csv'%(self.stockrow_folder,stk)
            stockrow_growthfile='%s/%s_Growth.csv'%(self.stockrow_folder,stk)
            try:
                combined_data = combine_stockrowdata(stockrowfile, stockrow_growthfile)
                combined_data['Stock']=[stk]*combined_data.shape[0]
                stockrow_data=pd.concat([stockrow_data,combined_data])
            except:
                print 'Data not available for stock: %s'%stk
        self.X=stockrow_data
        self.y=closes

        self.X.set_index('Date', inplace=True)
        self.X.drop(['Unnamed:'], inplace=True)
        self.X['Date'] = self.X.index
        self.X['Date'] = pd.to_datetime([datetime.datetime.strptime(x, '%Y-%m-%d') for x in self.X['Date']])
        self.X.reset_index(inplace=True, drop=True)

        dl = PB.calc_daily_returns(closes)
        ql = PB.calc_quaterly_returns(dl)

        df = self.X.set_index('Stock',drop=False)
        #df['Stock']=df.index
        train_test_data = None
        for stk in self.tickerlist:
            try:
                val = df.loc[stk]
                val.set_index('Date', inplace=True)
                per = val.index.to_period("Q")
                val_q = val.groupby(per).sum()
                val_q['Stock']=[stk]*val_q.shape[0]
                val_q['price'] = ql[stk]
                #val_q['price_shift']=val_q['price']-val_q['price'].shift(1)
                #val_q['price_shift']=val_q['price'].shift(-1)-val_q['price']
                val_q['price_shift']=np.sign(val_q['price'].shift(-1))
                train_test_data = pd.concat([train_test_data, val_q.iloc[1:-1]])
            except:
                print 'data does not exist for %s' % stk
        try:
            train_test_data.replace([np.inf, -np.inf], 0.0, inplace=True)
        except AttributeError as e:
            print e
        train_test_data['label']=train_test_data['price_shift'].apply(label)

        self.train_test_data=train_test_data ## drop 1st and last quarter row

    def gen_train_test(self,droplist_bool=True,test_size=0.2,date_threshold=None):
        droplist=['Average Inventory Growth','FCF per Share Growth'\
                  ,'Interest Debt per Share Growth','R&D to Revenue Growth','Weighted Average Shares Diluted Growth'\
                  ,'Income Quality Growth']
        self.predictors=[]
        for c in self.train_test_data.columns:
            if 'Growth' in c or 'delta' in c:
                self.predictors.append(c)
        self.target=['label']

        df=self.train_test_data[(np.abs(stats.zscore(self.train_test_data[self.predictors])) < 3).all(axis=1)] #remove outliers
        #self.train_test_stats = df[self.predictors].describe()

        if droplist_bool:
            self.predictors=list(set.difference(set(self.predictors), set(droplist)))
        if date_threshold is None:
            df_train,df_test=train_test_split(df[self.predictors+self.target], test_size=test_size, random_state=42)
        else:
            ind=np.where(df.index<=date_threshold)[0]
            df_train=df[self.predictors+self.target].iloc[ind,:]
            ind = np.where(df.index > date_threshold)[0]
            df_test = df[self.predictors + self.target].iloc[ind, :]

        self.train_stats=df_train.describe()
        m,s=df_train.mean(),df_train.std()
        df_train[self.predictors]=(df_train[self.predictors]-m[self.predictors])/s[self.predictors]
        df_test[self.predictors]=(df_test[self.predictors]-m[self.predictors])/s[self.predictors]
        # df_train[self.predictors]=np.log10(1 + df_train[self.predictors]-df_train[self.predictors].min())
        # df_test[self.predictors] = np.log10(1 + df_test[self.predictors] - df_train[self.predictors].min())

        self.train=df_train
        self.test=df_test
        #self.train, self.test = train_test_split(df[self.predictors+self.target],test_size=test_size, random_state=42)


    def hyperParameter_tuning(self,lr=0.1,max_depth=5,min_child_weight=1,gamma=0,subsample=0.8,\
                              colsample_bytree=0.8,objective='binary:logistic',nthread=4,\
                              scale_pos_weight=1,seed=42,n_estimators=500):

        def get_testscores(model,data,ytrue):
            ypred=model.predict(data)
            score=[metrics.accuracy_score(ytrue,ypred),metrics.roc_auc_score(ytrue,ypred)]
            return score

        self.params={'lr':lr,'max_depth':max_depth,'min_child_weight':min_child_weight,'gamma':gamma,\
                     'subsample':subsample,'colsample_bytree':colsample_bytree,'objective':objective,\
                'nthread':nthread,'scale_pos_weight':scale_pos_weight,'n_estimators':n_estimators}

        self.model= XGBClassifier(learning_rate =self.params['lr'],n_estimators=self.params['n_estimators'],\
                                  max_depth=self.params['max_depth'],min_child_weight=self.params['min_child_weight']\
                                  ,gamma=self.params['gamma'],subsample=self.params['subsample'],\
                                        colsample_bytree=self.params['colsample_bytree'],objective= self.params['objective'],\
                                        n_jobs=-1,scale_pos_weight=self.params['scale_pos_weight'],seed=seed,scoring='roc_auc')

        ### first tune for num trees....
        print 'Train baseline model to estimate no.of trees...'
        self.model,self.cvresult=modelfit(self.model,self.train,self.test,self.predictors,self.target[0])
        print self.model

        ### tune max_depth and min_child_Weight
        print 'Tune max_depth and min_child_weight...'
        param_test1 = {'max_depth': range(3, 11, 1),'min_child_weight': range(1, 6, 1)}
        self.gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate =self.params['lr'],n_estimators=len(self.cvresult),\
                                                        max_depth=self.params['max_depth'],\
                                                        min_child_weight=self.params['min_child_weight'],\
                                                        gamma=self.params['gamma'],subsample=self.params['subsample'],\
                                                        colsample_bytree=self.params['colsample_bytree'],\
                                                        objective= self.params['objective'],n_jobs=-1,\
                                                        scale_pos_weight=self.params['scale_pos_weight'],seed=seed,\
                                                        scoring='roc_auc'),param_grid=param_test1, scoring='roc_auc', \
                                n_jobs=-1,iid=False, cv=5)
        self.gsearch1.fit(self.train[self.predictors], self.train[self.target[0]])
        print self.gsearch1.best_params_, self.gsearch1.best_score_
        print get_testscores(self.gsearch1.best_estimator_,self.test[self.predictors],self.test[self.target[0]])


        ## increase the range of min_child
        print 'Fine tune min_child_weight...'
        params_test2={'min_child_weight': range(6, 13, 2)}
        self.gsearch2=GridSearchCV(estimator=self.gsearch1.best_estimator_,param_grid=params_test2,scoring='roc_auc',\
                                       n_jobs=-1,iid=False,cv=5)
        self.gsearch2.fit(self.train[self.predictors], self.train[self.target[0]])
        print self.gsearch2.best_params_, self.gsearch2.best_score_
        print get_testscores(self.gsearch2.best_estimator_, self.test[self.predictors], self.test[self.target[0]])

        ## test for gamma

        ## tune for subsample and colsample_bytree
        print 'Tune subsample, colsample_bytree...'
        params_test3={'subsample':[i/10.0 for i in range(6,10)],'colsample_bytree':[i/10.0 for i in range(6,10)]}
        self.gsearch3 = GridSearchCV(estimator=self.gsearch2.best_estimator_, param_grid=params_test3,
                                     scoring='roc_auc', n_jobs=-1, iid=False, cv=5)
        self.gsearch3.fit(self.train[self.predictors], self.train[self.target[0]])
        print self.gsearch3.best_params_, self.gsearch3.best_score_
        print get_testscores(self.gsearch3.best_estimator_, self.test[self.predictors], self.test[self.target[0]])

        ## test for regularization parameters
        print 'Tune regularization parameter ...'
        params_test4 = {'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05]}
        self.gsearch4 = GridSearchCV(estimator=self.gsearch2.best_estimator_, param_grid=params_test4,
                                     scoring='roc_auc', n_jobs=-1, iid=False, cv=5)
        self.gsearch4.fit(self.train[self.predictors], self.train[self.target[0]])
        print self.gsearch4.best_params_, self.gsearch3.best_score_
        print get_testscores(self.gsearch4.best_estimator_, self.test[self.predictors], self.test[self.target[0]])

        ## reset the model
        print 'reset model with hyper-parametertuned parameters and retrain...'
        self.model.set_params(n_estimators=500,max_depth=self.gsearch1.best_params_['max_depth'],\
                              min_child_weight=self.gsearch2.best_params_['min_child_weight'],\
                              colsample_bytree=self.gsearch3.best_params_['colsample_bytree'],\
                              subsample=self.gsearch3.best_params_['subsample'],\
                              reg_alpha=self.gsearch4.best_params_['reg_alpha'])
        self.model,self.cvresult=modelfit(self.model,self.train,self.test,self.predictors,self.target[0])
        print self.model

        ## Finally reduce learning rate
        print 'reduce learning rate, increase no.of trees and re train...'
        self.model.set_params(learning_rate=self.params['lr']/10.,n_estimators=2000)
        self.model, self.cvresult = modelfit(self.model, self.train,self.test, self.predictors, self.target[0])
        print self.model
        #print get_testscores(self.model, self.test[self.predictors], self.test[self.target])

def predict_returns_direction(model,predictors,train_stats,stock,save_dir,memo='Tmp_StockData'):
    from xgboost.sklearn import XGBClassifier
    indx=GI.Index(save_dir=save_dir,memo=memo,stock=stock)

    stockrowfile='%s/%s/%s_Metrics.csv'%(save_dir,memo,stock)
    stockrow_growthfile='%s/%s/%s_Growth.csv'%(save_dir,memo,stock)

    if not os.path.isfile(stockrowfile) and not os.path.isfile(stockrow_growthfile):
        if sys.platform=='darwin':
            vdisplay_bool=False
        else:vdisplay_bool=True
        types=['Metrics','Growth']
        for t in types:
            indx.get_stockrowData(type=t,vdisplay_bool=vdisplay_bool)

    combined_data = combine_stockrowdata(stockrowfile, stockrow_growthfile)
    data=(combined_data[predictors]-train_stats[predictors].loc['mean'])/train_stats[predictors].loc['mean']
    print model.predict(data.iloc[0:1])[0]
    return combined_data,data

def modelfit(alg, dtrain, dtest, predictors,target, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['label'], eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]
    dtest_predictions = alg.predict(dtest[predictors])
    dtest_predprob = alg.predict_proba(dtest[predictors])[:, 1]

    # Print model report:
    print "\nModel Report"
    print "Accuracy (Train): %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob)

    print "Accuracy (Test): %.4g" % metrics.accuracy_score(dtest[target].values, dtest_predictions)
    print "AUC Score (Test): %f" % metrics.roc_auc_score(dtest[target], dtest_predprob)

    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    #plt.ylabel('Feature Importance Score')
    print len(cvresult)
    return alg,cvresult

def plot_corr(df):
    sns.set(style="white")

    # Generate a large random dataset
    rs = np.random.RandomState(33)
    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = py.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    g=sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    for item in g.get_xticklabels():
        item.set_rotation(90)
    for item in g.get_yticklabels():
        item.set_rotation(90)
'''
dataReader.X.set_index('Date',inplace=True)
dataReader.X.drop(['Unnamed:'],inplace=True)
dataReader.X['Date']=dataReader.X.index
dataReader.X['Date']=pd.to_datetime([datetime.datetime.strptime(x,'%Y-%m-%d') for x in dataReader.X['Date']])
dataReader.X.reset_index(inplace=True,drop=True)

dl=PB.calc_daily_returns(closes)
ql=PB.calc_quaterly_returns(dl)

df=dataReader.X.set_index('Stock')
revised_data=None
for stk in stocklist:
    try:
        val=df.loc[stk]
        val.set_index('Date',inplace=True)
        per=val.index.to_period("Q")
        val_q=val.groupby(per).sum()
        val_q['price']=ql[stk]
        revised_data=pd.concat([revised_data,val_q])
    except:
        print 'data does not exist for %s'%stk 
'''