
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators

import pandas as pd 
pd.options.mode.copy_on_write = True 
import numpy as np 
import time
import os 
print(os.getcwd())
import requests 
from pandas.core.common import flatten
import json
import sys 

def create_time_series_data():
    dates = ['2022-06-30','2022-09-30','2022-12-30','2023-03-31' ,'2023-06-30','2023-09-29']
    sequence = {
                '2022-06-30':0
                ,'2022-09-30':1
                ,'2022-12-30':2
                ,'2023-03-31':3
                ,'2023-06-30':4
                ,'2023-09-29':5
                
                }


    def manipulate_consolidate_timeseries(directory):
        dlist = os.listdir(directory)
        l =[]
        errorlist =[]
        ld=[]
    
        for i in range(len(dlist)):
            try:
                tempfile = dlist[i:i+1][0]

                df = pd.read_csv(f'{directory}/{tempfile}',sep=',')
                startdate = df['date'].min()
                enddate = df['date'].max()
                df = df[df['date'].isin(dates)]
                df['sequence']=df['date'].map(sequence)
                df['quarter'] = pd.PeriodIndex(df.date, freq='Q')
                df['next_rolling62_adjustedclose'] = df['rolling62_adjustedclose'].shift(-1)
                l.append(df)
                
                
                dfdate = pd.DataFrame({'file':[tempfile],
                                        'startdate':[startdate],
                                        'enddate':[enddate]})
                ld.append(dfdate)

            except Exception as e:
                print('exception of ',e)
                errorlist.append(tempfile)
        df_out = pd.concat(l)
        df_dates =pd.concat(ld)
        df_dates.to_excel('ticker_TS_start_end_dates.xlsx') #to verify whether they show up in time period or not
        dfcore = df_out[['ticker','rolling62_adjustedclose','next_rolling62_adjustedclose','sequence']]    

        dfcore.loc[:, 'return'] =  (dfcore.loc[:,'next_rolling62_adjustedclose']/dfcore.loc[:,'rolling62_adjustedclose'])-1

        spy = dfcore[dfcore['ticker']=='SPY']
        spy = spy.rename(columns={'return':'SPYreturn'})
        spy = spy[['sequence','SPYreturn']]
        dfcore = dfcore.merge(spy,on='sequence',how='inner')
        dfcore['better_than_spy']=np.where(dfcore['return']>dfcore['SPYreturn'],1,0)

        if len(errorlist)==0:
            print('time series data consolidated')
        else:
            print('the following files had issues:',errorlist)
        #TODO remove this after DEV
        print('length of dfcore:',len(dfcore['ticker'].unique()))
        return dfcore,spy          

    consolidated_time_series,spy = manipulate_consolidate_timeseries('data/TimeSeriesAdjusted')

    #why exclude these? because they miss the cut off date by a few days.  
    consolidated_time_series=consolidated_time_series[consolidated_time_series['ticker']!='GEHC']
    consolidated_time_series=consolidated_time_series[consolidated_time_series['ticker']!='KVUE']
    consolidated_time_series.to_excel('wrangling/Consolidated_TimeSeries_Data.xlsx',index=False)
    spy.to_excel('spy.xlsx',index=False)

#print(len(consolidated_time_series))



create_time_series_data()