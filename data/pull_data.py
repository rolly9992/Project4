
import sqlite3
import numpy as np 
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import time 
#getting the list https://stackoverflow.com/questions/44232578/get-the-sp-500-tickers-list
import sys 
import requests
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import os 
#print(os.getcwd())
#from custdefs import custdefs as cd
from pandas.core.common import flatten
import requests 
import sys 
from datetime import date
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta, FR




def get_SandP_List():
        
    def list_wikipedia_sp500() -> pd.DataFrame:
        # Ref: https://stackoverflow.com/a/75845569/
        url = 'https://en.m.wikipedia.org/wiki/List_of_S%26P_500_companies'
        return pd.read_html(url, attrs={'id': 'constituents'}, index_col='Symbol')[0]

    df = list_wikipedia_sp500()
    df.index = df.index.str.replace(' ','')
    df.index = df.index.str.replace('.','-')
    df.index.name = 'ticker'

    columns = {'Symbol':'ticker',
        'Security':'security',
            'GICS Sector':'gics_sector',
            'GICS Sub-Industry':'gics_sub_industry',
                'Headquarters Location':'hq_location',
                'Date added':'date_added',
                'CIK':'cik',
                'Founded':'founded'}
    df = df.rename(columns = columns)   

    add_SPY = { 'security': ['index'],
            'gics_sector':['Multiple'],
            'gics_sub_industry':[''],
                'hq_location':[''],
                'date_added':['1993-01-22'],
                'cik':[''],
                'founded':['1993-01-22']}
    add_SPY = pd.DataFrame(add_SPY)
    add_SPY.index=['SPY']
    df = pd.concat([df,add_SPY]) # add SPY row 
    df.index.name = 'ticker'
    df.to_excel('data/sp500_tickerlist.xlsx')
    print('SPY list pulled and saved')



def pull_adjusted_time_series(key,pull_type):
      
    df = pd.read_excel('data/sp500_tickerlist.xlsx')
    tl = df['ticker']
    tl = list(set(tl))

 
    if pull_type == 'full':
        countpull = len(tl)
    else:
        countpull = 2    
    columns = {'date':'date',
            '1. open':'open_price','2. high':'high_price',
                '3. low':'low_price','4. close':'close_price',
                '5. adjusted close':'adjusted_close_price',
                '6. volume':'volume',
                '7. dividend amount':'dividend_amount',
                '8. split coefficient':'split coefficient'
                }

    df = pd.read_excel('data/sp500_tickerlist.xlsx')
    tl = df['ticker']

    countcheck = len(tl)
    print('pulling ',countpull,' tickers')
    # dates = ['2022-06-30','2022-09-30','2022-12-30','2023-03-31','2023-06-30']
    # sequence = {'2022-06-30':0
    #             ,'2022-09-30':1
    #             ,'2022-12-30':2
    #             ,'2023-03-31':3
    #             ,'2023-06-30':4 }

    errorlist = []
    errorfilesave = []


    l_meta=[]
    for i in range(0,countpull): #len(tl)): #-1,-1,-1):
        #start = time.time() 
        time.sleep(.3)
    
        try:
            
            ticker =tl[i]  #the 9 ETF sector tickers 
            ticker = ticker.replace(' ','')
            ticker = ticker.replace('.','-')
            
            ts = TimeSeries(key, output_format='pandas')
            ts_data, ts_meta_data = ts.get_daily_adjusted(symbol=ticker,outputsize = 'full')
            
            temp = {'ticker': [ts_meta_data['2. Symbol']], 'last_refreshed':[ts_meta_data['3. Last Refreshed']]}
            temp = pd.DataFrame(temp)
            l_meta.append(temp)
            
        
            df_stock = pd.DataFrame(ts_data)
            df_stock['ticker'] = ticker.upper()

            df_stock = df_stock.rename(columns = columns)
            df_stock = df_stock.sort_index(ascending=True)
            df_stock['rolling62_adjustedclose']=df_stock['adjusted_close_price'].rolling(62).mean()
            df_stock= df_stock[df_stock.index>='2022-04-30']
    
            df_stock.to_csv('data/TimeSeriesAdjusted/TS_for_' + ticker + '.csv')

        except Exception as e:
            errorlist.append((i,ticker))
            print(i,ticker)    
            print('this is the error: ',e)
            continue
    print('adjusted time series finished')
    #print('errorlist:',errorlist)
    df_meta=pd.concat(l_meta)
    df_meta.to_excel('data/ts_meta_check.xlsx')


##pulling BALANCE sheets
def pull_Balance_Sheets(key,pull_type):
    #start = time.time()  
    df = pd.read_excel('data/sp500_tickerlist.xlsx')
    tl = df['ticker']
    if pull_type == 'full':
        countpull = len(tl)
    else:
        countpull = 2    
        
    errorlist = []


    for i in range(0,countpull):
        #print(i) 
        time.sleep(.3)
        try:
            ticker=tl[i]
            ticker = ticker.replace(' ','')
            ticker = ticker.replace('.','-')
            url =  f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={ticker}&apikey={key}'
            r = requests.get(url)
            data = r.json()

            quarterly=data['quarterlyReports']
            l=[]
            for i in range(len(quarterly)):
                x=quarterly[i]
                temp=pd.DataFrame([x])
                temp=temp.replace('None',np.nan)
                temp['ticker']=ticker            
                l.append(temp) #.from_dict(quarterly[0],index=[0])
            df=pd.concat(l)
            df.to_csv(f'data/BalanceSheets/AV_BS_{ticker}.csv',sep='|',index=None)
        except Exception as e:
            errorlist.append(tl[i])
            print(e)
            pass
            
    if len(errorlist)>0:
        print('the following tickers did not pull data')
        print(errorlist)
    else:
        print('all balance sheets pulled')
    


###CASHFLOW STATEMENTS
def pull_Cashflow_Statements(key,pull_type):
    #start = time.time()  
    df = pd.read_excel('data/sp500_tickerlist.xlsx')
    tl = df['ticker']

    if pull_type == 'full':
        countpull = len(tl)
    else:
        countpull = 2    
    countcheck = len(tl)


    errorlist = []

    for i in range(0,countpull): 
        #print(i) 
        time.sleep(.3)
        try:
            ticker=tl[i]
            ticker = ticker.replace(' ','')
            ticker = ticker.replace('.','-')

            url = f'https://www.alphavantage.co/query?function=CASH_FLOW&symbol={ticker}&apikey={key}'
            r = requests.get(url)
            data = r.json()
            quarterly=data['quarterlyReports']
            l=[]
            for i in range(len(quarterly)):
                x=quarterly[i]
                temp=pd.DataFrame([x])
                temp=temp.replace('None',np.nan)
                temp['ticker']=ticker
                l.append(temp) #.from_dict(quarterly[0],index=[0])
            df=pd.concat(l)

            df.to_csv(f'data/CashFlowStatements/AV_CF_{ticker}.csv',sep='|',index=None)
            
        except Exception as e:
            errorlist.append(tl[i])
            print('exception of:',e)

            
    if len(errorlist)>0:
        print('the following tickers did not pull data')
        print(errorlist)
    else:
        print('all cash flow statements pulled')


###INCOME STATEMENT
def pull_Income_Statements(key,pull_type):
    #start = time.time()  
    df = pd.read_excel('data/sp500_tickerlist.xlsx')
    tl = df['ticker']
    if pull_type == 'full':
        countpull = len(tl)
    else:
        countpull = 2    
    errorlist=[]

    for i in range(0,countpull): 
        
        time.sleep(.3)
        try:
            ticker=tl[i]
            ticker = ticker.replace(' ','')
            ticker = ticker.replace('.','-')
            # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
            url = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={key}'
            r = requests.get(url)
            data = r.json()
            quarterly=data['quarterlyReports']
            l=[]
            for i in range(len(quarterly)):
                x=quarterly[i]
                temp=pd.DataFrame([x])
                temp=temp.replace('None',np.nan)
                temp['ticker']=ticker
                l.append(temp) #.from_dict(quarterly[0],index=[0])
            df=pd.concat(l)
            #df.to_csv(f'E:\\AlphaVantage_IncomeStatements\AV_IS_{ticker}.csv',sep='|',index=None)
            df.to_csv(f'data/IncomeStatements/AV_IS_{ticker}.csv',sep='|',index=None)
        except Exception as e:
            errorlist.append(tl[i])
            print('exception of:',e)

            
    if len(errorlist)>0:
        print('the following tickers did not pull data')
        print(errorlist)
    else:
        print('all income statements') 
    #print(data)



def main():
    #if len(sys.argv) == 4:

    #    messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
    if len(sys.argv) == 3:
        #todo make option to pull either FULL(500) or SAMPLE(2)

        key, pull_type = sys.argv[1:]    
        
        print('Starting data pull.\n ')
        get_SandP_List()
        pull_adjusted_time_series(key,pull_type)
        pull_Balance_Sheets(key,pull_type)
        pull_Cashflow_Statements(key,pull_type)
        pull_Income_Statements(key,pull_type)


    
    else:
        print('Please provide the AlphaVantage Password followed by either the word full or sample'\
              'full will pull the entire set of S*P 500 companies. '\
              'sample will pull 2 companies for a fast examination of the data ' )


if __name__ == '__main__':
    main()