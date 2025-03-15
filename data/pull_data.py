
 

import numpy as np 
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import time 
import sys 
import requests
from alpha_vantage.timeseries import TimeSeries
import os 



#NOTE getting the list https://stackoverflow.com/questions/44232578/get-the-sp-500-tickers-list
def get_SandP_List():
    '''INPUT 
    Nothing 
    OUTPUT 
    pulls latest list of S&P 500 from wikipedia to then use a a list of tickers to 
    pull data on 
    '''    
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
    tl = df.index
    tl = list(set(tl))
    return tl



def pull_adjusted_time_series(key,pull_type,tickerlist):
    '''
    INPUT 
    key - the alpha vantage password
    pull type either full (all data) or sample (2 files only for speed)
    tickerlist - the output of the previous definition
    OUTPUT  
    pulls individual adjusted time series data on the ticker list 
    and stores in a TimeSeries subdirectory
    '''


    #dir_ts = "data/TimeSeriesAdjusted"
    if not os.path.exists("data/TimeSeriesAdjusted"):
        os.makedirs("data/TimeSeriesAdjusted") 
     
    
    tl = tickerlist

 
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


    #countcheck = len(tl)
    print('pulling ',countpull,' tickers')
    errorlist = []

    l_meta=[]
    for i in range(0,countpull):
        time.sleep(.3)
    
        try:
            
            ticker =tl[i]   
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
            errorlist.append(ticker)
            print(i,ticker)    
            print('this is the error: ',e)
            continue
    print('adjusted time series data pulled')
    #print('errorlist:',errorlist)
    df_meta=pd.concat(l_meta)
    df_meta.to_excel('data/ts_meta_check.xlsx')
    return errorlist

##pulling BALANCE sheets
def pull_Balance_Sheets(key,pull_type,tickerlist):
    '''
    INPUT 
    key - the alpha vantage password
    pull type either full (all data) or sample (2 files only for speed)
    tickerlist - the output of the previous definition
    OUTPUT  
    pulls individual Balance Sheet data on the ticker list 
    and stores in a BalanceSheet subdirectory
    '''
    
    
    #dir_bs = "data/BalanceSheets"
    if not os.path.exists("data/BalanceSheets"):
        os.makedirs("data/BalanceSheets") 
      
    
    df = pd.read_excel('data/sp500_tickerlist.xlsx')
    tl = tickerlist
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
                l.append(temp) 
            df=pd.concat(l)
            df.to_csv(f'data/BalanceSheets/AV_BS_{ticker}.csv',sep='|',index=None)
        except Exception as e:
            if tl[i]=='SPY':
                pass
            else:
                errorlist.append(tl[i])
            
    if len(errorlist)>0:
        print('the following tickers did not pull balance sheet data:')
        print(errorlist)
    else:
        print('all balance sheets pulled')
    return errorlist    


###CASHFLOW STATEMENTS
def pull_Cashflow_Statements(key,pull_type,tickerlist):
    '''
    INPUT 
    key - the alpha vantage password
    pull type either full (all data) or sample (2 files only for speed)
    tickerlist - the output of the previous definition
    OUTPUT  
    pulls individual CashFlow Sheet data on the ticker list 
    and stores in a CashFlow Sheet subdirectory
    '''
    
    #dir_cf = "data/CashFlowStatements"
    if not os.path.exists("data/CashFlowStatements"):
        os.makedirs("data/CashFlowStatements") 
      
      
    df = pd.read_excel('data/sp500_tickerlist.xlsx')
    tl = tickerlist 

    if pull_type == 'full':
        countpull = len(tl)
    else:
        countpull = 2    



    errorlist = []

    for i in range(0,countpull): 
    
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
            
        except Exception:
            #the SPY itself won't have financial statements. 
            if tl[i]=='SPY':
                pass
            else:
                errorlist.append(tl[i])

            
    if len(errorlist)>0:
        print('the following tickers did not pull cash flow statement data:')
        print(errorlist)
    else:
        print('all cash flow statements pulled')
    return errorlist

###INCOME STATEMENT
def pull_Income_Statements(key,pull_type,tickerlist):
    '''
    INPUT 
    key - the alpha vantage password
    pull type either full (all data) or sample (2 files only for speed)
    tickerlist - the output of the previous definition
    OUTPUT  
    pulls individual Income Statement data on the ticker list 
    and stores in a Income Statement subdirectory
    '''
    
    #dir_ts = "data/IncomeStatements"
    if not os.path.exists("data/IncomeStatements"):
        os.makedirs("data/IncomeStatements") 
      
      
    df = pd.read_excel('data/sp500_tickerlist.xlsx')
    tl = tickerlist
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
                l.append(temp) 
            df=pd.concat(l)
            df.to_csv(f'data/IncomeStatements/AV_IS_{ticker}.csv',sep='|',index=None)
        except Exception:
            if tl[i]=='SPY':
                pass
            else:
                errorlist.append(tl[i])
                

            
    if len(errorlist)>0:
        print('the following tickers did not pull income statement data:')
        print(errorlist)
    else:
        print('all income statements pulled') 
    return errorlist



def main():
    if len(sys.argv) == 3:
        key, pull_type = sys.argv[1:]    
        
        print('Starting data pull.\n ')
        tl = get_SandP_List()

        #NOTE occassionally the AV API misses a ticker here or there. Giving it a second round with missed tickers
        missed = pull_adjusted_time_series(key,pull_type,tickerlist=tl)
        if len(missed)>0:
            pull_adjusted_time_series(key,pull_type,tickerlist=missed)
               
        missed = pull_Balance_Sheets(key,pull_type,tickerlist=tl)
        if len(missed)>0:
            pull_Balance_Sheets(key,pull_type,tickerlist=missed)
            
        missed = pull_Cashflow_Statements(key,pull_type,tickerlist=tl)
        if len(missed)>0:
            pull_Cashflow_Statements(key,pull_type,tickerlist=missed)
        
        missed = pull_Income_Statements(key,pull_type,tickerlist=tl)
        if len(missed)>0:
            pull_Income_Statements(key,pull_type,tickerlist=missed)
        print('''time series data, income statement, cash flow, balance sheet data pulled. 
              \nNOTE: It is possible that Alpha Vantage does not have data for some tickers.
              Any such exceptions would be listed in the above attempts. 
              ''')

    
    else:
        print('Please provide the AlphaVantage Password followed by either the word full or sample'\
              'full will pull the entire set of S*P 500 companies. '\
              'sample will pull 2 companies for a fast pull to verify that the code works ' )


if __name__ == '__main__':
    main()