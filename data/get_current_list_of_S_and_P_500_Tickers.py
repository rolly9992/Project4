#TODO 

#add multiprocessing to pull all data with 1 script. 
import sqlite3
import pandas as pd
#getting the list https://stackoverflow.com/questions/44232578/get-the-sp-500-tickers-list
import sys 

def list_wikipedia_sp500() -> pd.DataFrame:
    # Ref: https://stackoverflow.com/a/75845569/
    url = 'https://en.m.wikipedia.org/wiki/List_of_S%26P_500_companies'
    return pd.read_html(url, attrs={'id': 'constituents'}, index_col='Symbol')[0]

df = list_wikipedia_sp500()
#print(df.columns)
#print(df.index)
df.index = df.index.str.replace(' ','')
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
print(df.columns)
#df = df['ticker'].str.replace(' ','')


add_SPY = { 'security': ['index'],
           'gics_sector':[''],
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





