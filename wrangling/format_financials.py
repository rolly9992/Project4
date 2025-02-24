#format financial files 


from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
# from matplotlib.pyplot import figure
# import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import time
import os 
#print(os.getcwd())
import requests 
from pandas.core.common import flatten
import json
import sys 

#NOTE: the pandas interpolate method was deprecated in python 3.12 however it reappeared in 3.13. And it works in 3.11 which is what I'm using. 
#for this reason I am supressing the futures warning 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#TODO is this still needed? 
def calculate_non_blank_columns_in_row(row):
    '''INPUT 
    dataframe row
    OUTPUT value for new column that sums up the non NULL values in that row
    we can then insert this output in a new column by applying a function to the dataframe. 
    This can help us to only have 1 row per ticker per quarter'''
    #simple method to get non null count across a row
    return row.notna().sum() 



def manipulate_consolidate_data(directory):
    dlist = os.listdir(directory)
    l =[]
    errorlist =[]
    
    for i in range(len(dlist)):
        try:
            tempfile = dlist[i:i+1][0]
            #print(tempfile)
            df = pd.read_csv(f'{directory}/{tempfile}',sep='|')
            df=df.sort_values('fiscalDateEnding',ascending=True)
            
            
            #interpolate data if any nulls that are between ordered rows with data
            df = df.interpolate(method='linear', limit_direction='forward', axis=0)
            #if rows in the ordered beginning of a column are blank, use these 
            df = df.bfill()    
            #finally if any nulls at the end 
            df = df.ffill()
            
            #fiscalDateEnding in all 3 financial pieces: Balance Sheets, Cash Flows and Income Statements. 
            df = df[df['fiscalDateEnding']>'2021-06-30']
            #this is redundant 
            df.fillna(0,inplace=True)
            df['quarter'] = pd.PeriodIndex(df.fiscalDateEnding, freq='Q')
               
            # Sort the DataFrame by 'ID', 'Quarter', and 'Date' in descending order
            df = df.sort_values(by=['ticker', 'quarter', 'fiscalDateEnding'], ascending=[True, True, False])

            # Drop duplicates, keeping the latest fiscalDateEnding 
            df = df.drop_duplicates(subset=['ticker', 'quarter'], keep='first')

            #in case the above doesn't take care of the same ticker in the same quarter (not seeing how it would fail, but... )
            #applying the function created above to look for 2 items from the same ticker in the same quarter and keep one with more data
            df['row_non_blank_count'] = df.apply(calculate_non_blank_columns_in_row, axis=1)
            #print(df.shape) #shape after applying function-- we added the completeness column
            df = df.sort_values(by=['ticker', 'fiscalDateEnding','row_non_blank_count'], ascending=[True, True, False])
            df = df.drop_duplicates(subset=['ticker', 'fiscalDateEnding'], keep='first')  # Keep first (the more complete one)
            df = df.drop('row_non_blank_count',axis=1)  # Keep first (the more complete one)
            l.append(df)
        except Exception as e:
            print('exception of ',e)
            errorlist.append(tempfile)
    df_out = pd.concat(l)
    if len(errorlist)==0:
        print('data consolidated')
    else:
        print('the following files had issues:',errorlist)
    return df_out          
    
def wrap_up_consolidating_financials():
    consolidated_balance_sheets = manipulate_consolidate_data('data/BalanceSheets')
    print(len(consolidated_balance_sheets['ticker'].unique()))
    consolidated_balance_sheets.to_excel('wrangling/Consolidated_BalanceSheet_Data.xlsx',index=False)

    consolidated_income_statements = manipulate_consolidate_data('data/IncomeStatements')
    print(len(consolidated_income_statements['ticker'].unique()))
    consolidated_income_statements.to_excel('wrangling/Consolidated_IncomeStatements_Data.xlsx',index=False)

    consolidated_cashflow_statements = manipulate_consolidate_data('data/CashFlowStatements')
    print(len(consolidated_cashflow_statements['ticker'].unique()))
    consolidated_cashflow_statements.to_excel('wrangling/Consolidated_CashFlowStatements_Data.xlsx',index=False)

wrap_up_consolidating_financials()
