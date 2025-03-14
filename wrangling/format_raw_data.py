


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
import requests 
from pandas.core.common import flatten
#NOTE: the pandas interpolate method was deprecated in python 3.12 however it reappeared in 3.13. And it works in 3.11 which is what I'm using. 
#for this reason I am supressing the futures warning 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



def calculate_non_blank_columns_in_row(row):
    '''INPUT 
    dataframe row
    OUTPUT value for new column that sums up the non NULL values in that row
    we can then insert this output in a new column by applying a function to the dataframe. 
    This can help us to only have 1 row per ticker per quarter'''
    #simple method to get non null count across a row
    return row.notna().sum() 

def create_time_series_data():
    '''INPUT
    nothing 
    OUTPUT
    converts end of quarter dates to sequence numbers which are easier to both use and understand when coding.
    '''
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
        '''INPUT 
        each of the 504 time series files (including the SPY itself) from the time series subdirectory
        OUTPUT
        a consolidated file for each of the tickers with 6 sequence numbers
        also adds return per ticker per quarter/sequence along with our target variable better than SPY (yes/no)
        a separate saved excel file with SPY data for later comparison. 
        also created a separate excel file with start/end dates for each ticker for QC purposes. 
        
        '''
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
                #filtering to ONLY include end of quarter dates for rolling 62 adjusted close prices
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
        
        return dfcore,spy          

    consolidated_time_series,spy = manipulate_consolidate_timeseries('data/TimeSeriesAdjusted')

    #why exclude these? because they miss the cut off date by a few days.  
    consolidated_time_series=consolidated_time_series[consolidated_time_series['ticker']!='GEHC']
    consolidated_time_series=consolidated_time_series[consolidated_time_series['ticker']!='KVUE']
    consolidated_time_series.to_excel('wrangling/Consolidated_TimeSeries_Data.xlsx',index=False)
    spy.to_excel('spy.xlsx',index=False)



def manipulate_consolidate_data(directory):
    '''INPUT
    directory
    OUTPUT 
    this function is used consolidate each of the financial statement types
    (balance sheets, income statements and cash flow statements)
    It orders by quarter, fills in missing data dedupes if more than 1 statement per quarter. 
    Then filters for only the 6 quarters we are looking at. 
    Finally it combines all the ticker data for a particular financial data type into 1 dataframe 
    '''
    dlist = os.listdir(directory)
    l =[]
    errorlist =[]
    
    for i in range(len(dlist)):
        try:
            tempfile = dlist[i:i+1][0]
            
            df = pd.read_csv(f'{directory}/{tempfile}',sep='|')
            df=df.sort_values('fiscalDateEnding',ascending=True)
            
            
            #interpolate data if any nulls that are between ordered rows with data
            df = df.interpolate(method='linear', limit_direction='forward', axis=0)
            #if rows in the ordered beginning of a column are blank, use bfill 
            df = df.bfill()    
            #finally if any nulls at the end use ffill 
            df = df.ffill()
            
            #fiscalDateEnding in all 3 financial pieces: Balance Sheets, Cash Flows and Income Statements. 
            df = df[df['fiscalDateEnding']>'2021-06-30']
            
            df.fillna(0,inplace=True)
            df['quarter'] = pd.PeriodIndex(df.fiscalDateEnding, freq='Q')
               
            df = df.sort_values(by=['ticker', 'quarter', 'fiscalDateEnding'], ascending=[True, True, False])

            # Drop duplicates, keeping the latest fiscalDateEnding  
            df = df.drop_duplicates(subset=['ticker', 'quarter'], keep='last')

            #backup just in case... perhaps overkill... 
            #applying the function created above to look for 2 items from the same ticker in the same quarter and keep one with more data
            df['row_non_blank_count'] = df.apply(calculate_non_blank_columns_in_row, axis=1)
            df = df.sort_values(by=['ticker', 'fiscalDateEnding','row_non_blank_count'], ascending=[True, True, False])
            df = df.drop_duplicates(subset=['ticker', 'fiscalDateEnding'], keep='first')  # Keep first (the more complete one)
            df = df.drop('row_non_blank_count',axis=1)  # Keep first (the more complete one)
            l.append(df)
            
        except Exception as e:
            print('exception of ',e)
            errorlist.append(tempfile)
    df_out = pd.concat(l)
    if len(errorlist)!=0:
        print('the following files had issues:',errorlist)
    return df_out          
    
def wrap_up_consolidating_financials():
    '''INPUT
    nothing
    OUTPUT 
    cycles through the previous consolidation definition for each of the 3 types of financial data. 
    Then saves each consolidated piece to an excel file. 
    '''
    consolidated_balance_sheets = manipulate_consolidate_data('data/BalanceSheets')
    consolidated_balance_sheets.to_excel('wrangling/Consolidated_BalanceSheet_Data.xlsx',index=False)
    print('consolidated balance sheets completed')

    consolidated_income_statements = manipulate_consolidate_data('data/IncomeStatements')
    consolidated_income_statements.to_excel('wrangling/Consolidated_IncomeStatements_Data.xlsx',index=False)
    print('consolidated income statements completed')

    consolidated_cashflow_statements = manipulate_consolidate_data('data/CashFlowStatements')
    consolidated_cashflow_statements.to_excel('wrangling/Consolidated_CashFlowStatements_Data.xlsx',index=False)
    print('consolidated cashflow statements completed')


def create_machine_learning_data():
    '''INPUT
    nothing 
    OUTPUT 
    combines data from time series, balance sheets, cash flow statements, income statements, previously created by previous 
    definitions. 
    then drops some columns not particularly useful for this data set like cik, HQ location, sub industry, etc. 
    Since we're only dealing with 500, some categories can get very thin. 
    
    Next we add the financial ratios, using definitions from the corporate financial institute (see the file in the reference folder)
    Finally it saves the output from all of this as our machine learning input data. 
    '''
    quarters =['2022Q2','2022Q3','2022Q4','2023Q1','2023Q2','2023Q3'] 
    sequence = {'2022Q2':0
                ,'2022Q3':1
                ,'2022Q4':2
                ,'2023Q1':3
                ,'2023Q2':4
                ,'2023Q3':5}

    base = pd.read_excel('data/sp500_tickerlist.xlsx')

    consolidated_BS = pd.read_excel('wrangling/Consolidated_BalanceSheet_Data.xlsx') 
    consolidated_BS = consolidated_BS[consolidated_BS['quarter'].isin(quarters)]
    consolidated_BS = consolidated_BS.drop('fiscalDateEnding',axis=1)

    consolidated_IS = pd.read_excel('wrangling/Consolidated_IncomeStatements_Data.xlsx')
    consolidated_IS = consolidated_IS[consolidated_IS['quarter'].isin(quarters)]
    consolidated_IS = consolidated_IS.drop('reportedCurrency',axis=1)
    consolidated_IS = consolidated_IS.drop('fiscalDateEnding',axis=1)

    consolidated_CF = pd.read_excel('wrangling/Consolidated_CashFlowStatements_Data.xlsx')
    consolidated_CF = consolidated_CF[consolidated_CF['quarter'].isin(quarters)]
    consolidated_CF = consolidated_CF.drop('reportedCurrency',axis=1)
    consolidated_CF = consolidated_CF.drop('netIncome',axis=1) #dupe - also on balance sheet
    consolidated_CF = consolidated_CF.drop('fiscalDateEnding',axis=1)
    
    
    #making a copy
    mldata = base.copy()
    #merging in 1 step
    mldata = mldata.merge(consolidated_BS,on='ticker',how='inner').merge(consolidated_IS,on = ['ticker','quarter'],how='inner').merge(consolidated_CF,on = ['ticker','quarter'],how='inner')
    #drop irrelevant columns
    mldata = mldata.drop(columns=['security','gics_sub_industry','cik','reportedCurrency','hq_location','date_added','founded'],axis=1)

    #ADDING STANDARD FINANCIAL RATIOS USING DEFINITIONS FROM THE CORPORATE FINANCIAL INSTITUTE
    #NOTE: using a new df avoids a highly fragmented dataframe which gives a performance warning otherwise. So creating a ratios df then concatenating
    ratios = pd.DataFrame()
    ratios['current_ratio'] = mldata['totalCurrentAssets']/mldata['totalCurrentLiabilities']
    ratios['acid_test_ratio']=(mldata['totalCurrentAssets']-mldata['inventory'])/mldata['totalCurrentLiabilities']
    ratios['cash_ratio'] = mldata['cashAndCashEquivalentsAtCarryingValue']/mldata['totalCurrentLiabilities']
    ratios['operating_cash_flow_ratio'] = mldata['operatingCashflow']/mldata['totalCurrentLiabilities']

    #Efficiency Ratios
    ratios['debt_ratio']=mldata['totalLiabilities']/mldata['totalAssets']
    ratios['debt_to_equity_ratio']=mldata['totalLiabilities']/mldata['totalShareholderEquity']

    ratios['interest_coverage_ratio']=np.where(mldata['interestExpense']==0,0,mldata['operatingIncome']/mldata['interestExpense'])

    #Asset turnover ratio = Net sales / Average total assets
    ratios['asset_turnover_ratio']=mldata['totalRevenue']/(mldata['totalAssets'])
    ratios['averageInventory']=0.5*(mldata['inventory']*2 -mldata['changeInInventory'])

    ratios['inventory_turnover_ratio']= np.where(0.5*(mldata['inventory']*2 -mldata['changeInInventory'])==0,0,mldata['costofGoodsAndServicesSold']/(0.5*(mldata['inventory']*2 -mldata['changeInInventory'])))
    ratios['days_in_inventory_ratio']=np.where(ratios['inventory_turnover_ratio'] ==0,0,365/ratios['inventory_turnover_ratio']) #mldata['inventory_turnover']

    ratios['gross_margin_ratio']=np.where(mldata['netIncome']==0,0,mldata['grossProfit']/mldata['netIncome'])
    ratios['operating_margin_ratio']=np.where(mldata['netIncome']==0,0,mldata['operatingIncome']/mldata['netIncome'])
    ratios['roa_ratio']=mldata['netIncome']/mldata['totalAssets']
    ratios['roe_ratio']=mldata['netIncome']/mldata['totalShareholderEquity']
    ratios['sequence']=mldata['quarter'].map(sequence)
    

    mldata = pd.concat([mldata,ratios],axis =1)
    ts = pd.read_excel('wrangling/Consolidated_TimeSeries_Data.xlsx')
    mldata = mldata.merge(ts,on=['ticker','sequence'],how='inner')
    mldata = mldata[mldata['sequence']!=5] #5 is the endpoint. we have all we need from this from a training/testing perspective
    mldata =mldata.drop('SPYreturn',axis=1) #we will use this, but not directly in the ML 
    mldata =mldata.drop('quarter',axis=1) #no further use as we'll be using sequences instead. Less clunky.
    mldata = mldata.set_index('ticker')
    mldata.to_excel('model/ml_data.xlsx')
    print('machine learning dataset created')



def main():
    create_time_series_data()
    wrap_up_consolidating_financials()
    create_machine_learning_data()

if __name__ == '__main__':
    main()




