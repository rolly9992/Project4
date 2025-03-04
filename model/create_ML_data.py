


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

def create_machine_learning_data():
    quarters =['2022Q2','2022Q3','2022Q4','2023Q1','2023Q2','2023Q3'] 
    sequence = {'2022Q2':0
                ,'2022Q3':1
                ,'2022Q4':2
                ,'2023Q1':3
                ,'2023Q2':4
                ,'2023Q3':5}

    base = pd.read_excel('data/sp500_tickerlist.xlsx')

    #TODO some overlap in column names between the sheets. only use 1


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

    mldata = base.merge(consolidated_BS,on='ticker',how='inner')
    mldata = mldata.merge(consolidated_IS,on = ['ticker','quarter'],how='inner')
    mldata = mldata.merge(consolidated_CF,on = ['ticker','quarter'],how='inner')
    mldata = mldata.drop('security',axis=1)
    mldata = mldata.drop('gics_sub_industry',axis=1)
    mldata = mldata.drop('cik',axis=1)
    mldata = mldata.drop('reportedCurrency',axis=1)
    mldata = mldata.drop('hq_location',axis=1)
    mldata = mldata.drop('date_added',axis=1)
    #mldata = mldata.drop('hq_location',axis=1)
    mldata = mldata.drop('founded',axis=1)

    #mldata = mldata.drop('founded',axis=1)


    #TODO -- make catch if division by zero error for relevant ratios (potentially all of them)
    #liquidity ratios
    mldata['current_ratio'] = mldata['totalCurrentAssets']/mldata['totalCurrentLiabilities']
    mldata['acid_test_ratio']=(mldata['totalCurrentAssets']-mldata['inventory'])/mldata['totalCurrentLiabilities']
    mldata['cash_ratio'] = mldata['cashAndCashEquivalentsAtCarryingValue']/mldata['totalCurrentLiabilities']
    mldata['operating_cash_flow_ratio'] = mldata['operatingCashflow']/mldata['totalCurrentLiabilities']

    #Efficiency Ratios
    mldata['debt_ratio']=mldata['totalLiabilities']/mldata['totalAssets']
    mldata['debt_to_equity_ratio']=mldata['totalLiabilities']/mldata['totalShareholderEquity']

    mldata['interest_coverage_ratio']=np.where(mldata['interestExpense']==0,0,mldata['operatingIncome']/mldata['interestExpense'])

    #mldata['debt_service_ratio']=mldata['operatingIncome']/mldata['interestExpense']
    #The debt service coverage ratio reveals how easily a company can pay its debt obligations:
    #Debt service coverage ratio = Operating income / Total debt service

    #TODO make a note on using an alternate version. sub total revenues for net sales and totalassets for avg assets
    #idea consider using windows fxs to get previous value?? 
    #Asset turnover ratio = Net sales / Average total assets
    mldata['asset_turnover_ratio']=mldata['totalRevenue']/(mldata['totalAssets'])
    mldata['averageInventory']=0.5*(mldata['inventory']*2 -mldata['changeInInventory'])

    mldata['inventory_turnover_ratio']= np.where(0.5*(mldata['inventory']*2 -mldata['changeInInventory'])==0,0,mldata['costofGoodsAndServicesSold']/(0.5*(mldata['inventory']*2 -mldata['changeInInventory'])))

    #The accounts receivable turnover ratio measures how many times a company can turn receivables into cash over a given period:
    #Receivables turnover ratio = Net credit sales / Average accounts receivable
    #mldata['accounts_receivable_turnover_ratio']= net credit sales is NOT in the alpha vantage data
    #Net credit sales / Average accounts receivable



    #mldata['inventory_turnover']= mldata['costofGoodsAndServicesSold']/(0.5*(mldata['inventory']*2 -mldata['changeInInventory']))
    #365 days / Inventory turnover ratio
    mldata['days_in_inventory_ratio']=np.where(mldata['inventory_turnover_ratio'] ==0,0,365/mldata['inventory_turnover_ratio']) #mldata['inventory_turnover']

    #profitability ratios 
    #OBSOLETE as Net Income was in the mix
    #NetIncome=EBIT−InterestExpense−Taxes
    #mldata['netIncome']= mldata['ebit']-mldata['interestExpense']-mldata['incomeTaxExpense']

    mldata['gross_margin_ratio']=np.where(mldata['netIncome']==0,0,mldata['grossProfit']/mldata['netIncome'])
    mldata['operating_margin_ratio']=np.where(mldata['netIncome']==0,0,mldata['operatingIncome']/mldata['netIncome'])
    mldata['roa_ratio']=mldata['netIncome']/mldata['totalAssets']
    mldata['roe_ratio']=mldata['netIncome']/mldata['totalShareholderEquity']
    mldata['sequence']=mldata['quarter'].map(sequence)
    mlcols = mldata.columns.tolist()

    ts = pd.read_excel('wrangling/Consolidated_TimeSeries_Data.xlsx')
    mldata = mldata.merge(ts,on=['ticker','sequence'],how='inner')
    mldata = mldata[mldata['sequence']!=5] #5 is the endpoint. we have all we need from this from a training/testing perspective
    mldata =mldata.drop('SPYreturn',axis=1) #we will use this, but not directly in the ML 
    mldata =mldata.drop('quarter',axis=1) #no further use
    #mldata =mldata.drop('inventory_turnover',axis=1) #no further use
    mldata = mldata.set_index('ticker')





    mldata.to_excel('model/ml_data.xlsx')
    print('machine learning dataset created')
    




create_machine_learning_data()