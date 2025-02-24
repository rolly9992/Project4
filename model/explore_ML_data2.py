#TODO add grid searches 
#TODO add regressors
#TODO collect probabilities 
#TODO make an alg to compare spy returns vs individual picks. 
#   say, number of securities to split into, low prob for classifier, and predicted return for regressor. 


import pandas as pd 
pd.options.mode.chained_assignment = None
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
from scipy import stats
pd.options.display.float_format = '{:20,.2f}'.format
import sys 
from sklearn import svm


#df = pd.read_excel('/home/milo/Documents/Project4/data/machineLearningData.xlsx')
df = pd.read_excel('ml_data.xlsx')
df = df.set_index('ticker')

#sys.exit()
df_orig = df

#print(df.columns.tolist())

['gics_sector', 'totalAssets', 'totalCurrentAssets', 'cashAndCashEquivalentsAtCarryingValue', 
 'cashAndShortTermInvestments', 'inventory', 'currentNetReceivables', 'totalNonCurrentAssets',
 'propertyPlantEquipment', 'accumulatedDepreciationAmortizationPPE', 'intangibleAssets', 
 'intangibleAssetsExcludingGoodwill', 'goodwill', 'investments', 'longTermInvestments',
 'shortTermInvestments', 'otherCurrentAssets', 'otherNonCurrentAssets', 'totalLiabilities', 
 'totalCurrentLiabilities', 'currentAccountsPayable', 'deferredRevenue', 'currentDebt', 'shortTermDebt',
 'totalNonCurrentLiabilities', 'capitalLeaseObligations', 'longTermDebt', 'currentLongTermDebt', 
 'longTermDebtNoncurrent', 'shortLongTermDebtTotal', 'otherCurrentLiabilities', 
 'otherNonCurrentLiabilities', 'totalShareholderEquity', 'treasuryStock',
 'retainedEarnings', 'commonStock', 'commonStockSharesOutstanding', 'grossProfit', 
 'totalRevenue', 'costOfRevenue', 'costofGoodsAndServicesSold', 'operatingIncome', 
 'sellingGeneralAndAdministrative', 'researchAndDevelopment', 'operatingExpenses', 
 'investmentIncomeNet', 'netInterestIncome', 'interestIncome', 'interestExpense', 
 'nonInterestIncome', 'otherNonOperatingIncome', 'depreciation', 'depreciationAndAmortization', 
 'incomeBeforeTax', 'incomeTaxExpense', 'interestAndDebtExpense', 
 'netIncomeFromContinuingOperations', 'comprehensiveIncomeNetOfTax', 'ebit',
 'ebitda', 'netIncome', 'operatingCashflow', 'paymentsForOperatingActivities', 
  'proceedsFromOperatingActivities', 'changeInOperatingLiabilities', 
 'changeInOperatingAssets', 'depreciationDepletionAndAmortization',
 'capitalExpenditures', 'changeInReceivables', 'changeInInventory', 'profitLoss', 
 'cashflowFromInvestment', 'cashflowFromFinancing', 'proceedsFromRepaymentsOfShortTermDebt',
 'paymentsForRepurchaseOfCommonStock', 'paymentsForRepurchaseOfEquity',
 'paymentsForRepurchaseOfPreferredStock', 'dividendPayout', 'dividendPayoutCommonStock', 
 'dividendPayoutPreferredStock', 'proceedsFromIssuanceOfCommonStock', 
 'proceedsFromIssuanceOfLongTermDebtAndCapitalSecuritiesNet', 
 'proceedsFromIssuanceOfPreferredStock', 'proceedsFromRepurchaseOfEquity',
 'proceedsFromSaleOfTreasuryStock', 'changeInCashAndCashEquivalents', 
 'changeInExchangeRate', 'current_ratio', 'acid_test_ratio', 'cash_ratio',
 'operating_cash_flow_ratio', 'debt_ratio', 'debt_to_equity_ratio', 
 'interest_coverage_ratio', 'asset_turnover_ratio', 'averageInventory',
 'inventory_turnover_ratio', 'days_in_inventory_ratio', 'gross_margin_ratio', 
 'operating_margin_ratio', 'roa_ratio', 'roe_ratio', 'sequence',
 'rolling90_adjustedclose', 'next_rolling90_adjustedclose', 'return', 'better_than_spy']
corr = df[['totalAssets', 'totalCurrentAssets', 'cashAndCashEquivalentsAtCarryingValue', 
 'cashAndShortTermInvestments', 'inventory', 'currentNetReceivables', 'totalNonCurrentAssets',
 'propertyPlantEquipment', 'accumulatedDepreciationAmortizationPPE', 'intangibleAssets', 
 'intangibleAssetsExcludingGoodwill', 'goodwill', 'investments', 'longTermInvestments',
 'shortTermInvestments', 'otherCurrentAssets', 'otherNonCurrentAssets', 'totalLiabilities', 
 'totalCurrentLiabilities', 'currentAccountsPayable', 'deferredRevenue', 'currentDebt', 'shortTermDebt',
 'totalNonCurrentLiabilities', 'capitalLeaseObligations', 'longTermDebt', 'currentLongTermDebt', 
 'longTermDebtNoncurrent', 'shortLongTermDebtTotal', 'otherCurrentLiabilities', 
 'otherNonCurrentLiabilities', 'totalShareholderEquity', 'treasuryStock',
 'retainedEarnings', 'commonStock', 'commonStockSharesOutstanding', 'grossProfit', 
 'totalRevenue', 'costOfRevenue', 'costofGoodsAndServicesSold', 'operatingIncome', 
 'sellingGeneralAndAdministrative', 'researchAndDevelopment', 'operatingExpenses', 
 'investmentIncomeNet', 'netInterestIncome', 'interestIncome', 'interestExpense', 
 'nonInterestIncome', 'otherNonOperatingIncome', 'depreciation', 'depreciationAndAmortization', 
 'incomeBeforeTax', 'incomeTaxExpense', 'interestAndDebtExpense', 
 'netIncomeFromContinuingOperations', 'comprehensiveIncomeNetOfTax', 'ebit',
 'ebitda', 'netIncome', 'operatingCashflow', 'paymentsForOperatingActivities', 
  'proceedsFromOperatingActivities', 'changeInOperatingLiabilities', 
 'changeInOperatingAssets', 'depreciationDepletionAndAmortization',
 'capitalExpenditures', 'changeInReceivables', 'changeInInventory', 'profitLoss', 
 'cashflowFromInvestment', 'cashflowFromFinancing', 'proceedsFromRepaymentsOfShortTermDebt',
 'paymentsForRepurchaseOfCommonStock', 'paymentsForRepurchaseOfEquity',
 'paymentsForRepurchaseOfPreferredStock', 'dividendPayout', 'dividendPayoutCommonStock', 
 'dividendPayoutPreferredStock', 'proceedsFromIssuanceOfCommonStock', 
 'proceedsFromIssuanceOfLongTermDebtAndCapitalSecuritiesNet', 
 'proceedsFromIssuanceOfPreferredStock', 'proceedsFromRepurchaseOfEquity',
 'proceedsFromSaleOfTreasuryStock', 'changeInCashAndCashEquivalents', 
 'changeInExchangeRate', 'current_ratio', 'acid_test_ratio', 'cash_ratio',
 'operating_cash_flow_ratio', 'debt_ratio', 'debt_to_equity_ratio', 
 'interest_coverage_ratio', 'asset_turnover_ratio', 'averageInventory',
 'inventory_turnover_ratio', 'days_in_inventory_ratio', 'gross_margin_ratio', 
 'operating_margin_ratio', 'roa_ratio', 'roe_ratio', 'sequence',
 'rolling90_adjustedclose', 'next_rolling90_adjustedclose', 'return', 'better_than_spy']]
corr = corr.corr()
#sys.exit()


#TODO add grid search. 
#TODO add regression as secondary filter. 

def prep_for_classifier(df,sequence):
    df = df[df['sequence']==sequence]

    #todo see if we can get this working 
    
    df = df.drop('days_in_inventory_ratio',axis=1)
    df = df.drop('interest_coverage_ratio',axis=1)
    
    #df_orig = df
    df = df[df['sequence']==sequence]
    df = df.drop('sequence',axis=1)
    df = df.drop('return',axis=1) #we will use this later. But we would not know this in advance. 

    
    
    #borrowing some code from project 1 
    num_vars = df.select_dtypes(include=['float','int']).columns
    cat_vars = df.select_dtypes(include=['object']).columns
    dfnum = df[num_vars]
    booleanvars = [col for col in dfnum.columns if set(dfnum[col].unique()).issubset({0,1})]
    #print(len(booleanvars))
    nonbooleanvars = list(set(num_vars)-set(booleanvars))

    #create dummy variables for the categorical variable set then drop the original categorical non numerical columns
    l = [df[cat_vars]]
    for i in range(len(cat_vars)):
        temp = cat_vars[i]
        catout = pd.get_dummies(df[temp],prefix=temp,prefix_sep='_',dummy_na=True,drop_first=True)
        l.append(catout)
    dfcat=pd.concat(l,axis=1)
    dfcat=dfcat.drop(columns=cat_vars,axis=0)
    #print(df.shape)
    #print(dfcat.shape) #expecting an increase due to adding dummies. 
    cat_cols = dfcat.columns.tolist()
    for i in range(len(cat_cols)):
        dfcat[cat_cols[i]] = dfcat[cat_cols[i]].astype(int)
    
    df_bool = df[booleanvars]
    df_nonbool = df[nonbooleanvars]
    df_nonbool = (df_nonbool-df_nonbool.mean())/df_nonbool.std()
    new_df = pd.concat([dfcat,df_nonbool,df_bool],axis=1)
    


    
    #constant_columns = []    
    X = new_df.iloc[:,:-2]
    #scrap this. bagged use of logistic regression model 
    #just in case there is a column in a subset where all rows the same value. 
    # try:
    #     #X=X.drop('gic_sector_nan',axis=1)
    #     for column in X.columns:
    #         if X[column].nunique() <= 1:  
    #             constant_columns.append(column)
    #     for i in range(len(constant_columns)):
    #         X=X.drop(constant_columns[i],axis=1)
    # except Exception as e:
    #     print('\n\nexception of:',e)
        #sys.exit()
    X.to_excel('X.xlsx')    
    print(X.columns.tolist())
    y = new_df.iloc[:,-1:]
    y=y['better_than_spy'].values
    return new_df, X, y 


######

#df_orig = df
df = df[df['sequence']!=3 ]
df = df.drop('sequence',axis=1)
df = df.drop('return',axis=1) #we will use this later. But we would not know this in advance. 



#borrowing some code from project 1 
num_vars = df.select_dtypes(include=['float','int']).columns
cat_vars = df.select_dtypes(include=['object']).columns
dfnum = df[num_vars]
booleanvars = [col for col in dfnum.columns if set(dfnum[col].unique()).issubset({0,1})]
#print(len(booleanvars))
nonbooleanvars = list(set(num_vars)-set(booleanvars))

#create dummy variables for the categorical variable set then drop the original categorical non numerical columns
l = [df[cat_vars]]
for i in range(len(cat_vars)):
    temp = cat_vars[i]
    catout = pd.get_dummies(df[temp],prefix=temp,prefix_sep='_',dummy_na=True,drop_first=True)
    l.append(catout)
dfcat=pd.concat(l,axis=1)
dfcat=dfcat.drop(columns=cat_vars,axis=0)
#print(df.shape)
#print(dfcat.shape) #expecting an increase due to adding dummies. 
cat_cols = dfcat.columns.tolist()
for i in range(len(cat_cols)):
    dfcat[cat_cols[i]] = dfcat[cat_cols[i]].astype(int)

df_bool = df[booleanvars]
df_nonbool = df[nonbooleanvars]
df_nonbool = (df_nonbool-df_nonbool.mean())/df_nonbool.std()
new_df = pd.concat([dfcat,df_nonbool,df_bool],axis=1)




#constant_columns = []    
X = new_df.iloc[:,:-2]
#scrap this. bagged use of logistic regression model 
#just in case there is a column in a subset where all rows the same value. 
# try:
#     #X=X.drop('gic_sector_nan',axis=1)
#     for column in X.columns:
#         if X[column].nunique() <= 1:  
#             constant_columns.append(column)
#     for i in range(len(constant_columns)):
#         X=X.drop(constant_columns[i],axis=1)
# except Exception as e:
#     print('\n\nexception of:',e)
    #sys.exit()
X.to_excel('X.xlsx')    
print(X.columns.tolist())
y = new_df.iloc[:,-1:]
y=y['better_than_spy'].values



sys.exit()

def evaluate_model(model, X_test, Y_test): #, category_names=None):
    '''INPUT
    the machine learning model we built earlier
    X_test data
    Y_test data
    the category names (pulled from the Y_test dataframe)
    
    OUTPUT 
    excel metric files on how well each category performed, including 
    -accuracy
    -precision
    -recall
    -the F1 score
    
    the output files are later used in some visuals in the Flask app
       
    '''

    y_pred = model.predict(X_test)
    #y#cols = Y_test.columns.tolist()
    #category_names =ycols
    #y_pred2 = pd.DataFrame(y_pred,columns=ycols)
    #l = []
    #for i in range(len(ycols)):
        
    accuracy=accuracy_score(Y_test,y_pred)
    precision=precision_score(Y_test,y_pred,average='weighted',zero_division=1)
    recall=recall_score(Y_test,y_pred,average='weighted')
    f1score = f1_score(Y_test,y_pred,average='weighted')    
    
    temp ={  'model':model,
           #'sequence':sequence, 
           'Accurancy':accuracy,
                    'Precision':precision,
                    'Recall':recall,
                    'F1 Score':f1score}
    print(temp)
    return temp




sys.exit()
sequences = [0,1,2,3]

j=3


# ############################################################
# #train first model to then look at model decay in future months 
# ###########################################################
# print('\n\n\n',j)
# dfout,X,y = prep_for_classifier(df,sequence=sequences[j])
# print('shapes')
# print(dfout.shape,X.shape,y.shape)

# ### starting ML ##############
# #split into train/test sets 
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42)
# #nans = X_train.isna().sum()
# #print('max nan',nans.max())
# #now work on random forest 
# rf = RandomForestClassifier(random_state=42)
# rf.fit(X_train,y_train)
# #y_pred_train = rf.predict(X_train)
# #pred_probabilities_train = rf.predict_proba(X_train)

# #y_pred = rf.predict(X_test)

# #pred_probabilities = rf.predict_proba(X_test)

# #prob = pd.DataFrame(pred_probabilities)
# #pred_probabilities = logreg.predict_proba(X_test)
# #prob = pd.DataFrame(pred_probabilities)
# evaluate_model(model=rf, X_test=X_test, Y_test=y_test)

# #####################################
# #look at decay 
# ##############################################
# for j in range(0,4): #@len(sequences)):
#     print('\n\n\n',j)
#     dfout,X,y = prep_for_classifier(df,sequence=sequences[j])
#     print('shapes')
#     print(dfout.shape,X.shape,y.shape)

#     ### starting ML ##############
#     #split into train/test sets 
#     X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42)
#     #nans = X_train.isna().sum()
#     #print('max nan',nans.max())
#     #now work on random forest 
#     #rf = RandomForestClassifier(random_state=42)
#     #rf.fit(X_train,y_train)
#     #y_pred_train = rf.predict(X_train)
#     #pred_probabilities_train = rf.predict_proba(X_train)
    
#     #y_pred = rf.predict(X_test)
    
#     #pred_probabilities = rf.predict_proba(X_test)
    
#     #prob = pd.DataFrame(pred_probabilities)
#     #pred_probabilities = logreg.predict_proba(X_test)
#     #prob = pd.DataFrame(pred_probabilities)
#     try:
#         evaluate_model(model=rf, X_test=X_test, Y_test=y_test)
#     except Exception as e:
#         print(e)

############################################################
###now train 4 models each on the preceding quarter's data. trying with grid search  
###########################################################
for j in range(0,1):
    print('\n\n\n',j)
    dfout,X,y = prep_for_classifier(df,sequence=sequences[j])
    print('shapes')
    print(dfout.shape,X.shape,y.shape)

    ### starting ML ##############
    #split into train/test sets 
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42)
    #nans = X_train.isna().sum()
    #print('max nan',nans.max())
    #now work on random forest 
    rfc = RandomForestClassifier(random_state=42,n_jobs=-1)
    
    #rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True) 
    
    param_grid = { 
        'n_estimators': [100,500, 900],
        'max_depth':[1,10,20],
        'min_samples_split':[2,5,10],
        'min_samples_leaf':[2,8,16]
         #'max_features': ['auto', 'sqrt', 'log2']
    }
    
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
    CV_rfc.fit(X, y)
    print( CV_rfc.best_params_)
    y_pred_test = CV_rfc.predict(X_test)
    #rf.fit(X_train,y_train)
    #y_pred_train = rf.predict(X_train)
    #pred_probabilities_train = rf.predict_proba(X_train)
    
    #y_pred = rf.predict(X_test)
    
    pred_probabilities = CV_rfc.predict_proba(X_test)
    
    prob = pd.DataFrame(pred_probabilities)
    #pred_probabilities = logreg.predict_proba(X_test)
    #prob = pd.DataFrame(pred_probabilities)
    #sys.exit()
    evaluate_model(model=CV_rfc, X_test=X_test, Y_test=y_test)

sys.exit()
sys.exit()

#################################################################
#borrowing some code from my first project 
# feature_importances = list(zip(X_train.columns.tolist(), rf.feature_importances_))
#############################################################################
# df_feature_importances = pd.DataFrame(feature_importances, columns=['Feature', 'Model Weight'])
# filtered_features =df_feature_importances[df_feature_importances['Model Weight']>0] #screening out the lower rated features 
# sorted_features=filtered_features.sort_values(['Model Weight'],ascending=[False])
# top10 = sorted_features[:10] #looking at the top 10 

# top10 = top10.reset_index()
# top10.index += 1 #make the first number 1 since we're looking at a top 10.
# #top10[['Feature','Model Weight']]



for j in range(0,1): #@len(sequences)):
    try:
        print('\n\n\n',j)
        dfout,X,y = prep_for_classifier(df,sequence=sequences[j])
        print('shapes')
        print(dfout.shape,X.shape,y.shape)
    
        ### starting ML ##############
        #split into train/test sets 
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42)
        #nans = X_train.isna().sum()
        #nansy =y_train.isna().sum()
        #print('max nan',nans.max(),nansy.max())
        #now work on random forest 
        SVM = svm.SVC(random_state=42)
        SVM.fit(X_train, y_train)
        #rf = RandomForestClassifier(random_state=42)
        #rf.fit(X_train,y_train)
        #y_pred_train = rf.predict(X_train)
        #pred_probabilities_train = rf.predict_proba(X_train)
        
        #y_pred = rf.predict(X_test)
        
        #pred_probabilities = rf.predict_proba(X_test)
        
        #prob = pd.DataFrame(pred_probabilities)
        #pred_probabilities = logreg.predict_proba(X_test)
        #prob = pd.DataFrame(pred_probabilities)
        evaluate_model(model=SVM, X_test=X_test, Y_test=y_test)
    except Exception as e:
        print('exception of:',e)
        dfout.to_excel('dfout_QC.xlsx')
        sys.exit()

#borrowing some code from my first project 
feature_importances_svm = list(zip(X_train.columns.tolist(), rf.feature_importances_))

df_feature_importances_svm = pd.DataFrame(feature_importances_svm, columns=['Feature', 'Model Weight'])
filtered_features_svm =df_feature_importances_svm[df_feature_importances_svm['Model Weight']>0] #screening out the lower rated features 
sorted_features_svm=filtered_features_svm.sort_values(['Model Weight'],ascending=[False])
top10_svm = sorted_features_svm[:10] #looking at the top 10 

top10_svm = top10_svm.reset_index()
top10_svm.index += 1 #make the first number 1 since we're looking at a top 10.
#top10_svm[['Feature','Model Weight']]


#df['preds'] = np.hstack([y_pred_train, y_pred_test])







