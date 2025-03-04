

#TODO need to clean this up so output is a nice, neat dataframe. 
#TODO delete unncessary commented out code
#make sys args

import pandas as pd 
pd.options.mode.chained_assignment = None
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm
#from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import max_error, mean_absolute_error, median_absolute_error 
#import seaborn as sns
from sklearn import linear_model

#import matplotlib.pyplot as plt
#from scipy import stats
pd.options.display.float_format = '{:20,.2f}'.format
import sys 
#from sklearn import svm

#TODO make these sys args. all or some? which ones?



#TODO question. is it worth doing feature reduction? 
#NOTE grid search seems to overfit due to smaller row size. 

feature_keep =90
regression_output_count = 300
starting_money = 10000
number_of_tickers_to_invest_in =5


classification_probability = 0.55

#not a sys argument

threshold = 0.005
sequences = [0,1,2,3,4,5]


df = pd.read_excel('ml_data.xlsx')
df = df.set_index('ticker')

bools = ['gics_sector_Consumer Discretionary', 'gics_sector_Consumer Staples', 
 'gics_sector_Energy', 'gics_sector_Financials', 'gics_sector_Health Care',
 'gics_sector_Industrials', 'gics_sector_Information Technology', 
 'gics_sector_Materials', 'gics_sector_Real Estate', 'gics_sector_Utilities']



def get_top_n_important_features(df_train,model,n= feature_keep):
    #TODO verify this try except covers models that have feature importances and those that don't
    try: #was using this for previous models like Random Forest Classifier
        feature_importances = list(zip(df_train.columns.tolist(), model.feature_importances_))
    except:
        topcoeff = model.coef_[0]
        #topcoeff = topcoeff.T
        #topcoeff = flatten_list(topmodel)
        feature_importances = list(zip(df_train.columns.tolist(), topcoeff))
    #feature_importances = list(zip(df_train.columns.tolist(), model.feature_importances_))

    df_feature_importances = pd.DataFrame(feature_importances, columns=['Feature', 'Model Weight'])
    df_feature_importances=df_feature_importances.sort_values(['Model Weight'],ascending=[False])    
    #filtered_features =df_feature_importances[:n] #df_feature_importances['Model Weight']>=weight] #screening out the lower rated features 

    top_n = df_feature_importances[:n]  #looking at the top 10 

    top_n = top_n.reset_index()
    top_n = top_n.drop('index',axis=1)
    top_n.index += 1 #make the first number 1 since we're looking at a top n. 
    top_n[['Feature','Model Weight']]
    features_keep = top_n['Feature'].tolist()
    
    return features_keep



def spy_returns_by_quarter():
    #TODO refactor location when running from terminal
    df = pd.read_excel('../wrangling/Consolidated_TimeSeries_Data.xlsx')
    spy = df[df['ticker']=='SPY']
    startingmoney = starting_money
    value = startingmoney
    l = []
    #skip first and last quarter. 
    #first set of data used as input data for first model. Last is used to test the preceding quarter 
    for i in range(0,5):
        
        temp = pd.DataFrame({'sequence':[i],
                            'Spy Return':[spy['return'].iloc[i]]
                            ,'new value': [value + value * spy['return'].iloc[i]]}
                            )
        value = value + value * spy['return'].iloc[i]
        l.append(temp)
    SPY= pd.concat(l)

    #print('SPY by quarter:',dfout)
    #print('SPY overall return:', value/startingmoney-1 )
    return SPY
SPY = spy_returns_by_quarter()
df = df.merge(SPY[['sequence','Spy Return']],on='sequence',how='left')
df['return diff']=df['return'] - df['Spy Return']
dfm = df[['better_than_spy','return','next_rolling62_adjustedclose','rolling62_adjustedclose','sequence']]

dfcols = df.columns.tolist()

#df.to_excel('df_with_SPY.xlsx')

################################################################
#### REGRESSOR DEFINITIONS 
################################################################

def prep_for_regressor(df):
    df = df[df['sequence']!=4]
    df = df.copy()


    df = df.drop('return',axis=1) 
    df = df.drop('Spy Return',axis=1)
    
    #pulling out sequence separately. we do not want to normalize this particular column. 
    seqvar = pd.DataFrame(df['sequence'])
    df = df.drop('sequence',axis=1)
    df=df.drop('better_than_spy',axis=1)
    df=df.drop('rolling62_adjustedclose',axis=1)    
    df=df.drop('next_rolling62_adjustedclose',axis=1)    
    returndiffs = df['return diff']
    df = df.drop('return diff',axis=1)
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
    df_return_diff = pd.DataFrame(returndiffs)
    df_return_diff.columns=['return diff']
    new_df = pd.concat([dfcat,seqvar,df_nonbool,df_bool,df_return_diff],axis=1)
    #TODO remove after dev
    new_df.to_excel('new_df.xlsx')
    
    return new_df

def get_X_y_data_regressor(df,sequence):
    #print(df.shape)
    print('sequence:',sequence)
    df = df[df['sequence']==sequence]
    #print(df.shape)
    df = df.drop('sequence',axis=1)
           
    X = df #.iloc[:,:-1]
    #X = X.drop('next_rolling62_adjustedclose',axis=1)
    X = X.drop('return diff',axis=1)
    #X = X.drop('rolling62_adjustedclose',axis=1)
    #X = X.drop('return diff',axis=1)    

    #X.to_excel('X.xlsx')    
    #print(X.columns.tolist())
    y = df['return diff'].values
    #y=df['next_rolling62_adjustedclose'].values
    #y=y['better_than_spy'].values
    return X,y
    


def evaluate_regressor_model(model, X_test, y_test): #, category_names=None):
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
    #TODO remove prints after DEV     
    #source:https://stackoverflow.com/questions/50789508/random-forest-regression-how-do-i-analyse-its-performance-python-sklearn
    
    print('\nMean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    mape = np.mean(np.abs((y_test - y_pred) / np.abs(y_test)))
    print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))
    print('Accuracy:', round(100*(1 - mape), 2))
    print('Score:',r2_score(y_test, y_pred))
    roundmetrics = { 'nMean Absolute Error (MAE)': [metrics.mean_absolute_error(y_test, y_pred)],
        'Mean Squared Error (MSE)': [metrics.mean_squared_error(y_test, y_pred)],
        'Root Mean Squared Error (RMSE)': [np.sqrt(metrics.mean_squared_error(y_test, y_pred))],
        'Mean Absolute Percentage Error (MAPE)': [round(mape * 100, 2)],
        'Accuracy': [round(100*(1 - mape), 2)],
        'Score':[r2_score(y_test, y_pred) ]  }
    dfmetrics = pd.DataFrame(roundmetrics)
    return y_pred,dfmetrics



def train_regressor_feature_selection_run_on_next_sequence(df,model,sequence):

    X,y=get_X_y_data_regressor(dfin,sequence=s)
    
    ######RANDOM FOREST CLASSIFIER
    ###########################################
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
    # #train the generic model with all features to find more important ones
    #modelclf1 =RandomForestClassifier(random_state=42)
    model.fit(X_train,y_train)
    evaluate_regressor_model(model, X_test=X_test, y_test=y_test)
    print(cross_val_score(model, X_train, y_train, cv=5))
    #CONDITIONAL BASED ON MODEL TYPE
    try:
        #attempt to refit model with reduced number of features
        top_n = get_top_n_important_features(X_train,model,n= feature_keep)
        X_train =X_train[top_n]
        X_test = X_test[top_n]
        model.fit(X_train,y_train)
        evaluate_regressor_model(model, X_test=X_test, Y_test=y_test)
        print(cross_val_score(model, X_train, y_train, cv=5))
        #use model on next sequence
        X,y=get_X_y_data_regressor(dfin,sequence=s+1)
        # #split into train/test sets 
        
        X =X[top_n]
        ynext=evaluate_regressor_model(model, X_test=X, Y_test=y)
        
    except Exception:
        #print('excception of:',e)
        
        #SVC and KNN can't use the above route.         
        X,y=get_X_y_data_regressor(dfin,sequence=s+1)
        
        # #split into train/test sets 
        #X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
        ynext=evaluate_regressor_model(model, X_test=X, y_test=y) 

    
    return ynext





# ##############################################
s = 0
dfin = prep_for_regressor(df)
# # #split specific sequence into X, y components (all features)
models =[svm.SVR(), linear_model.Lasso(alpha=1.0,tol=1e-2),RandomForestRegressor(random_state=42)]

youts = []
mnames =[]
X,y=get_X_y_data_regressor(dfin,sequence=s)
Xcols = X.columns.tolist()
# for i in range(1,2): #len(models)):
#     yout = train_regressor_feature_selection_run_on_next_sequence(df=dfin,model=models[i],sequence=s)    
#     model_name = type(models[i]).__name__
#     print(model_name)
#     X[model_name]=yout
#     mnames.append(model_name)
    
# X['AVG_Prob']=(X[mnames[0]]+X[mnames[1]]+X[mnames[2]]+X[mnames[3]]+X[mnames[4]])/5
# X = X.merge(dfm[dfm['sequence']==s+1],left_index=True,right_index=True,how = 'left')
 

#sys.exit()
i=2
X,y=get_X_y_data_regressor(dfin,sequence=s)
model = models[i]
######RANDOM FOREST CLASSIFIER
###########################################

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
# #train the generic model with all features to find more important ones
#modelclf1 =RandomForestClassifier(random_state=42)
model.fit(X_train,y_train)
evaluate_regressor_model(model, X_test=X_test, y_test=y_test)
print(cross_val_score(model, X_train, y_train, cv=5))
#CONDITIONAL BASED ON MODEL TYPE
# try:
#     #attempt to refit model with reduced number of features
#     top_n = get_top_n_important_features(X_train,model,n= feature_keep)
#     X_train =X_train[top_n]
#     X_test = X_test[top_n]
#     model.fit(X_train,y_train)
#     evaluate_regressor_model(model, X_test=X_test, Y_test=y_test)
#     print(cross_val_score(model, X_train, y_train, cv=5))
#     #use model on next sequence
#     X,y=get_X_y_data_regressor(dfin,sequence=s+1)
#     # #split into train/test sets 
    
#     X =X[top_n]
#     ynext=evaluate_regressor_model(model, X_test=X, Y_test=y)
    
# except Exception as e:
       
#     #SVC and KNN can't use the above route.         
#     X,y=get_X_y_data_regressor(dfin,sequence=s+1)
    
#     # #split into train/test sets 
#     #X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
#     ynext=evaluate_regressor_model(model, X_test=X, y_test=y) 

#     model_name = type(models[i]).__name__
#     print(model_name)

