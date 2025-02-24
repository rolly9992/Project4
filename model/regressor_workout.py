
'''TODO 
look at both y variables total and per quarter 
both better_than_spy 
and return (or future price) #add a better_return_than_spy?

'''

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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.metrics import make_scorer

import matplotlib.pyplot as plt
from scipy import stats
pd.options.display.float_format = '{:20,.2f}'.format
import sys 
from sklearn import svm

#TODO remove this after DEV 
import os
path =r'/home/milo/Documents/GitHub/Project4' 
os.chdir(path)


##PARAMETERS
#top features to pull out in descending rank order (using above def in each model)
topfeatures = 70
minimumweight = 0.01

#df = pd.read_excel('/home/milo/Documents/Project4/data/machineLearningData.xlsx')
df = pd.read_excel('model/ml_data.xlsx')
df = df.set_index('ticker')
df.shape



#borrowing some code from my first project to save a little time and baking it into a definition
def get_top_n_important_features(df_train,model,n,weight=minimumweight):
    feature_importances = list(zip(df_train.columns.tolist(), model.feature_importances_))

    df_feature_importances = pd.DataFrame(feature_importances, columns=['Feature', 'Model Weight'])
    filtered_features =df_feature_importances[df_feature_importances['Model Weight']>0] #screening out the lower rated features 
    sorted_features=filtered_features.sort_values(['Model Weight'],ascending=[False])
    top_n = sorted_features[:n] #looking at the top 10 
    top_n = top_n.reset_index()
    top_n = top_n.drop('index',axis=1)
    top_n.index += 1 #make the first number 1 since we're looking at a top 10.
    top_n[['Feature','Model Weight']]
    topfeatures = top_n[top_n['Model Weight']>=weight]
    topfeatures = topfeatures['Feature'].tolist()
    
    return topfeatures



#TODO use gradient booster model for classifier and random forest regressor for regressor. 
# retrain model each quarter. use top 30 features. do grid search

#note the sequences are simply shorthand for the 4 quarters we are looking at
sequences = [0,1,2,3] 


def prep_for_regressor(df):


    df = df[df['sequence']!=4]

    #keeping return in ML would mean data leakage. 
    df = df.drop('return',axis=1) #we will use this later. But we would not know this in advance. 
    #pulling out sequence separately. we do not want to normalize this particular column. 
    seqvar = pd.DataFrame(df['sequence'])
    df = df.drop('sequence',axis=1)
    df=df.drop('better_than_spy',axis=1)
    
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
    new_df = pd.concat([dfcat,seqvar,df_nonbool,df_bool],axis=1)

    return new_df
def get_X_y_data_regressor(df,sequence):
    #print(df.shape)
    
    df = df[df['sequence']==sequence]
    #print(df.shape)
    df = df.drop('sequence',axis=1)
           
    X = df #.iloc[:,:-1]
    X = X.drop('next_rolling62_adjustedclose',axis=1)

    #X.to_excel('X.xlsx')    
    #print(X.columns.tolist())
    y = df['next_rolling62_adjustedclose'].values
    y=df['next_rolling62_adjustedclose'].values
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
    return dfmetrics

dfr = prep_for_regressor(df)

#run the data through a regressor model to see what features it considers important
X,y=get_X_y_data_regressor(dfr,sequence=0)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42)

rfr = RandomForestRegressor(random_state=42)
rfr.fit(X_train,y_train)


evaluate_regressor_model(model=rfr, X_test=X_test, y_test=y_test)

top_features= get_top_n_important_features(df_train=X_train,model=rfr,n=topfeatures,weight=0.02)

param_grid = { 
        "n_estimators"      : [10,20,30,50],
        "min_samples_split" : [2,4,8],
        "bootstrap": [True, False],
        }
#reducing features with what the generic model considers the best
X = X[top_features]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42)
grid = GridSearchCV(rfr, param_grid, n_jobs=-1, cv=5)
grid.fit(X_train, y_train)
evaluate_regressor_model(model=grid, X_test=X_test, y_test=y_test)

lmetrics = []
for i in range(4):
    #run the data through a regressor model to see what features it considers important
    X,y=get_X_y_data_regressor(dfr,sequence=i)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42)
    #retraining with new data
    rfr = RandomForestRegressor(random_state=42)
    rfr.fit(X_train,y_train)


    #evaluate_regressor_model(model=rfr, X_test=X_test, y_test=y_test)
    #in case new model has different ranking of features 
    top_features= get_top_n_important_features(df_train=X_train,model=rfr,n=topfeatures,weight=minimumweight)

    param_grid = { 
            "n_estimators"      : [10,20,30,50],
            "min_samples_split" : [2,4,8],
            "bootstrap": [True, False],
            }
    #reducing features with what the generic model considers the best
    X = X[top_features]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42)
    grid = GridSearchCV(rfr, param_grid, n_jobs=-1, cv=5)
    grid.fit(X_train, y_train)
    outmetrics = evaluate_regressor_model(model=grid, X_test=X_test, y_test=y_test)
    outmetrics['sequence']= np.nan
    outmetrics.loc[0,'sequence']=i
    #outmetrics['sequence'].iloc[0]=i
    lmetrics.append(outmetrics)
df_metrics = pd.concat(lmetrics,axis=0)


    








