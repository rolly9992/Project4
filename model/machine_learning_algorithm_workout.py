

#TODO need to clean this up so output is a nice, neat dataframe. 
#TODO delete unncessary commented out code
#make sys args

import pandas as pd 
pd.options.mode.chained_assignment = None
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.linear_model import LinearRegression
#from sklearn import preprocessing, svm
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


feature_number =90
regression_output_count = 300
starting_money = 10000
number_of_tickers_to_invest_in =5
classification_probability = 0.55

#not a sys argument

weight_threshold = 0.005
sequences = [0,1,2,3,4]


df = pd.read_excel('ml_data.xlsx')
df = df.set_index('ticker')
dfm = df[['better_than_spy','return','next_rolling62_adjustedclose','rolling62_adjustedclose','sequence']]


def spy_returns_by_quarter():
    #TODO refactor location when running from terminal
    df = pd.read_excel('../wrangling/Consolidated_TimeSeries_Data.xlsx')
    spy = df[df['ticker']=='SPY']
    startingmoney = starting_money
    value = startingmoney
    l = []
    #skip first quarter. only used as data for first model. 
    for i in range(1,5):
        
        temp = pd.DataFrame({'quarter':[i],
                            'Spy Return':[spy['return'].iloc[i]]
                            ,'new value': [value + value * spy['return'].iloc[i]]}
                            )
        value = value + value * spy['return'].iloc[i]
        l.append(temp)
    dfout = pd.concat(l)

    #print('SPY by quarter:',dfout)
    #print('SPY overall return:', value/startingmoney-1 )
    return dfout
SPY = spy_returns_by_quarter()

def get_top_n_important_features(df_train,model,n,weight):
    feature_importances = list(zip(df_train.columns.tolist(), model.feature_importances_))

    df_feature_importances = pd.DataFrame(feature_importances, columns=['Feature', 'Model Weight'])
    filtered_features =df_feature_importances[df_feature_importances['Model Weight']>=weight] #screening out the lower rated features 
    sorted_features=filtered_features.sort_values(['Model Weight'],ascending=[False])
    top_n = sorted_features[:n] #looking at the top 10 

    top_n = top_n.reset_index()
    top_n = top_n.drop('index',axis=1)
    top_n.index += 1 #make the first number 1 since we're looking at a top 10.
    top_n[['Feature','Model Weight']]
    
    return top_n


################################################################
#### CLASSIFIER DEFINITIONS 
################################################################
def prep_for_classifier(df):
    df = df[df['sequence']!=5]

    #keeping return in ML would mean data leakage. 
    df = df.drop('return',axis=1) #we will use this later. But we would not know this in advance. 
    #pulling out sequence separately. we do not want to normalize this particular column. 
    seqvar = pd.DataFrame(df['sequence'])
    df = df.drop('sequence',axis=1)
    df = df.drop('next_rolling62_adjustedclose',axis=1) #we will use this for the regressor models 
    
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
    #in case the nan sector field does not exist in each sequence or all blanks. 
    #we need to remove 1 of the sector dummy cols anyway. This is the best one. 
    try:
        new_df = new_df.drop('gics_sector_nan',axis=1)
    except Exception:
        pass
    
    new_df.to_excel('new_df.xlsx')
    return new_df

def get_X_y_data_classifier(df,sequence):
    #print(df.shape)
    
    df = df[df['sequence']==sequence]
    #print(df.shape)
    df = df.drop('sequence',axis=1)
    X = df.iloc[:,:-1]

    #X.to_excel('X.xlsx')    
    #print(X.columns.tolist())
    y = df.iloc[:,-1:]
    y=y['better_than_spy'].values
    return X,y



def train_classifier_model(df,sequence):
    
    #get dataset for classifier with all but the last sequence
    dfin = prep_for_classifier(df)
    #split specific sequence into X, y components (all features)
    X,y=get_X_y_data_classifier(dfin,sequence=sequence)
    
    print('shapes')
    print(dfin.shape,X.shape,y.shape)
    
    
    #split into train/test sets 
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.30,random_state=42,shuffle=True)
    #train the generic model with all features to find more important ones
    clf =RandomForestClassifier(random_state=42)
    clf.fit(X_train,y_train)
    top_features_df =get_top_n_important_features(X_train,model=clf,n=feature_number,weight=weight_threshold)
    top_features_df.to_excel('top_features.xlsx')       
    top_features = top_features_df['Feature'].tolist()    
    print('number of features determined by grid search:',len(top_features))
    #TODO remove the save after dev


    #reducing features to the filtered set. 
    X = X[top_features]
    print('reduced shape:',X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.30,random_state=42,shuffle=True)
    print('reduced shapes:',X_train.shape,X_test.shape)
    #retrain the generic model with the filtered X set
    clf.fit(X_train,y_train)    

    
    print('starting grid search')
    '''REFERENCE: categories to use in params:['bootstrap', 'ccp_alpha', 
    'class_weight', 'criterion', 'max_depth', 
    'max_features', 'max_leaf_nodes', 'max_samples', 'min_impurity_decrease',
    'min_samples_leaf', 'min_samples_split', 'min_weight_fraction_leaf', 
    'monotonic_cst', 'n_estimators', 'n_jobs', 'oob_score', 'random_state', 
    'verbose', 'warm_start'] '''
    parameters = {
    'n_estimators': [100,200,300,400,500],
     'max_depth': [2,5, 10,20,None],
    #'min_samples_split': [2, 5, 10],
    #'min_samples_leaf': [1, 2, 4], #trying to reduce computational cost, esp. since retraining... 
    'oob_score':[True],
    #'max_features':[None,50,75,100]
        }
    
    '''Options for what to optimize: {'neg_mean_squared_log_error', 
    'normalized_mutual_info_score', 'average_precision', 'jaccard_macro', 
    'f1_samples', 'recall_samples', 'jaccard_micro', 'neg_mean_absolute_percentage_error', 
    'recall_macro', 'neg_root_mean_squared_error', 'explained_variance', 
    'positive_likelihood_ratio', 'r2', 'neg_mean_absolute_error', 
    'd2_absolute_error_score', 'neg_mean_gamma_deviance', 'neg_brier_score',
    'jaccard_samples', 'recall_micro', 'neg_mean_poisson_deviance',
    'precision_micro', 'roc_auc_ovr', 'recall_weighted', 'roc_auc_ovo',
    'roc_auc_ovr_weighted', 'top_k_accuracy', 'roc_auc', 'neg_mean_squared_error',
    'f1_micro', 'adjusted_mutual_info_score', 'precision_samples', 
    'recall', 'balanced_accuracy', 'accuracy', 'neg_negative_likelihood_ratio',
    'homogeneity_score', 'adjusted_rand_score', 'v_measure_score', 
    'neg_max_error', 'precision_macro', 'rand_score', 'fowlkes_mallows_score',
    'f1_macro', 'neg_log_loss', 'precision', 'jaccard_weighted', 'f1', 
    'neg_median_absolute_error', 'mutual_info_score', 'roc_auc_ovo_weighted',
    'f1_weighted', 'completeness_score', 'precision_weighted',
    'matthews_corrcoef', 'jaccard', 'neg_root_mean_squared_log_error'}, 
    '''
    #trying to find the best precision here
    #tried precision, avg precision roc_auc
    clfgrid = GridSearchCV(clf, parameters,scoring='precision',cv=5, n_jobs=-1)
    
    clfgrid.fit(X_train, y_train)

    best_params = clfgrid.best_params_
    print('best params:',best_params)
    print('best score:',clfgrid.best_score_)
    #converting the clf.cv_results to dataframe
    df_results =pd.DataFrame.from_dict(clfgrid.cv_results_)
    df_results.to_excel('grid_results.xlsx')
    
    # Create a new instance of the model with the best parameters
    #@improved_clf = RandomForestClassifier(**best_params,random_state=42)
    
    # Fit the model on the training data
    #improved_clf.fit(X_train, y_train)
    #y_pred = clfgrid.predict(X_test)
    #print('avg ypred',y_pred.mean())
    evaluate_classifier_model(model=clfgrid, X_test=X_test, Y_test=y_test)
    return clfgrid,top_features




def evaluate_classifier_model(model, X_test, Y_test): #, category_names=None):
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

        
    accuracy=accuracy_score(Y_test,y_pred)
    precision=precision_score(Y_test,y_pred,average='weighted',zero_division=1)
    recall=recall_score(Y_test,y_pred,average='weighted')
    f1score = f1_score(Y_test,y_pred,average='weighted')    
    print('\nscores:')
    temp ={  'model':model,
           #'sequence':sequence, 
           'Accurancy':accuracy,
                    'Precision':precision,
                    'Recall':recall,
                    'F1 Score':f1score}
    print(temp)
    return temp


################################################################
#### REGRESSOR DEFINITIONS 
################################################################
def prep_for_regressor(df):
    df = df[df['sequence']!=4]
    df = df.copy()

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



#############################################################

#get model to use for predictions in the following quarter training on particular sequence

def create_ticker_subset_from_ML(sequence,num_tickers_invest=number_of_tickers_to_invest_in ,df=df):
    '''NOTE TO SELF 
    this trains 2 models using the previous quarter. 
    in real life we would not have the output to test. 
    but since we are using an algorithm we can use test to evaluate as we go 
    without a human picking any specific stocks  
    the output would be a list of suggested tickers based on the outcome of the model 
    run the quarter AFTER the training. 
    '''
    #CLASSIFIER TRAINING
    #train on sequence n
    clfmodel,top_features =train_classifier_model(df=df,sequence=sequence)
    #use model on the ENTIRE NEXT QUARTER (unseen data)
    #get df ready for classifier    
    newdf = prep_for_classifier(df)
    #run on sequence n+1
    X_,y_ =get_X_y_data_classifier(df=newdf,sequence=sequence+1)
    X_ = X_[top_features]
    #throwing in the next sequence data to evaluate (since we have that data)
    #print('eval of next quarter')
    evaluate_classifier_model(clfmodel, X_, y_)
    #using the prediction to look at probs 
    y_probs = clfmodel.predict_proba(X_)
    probs_df = pd.DataFrame(y_probs)
    #print('probs df column names:',probs_df.columns.tolist())
    y_pred = clfmodel.predict(X_)
    y_pred_avg = y_pred.mean()
    print('clf avg pred:',y_pred_avg)
    X_.reset_index(inplace=True)
    Xnew= pd.concat([X_.reset_index(drop=True),probs_df[0].reset_index(drop=True)],axis =1)
    Xnew.rename(columns={0:'probs'},inplace=True)
    StockFilter1 = Xnew[Xnew['probs']>=classification_probability]
    StockFilter1 = StockFilter1[['ticker','probs']]
    StockFilter1.sort_values('probs',ascending=False,inplace=True)
    ##### end of first filter (the classifier )
    #####################################################################
    #REGRESSION TRAINING
    new_df = prep_for_regressor(df)
    #NOTE normal linear regression largedrop off 
  
    #training on sequence 0
    lasso= linear_model.Lasso(alpha=1.0,tol=1e-2) #LinearRegression()
    X,y = get_X_y_data_regressor(df=new_df,sequence=sequence)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.30,random_state=42,shuffle=True)
    
    param_lasso = {
        'alpha': np.logspace(-4, 4, 50),
        'selection': ['cyclic', 'random'],  # Selection method for coordinate descent
        'max_iter': [ 200, 500,1000]  # Maximum iterations
    }
    lassogrid = GridSearchCV(lasso, param_lasso)
    lassogrid.fit(X_train, y_train)
    
    #lassogrid.fit(X_train,y_train)
    y_pred,dfmetrics=evaluate_regressor_model(model=lassogrid, X_test=X_test, y_test=y_test)
    
   
    X_,y_ = get_X_y_data_regressor(df=new_df,sequence=1)
    X_train, X_test, y_train, y_test = train_test_split(X_,y_,test_size=.30,random_state=42,shuffle=True)
    #lr.fit(X_train,y_train)
    y_pred,dfmetrics=evaluate_regressor_model(model=lassogrid, X_test=X_, y_test=y_)
    y_pred_df = pd.DataFrame(y_pred)
    avg_reg_pred = y_pred_df.mean()
    print('avg_reg__pred:',avg_reg_pred)
    X_.reset_index(inplace=True)
    Xnew= pd.concat([X_.reset_index(drop=True),y_pred_df[0].reset_index(drop=True)],axis =1)
    Xnew.rename(columns={0:'reg_pred'},inplace=True)
    Xnew =Xnew[['ticker','reg_pred']].sort_values('reg_pred',ascending=False)
    StockFilter2 = Xnew[:regression_output_count]
   
    StockFilter3 = StockFilter1.merge(StockFilter2,on='ticker',how='inner'  )
    StockFilter3 = StockFilter3.sort_values('reg_pred',ascending=False)
    StockFilter3['start_sequence']=sequence
    StockFilter3['end sequence']= sequence+1
    dfm = df[['better_than_spy','return','next_rolling62_adjustedclose','rolling62_adjustedclose','sequence']]
    StockFilter3 = StockFilter3.merge(dfm[dfm['sequence']==sequence],left_on='ticker',right_index=True,how='left')
    
    if len(StockFilter3)< num_tickers_invest: 
        StockFilter3['starting_money']= starting_money/len(StockFilter3)
        StockFilter3['ending_money']= StockFilter3['starting_money']+StockFilter3['starting_money']* StockFilter3['return']
    else:        
        StockFilter3 = StockFilter3[:num_tickers_invest]
        StockFilter3['starting_money']= starting_money/num_tickers_invest
        StockFilter3['ending_money']= StockFilter3['starting_money']+StockFilter3['starting_money']* StockFilter3['return']
          
    return StockFilter3

#TODO refactor this. the start monies diverge after the first month. 
#TODO make a cumulative to account for multiple quarters as the 2 paths could diverge 
def compare_Spy_vs_Spy(StockFilter,sequence,Spystart=starting_money,StockStart=starting_money,SPY=SPY):
    #SPY_= SPY[SPY['return']==sequence]
    #SPYstart = starting_money
    SPYend = SPY['Spy Return'].iloc[sequence]* Spystart +Spystart
    Stockstart = StockFilter['starting_money'].sum()
    Stockend = StockFilter['ending_money'].sum()
    print('SPY results')
    print({'SPY Start':Spystart,
           'SPY End':Spystart+SPYend,
          'SPY return':SPY['Spy Return'].iloc[sequence]})
    print('stock results')
    print({'stock starting money':starting_money,
           'stock ending money':Stockend,
           'stock return':(Stockend/Stockstart)-1
           })
    SPYreturn = SPY['Spy Return'].iloc[sequence]
    Stockreturn = (Stockend/Stockstart)-1
    return SPYend, Stockend, SPYreturn, Stockreturn 
    
Spystart = starting_money
StockStart = starting_money
for i in range(0,2):
    print('\n\n\nstarting sequence ',i,' to ',i+1)
    #TODO add beginning and ending avg 62 adjusted close, plus end return 
    StockFilter = create_ticker_subset_from_ML(sequence=i)    
    SPYend, Stockend, SPYreturn, Stockreturn = compare_Spy_vs_Spy(StockFilter=StockFilter,sequence=i+1,Spystart=Spystart,StockStart=StockStart)
    Spystart = SPYend
    StockStart = Stockend



