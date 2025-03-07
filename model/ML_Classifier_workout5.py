




#TODO need to clean this up so output is a nice, neat dataframe. 
#TODO delete unncessary commented out code
#make sys args

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
from sklearn.neighbors import KNeighborsClassifier
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
from sklearn.ensemble import VotingClassifier

#import matplotlib.pyplot as plt
#from scipy import stats
pd.options.display.float_format = '{:20,.2f}'.format
import sys 

#TODO make these sys args. all or some? which ones?

#NOTE grid search seems to overfit due to smaller row size. 

#NOTE added to a def parameter for more flexibility
#feature_keep =90
regression_output_count = 300
starting_money = 10000
number_of_tickers_to_invest_in =5


classification_probability = 0.55

#not a sys argument

threshold = 0.005
sequences = [0,1,2,3,4,5]


df = pd.read_excel('ml_data.xlsx')
df = df.set_index('ticker')
dfm = df[['better_than_spy','return','next_rolling62_adjustedclose','rolling62_adjustedclose','sequence']]




bools = ['gics_sector_Consumer Discretionary', 'gics_sector_Consumer Staples', 
 'gics_sector_Energy', 'gics_sector_Financials', 'gics_sector_Health Care',
 'gics_sector_Industrials', 'gics_sector_Information Technology', 
 'gics_sector_Materials', 'gics_sector_Real Estate', 'gics_sector_Utilities']


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

#Note default n of 100. 
def get_top_n_important_features(df_train,model,n= 100):
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
    return new_df

def get_X_y_data_classifier(df,sequence):
    '''INPUT 
    dataframe, sequence (quarter) number
    OUTPUT X and y for machine learning
    '''    
    df = df[df['sequence']==sequence]
    df = df.drop('sequence',axis=1)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1:]
    y=y['better_than_spy'].values
    return X,y


#KEEP
def evaluate_classifier_training(model, X_train, Y_train,sequence): #, category_names=None):
    '''INPUT
    the machine learning model we built earlier
    X_test data
    Y_test data
    sequence number - for reference in metric output
    the category names (pulled from the Y_test dataframe)
    
    OUTPUT 
    dataframe metrics on how well each category performed, including 
    -model name
    -accuracy
    -precision
    -recall
    -the F1 score
    
    the output files are later used in some visuals in the Flask app
       
    '''

    y_pred = model.predict(X_train)
    y_probs = model.predict_proba(X_train)[:,1] #only get if yes since we can derive no

        
    accuracy=accuracy_score(Y_train,y_pred)
    precision=precision_score(Y_train,y_pred,average='weighted',zero_division=1)
    recall=recall_score(Y_train,y_pred,average='weighted')
    f1score = f1_score(Y_train,y_pred,average='weighted')    
    print('\nscores:')
    metricsout ={'test for':['training'],  
        'model':[model],
           'sequence':[sequence], 
           'Accurancy':[accuracy],
                    'Precision':[precision],
                    'Recall':[recall],
                    'F1 Score':[f1score]}
    #TODO remove print statement. curate into 1 larger df for easy comparison
    print(metricsout)
    metricsout = pd.DataFrame(metricsout)
    return y_probs,y_pred ,metricsout

#KEEP
def evaluate_classifier_model(model, X_test, Y_test,sequence): #, category_names=None):
    '''INPUT
    the machine learning model we built earlier
    X_test data
    Y_test data
    sequence number - for reference in metric output
    the category names (pulled from the Y_test dataframe)
    
    OUTPUT 
    dataframe metrics on how well each category performed, including 
    -model name
    -accuracy
    -precision
    -recall
    -the F1 score
    
    the output files are later used in some visuals in the Flask app
       
    '''

    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:,1] #only get if yes since we can derive no

        
    accuracy=accuracy_score(Y_test,y_pred)
    precision=precision_score(Y_test,y_pred,average='weighted',zero_division=1)
    recall=recall_score(Y_test,y_pred,average='weighted')
    f1score = f1_score(Y_test,y_pred,average='weighted')    
    print('\nscores:')
    metricsout ={ 'test for': ['test'] ,
        'model':[model],
           'sequence':[sequence], 
           'Accurancy':[accuracy],
                    'Precision':[precision],
                    'Recall':[recall],
                    'F1 Score':[f1score]}
    #TODO remove print statement. curate into 1 larger df for easy comparison
    print(metricsout)
    metricsout = pd.DataFrame(metricsout)
    return y_probs,y_pred ,metricsout





#TODO refactor this def. 
def train_model_reduced_features(model,X_train_y_train,feature_keep):
    model.fit(X_train,y_train)
 
    try:
        #attempt to refit model with reduced number of features
        top_n = get_top_n_important_features(X_train,model,n= feature_keep)
        X_train_reduced =X_train[top_n]
        #X_test_reduced = X_test[top_n]
        model.fit(X_train_reduced,y_train)
        
    except Exception as e:
        print('exception of:',e)
          
    return model, top_n

#TODO refactor this def. 
# def train_classifier_feature_selection_run_on_next_sequence_v2(df,model,sequence):
#     '''
#     get inputs 
#     train model
#     test model 
#     save metrics
#     run prediction on next sequence get those pics 
#     also save metrics from that for the very end. 

#     '''

#     X,y=get_X_y_data_classifier(df,sequence=s)
#     X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
#     # #train the generic model with all features to find more important ones
#     #modelclf1 =RandomForestClassifier(random_state=42)
#     model.fit(X_train,y_train)
#     evaluate_classifier_model(model, X_test=X_test, Y_test=y_test,sequence=sequence)
#     crossval = (cross_val_score(model, X_train, y_train, cv=5))
#     print('average cross val:',crossval.mean())
    
#     #CONDITIONAL BASED ON MODEL TYPE
#     try:
#         #attempt to refit model with reduced number of features
#         top_n = get_top_n_important_features(X_train,model,n= feature_keep)
#         X_train =X_train[top_n]
#         X_test = X_test[top_n]
#         model.fit(X_train,y_train)
#         #evaluate_classifier_model(model, X_test=X_test, Y_test=y_test,sequence=sequence)
#         print(cross_val_score(model, X_train, y_train, cv=5))
        
#         #use model on NEXT sequence
#         X,y=get_X_y_data_classifier(dfin,sequence=s+1)
#         X =X[top_n]
#         y_probs,y_pred=evaluate_classifier_model(model, X_test=X, Y_test=y,sequence=sequence)
    
#     except Exception:
#         #SVC and KNN can't use the above route.         
#         X,y=get_X_y_data_classifier(dfin,sequence=s+1)
#         y_probs,y_pred=evaluate_classifier_model(model, X_test=X, Y_test=y,sequence=sequence) 

    
#     return y_probs,y_pred


###workout refactoring 


#make universal
#keep the version with the higher precision
#TODO refactor this 
# def train_improve_model(model,sequence):
#     top_n=None
#     prec2=0
#     #split
#     X,y=get_X_y_data_classifier(dfin,sequence=s)
#     #train test
#     X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
    
#     # #train the generic model with all features to find more important ones
#     #modelclf1 =RandomForestClassifier(random_state=42)
#     model.fit(X_train,y_train)
#     y_probs,y_pred,metricsout1=evaluate_classifier_model(model, X_test=X_test, Y_test=y_test,sequence=sequence)
#     crossval = (cross_val_score(model, X_train, y_train, cv=5))
#     print('average cross val:',crossval.mean())
#     prec1 = metricsout1['Precision'].iloc[0]
#     print('prec1:',prec1)
#     #CONDITIONAL BASED ON MODEL TYPE
#     try:
#         model2 = model
#         #attempt to refit model with reduced number of features
#         top_n = get_top_n_important_features(X_train,model,n= feature_keep)
#         X_train =X_train[top_n]
#         X_test = X_test[top_n]
#         model2.fit(X_train,y_train)
#         y_probs,y_pred,metricsout2=evaluate_classifier_model(model2, X_test=X_test, Y_test=y_test,sequence=sequence)
#         crossval2 = (cross_val_score(model2, X_train, y_train, cv=5))
#         #print(cross_val_score(model, X_train, y_train, cv=5))
#         print('average cross val:',crossval2.mean())
#         prec2 = metricsout2['Precision'].iloc[0]
#         print('prec2:',prec2)
#     except Exception:
#         pass #keeps initial training 
#     if prec1>prec2:
#         modelchoice = model
#         finalmetrics =metricsout1
#         finalcrossval = crossval
#     else:
#         modelchoice = model2
#         finalmetrics = metricsout2
#         finalcrossval = crossval2
#     print('final:',modelchoice,finalmetrics['Precision'].iloc[0],finalcrossval.mean() )        
#     return modelchoice,finalmetrics,finalcrossval ,top_n




#######################################################################################
#NEW Spartan DEFS that each only do 1 thing to make cleaner code 
#TODO add input output statements 
#######################################################################################
#preps data for classifier --only need to do once. 
#s = 3 #consolidate s into sequence for clarity 
sequence = 0
dfin = prep_for_classifier(df)

X,y=get_X_y_data_classifier(dfin,sequence=sequence)
#train test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
 

def train_generic_model(model,X_train,y_train):
    model.fit(X_train,y_train)
    return model

def train_feature_reduction_model(model,X_train,y_train,features_keep):
    pass

def train_grid_search_model(model,X_train,y_train):
    pass

def train_voter_model(voters,X_train,y_train):
    combined_model = VotingClassifier(estimators=voters)
    combined_model.fit(X_train,y_train)  
    return combined_model


def cross_validate_model_mean(model,X_train,y_train):
    crossval = (cross_val_score(model, X_train, y_train, cv=5))
    return crossval.mean()

def evaluate_model(model, X_test, Y_test,sequence): #, category_names=None):
    '''INPUT
    the machine learning model we built earlier
    X_test data
    Y_test data
    sequence number - for reference in metric output
    the category names (pulled from the Y_test dataframe)
    
    OUTPUT 
    dataframe metrics on how well each category performed, including 
    -model name
    -accuracy
    -precision
    -recall
    -the F1 score
        
    '''

    y_pred = model.predict(X_test)
    accuracy=accuracy_score(Y_test,y_pred)
    precision=precision_score(Y_test,y_pred,average='weighted',zero_division=1)
    recall=recall_score(Y_test,y_pred,average='weighted')
    f1score = f1_score(Y_test,y_pred,average='weighted')    
    metricsout ={ 'test for': ['test'] ,
        'model':[model],
           'sequence':[sequence], 
           'Accurancy':[accuracy],
                    'Precision':[precision],
                    'Recall':[recall],
                    'F1 Score':[f1score]}
    metricsout = pd.DataFrame(metricsout)
    return metricsout


#if top_n exists, need to include that 
def predict_data(model,sequence,top_n):
    pass
    #CLEAN THIS UP 
    X,y=get_X_y_data_classifier(dfin,sequence=sequence)
    if top_n==None:
        pass
    else:
        X=X[top_n]
    #model.predict(X)
    y_probs,y_pred,metricsout=evaluate_classifier_model(model, X_test=X, Y_test=y,sequence=sequence)
    return y_probs,y_pred




#TODO
'''
since only 500 rows try multiple models 
1) 5 generic models plus voter
2) 5 reducted feature models plus voter (make the same reduced columns)
    try 100,90,80,70,60,50
    plus voter
3) grid search models plus voter 
4) combo of 2 and 3 plus voter
    
run  the 5 models through sequence i, get metrics and cv
if possible, reduce features on the model and rerun, collecting metrics
then test on the next sequence to look at drop off

use precision weights before training the ensemble model
do the same thing on the ensemble model, running on sequence i, get metrics and cv

'''



generic_models = [RandomForestClassifier(random_state=42),
       GradientBoostingClassifier(),
      LogisticRegression(random_state=42),
      SVC(probability=True),
      KNeighborsClassifier()     ]

###########################################################
## ROUND 1 the 5 generic models and a voter
###########################################################
# #split specific sequence into X, y components (all features)


#train 5 generic models plus voter 
  
# mnames =[]
# voters =[]  
# #list of future prediction metrics 
# collectmetrics =[]
# for i in range(5):
#     model = generic_models[i]   
#     model = train_generic_model(model,X_train,y_train)
#     metricsout = evaluate_model(model=model, X_test=X_test, Y_test=y_test,sequence=sequence)
#     #model,metricsouttest,crossval,top_n =train_improve_model(model,sequence)
#     model_name = type(model).__name__
#     mnames.append(model_name)
#     voters.append((model_name,model))    
#     collectmetrics.append(metricsout)
# metricstest_generic = pd.concat(collectmetrics,axis=0)

# voter_model = VotingClassifier(estimators=voters)
# voter_model.fit(X_train,y_train)    
# # # predicting the voter model
# pred_generic = voter_model.predict(X_test)

# print('evaluate combined generic model')
# metricstest_genericCombined = evaluate_model(model=voter_model, X_test=X_test, Y_test=y_test,sequence=0)


###########################################################
## ROUND 2 the 5 models and feature reduction
###########################################################

#run models thru a loop to evaluate individually with feature reduction
# then run thru ensemble 

featurenumbers =[100,90,80,70,60,50]
#features_keep = get_top_n_important_features(X_train,model,n= 100)

featurelist = []
for i in range(len(featurenumbers)):
    model = generic_models[0] #using RandomForestClassifier for feature reduction
    features_keep = get_top_n_important_features(X_train,model,n= 100)
    featurelist.append(features_keep)    
    

#step 1 train 5 generic models plus voter 
#model = models[0]  
mnames =[]
voters =[]  
#list of future prediction metrics 
lf =[]
#using random forest classifier to get feature reduction
collectmetrics =[]
for i in range(5):
    model = generic_models[i]   
    model = train_generic_model(model,X_train,y_train)
    metricsout = evaluate_model(model=model, X_test=X_test, Y_test=y_test,sequence=sequence)
    #model,metricsouttest,crossval,top_n =train_improve_model(model,sequence)
    model_name = type(model).__name__
    mnames.append(model_name)
    voters.append((model_name,model))    
    collectmetrics.append(metricsout)
metricstest_generic = pd.concat(collectmetrics,axis=0)

combined_model = VotingClassifier(estimators=voters)
combined_model.fit(X_train,y_train)    
# # predicting the output on the test dataset
pred_generic = combined_model.predict(X_test)

print('evaluate combined generic model')
metricstest_genericCombined = evaluate_model(model=combined_model, X_test=X_test, Y_test=y_test,sequence=0)

















