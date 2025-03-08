




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
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
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
    SPY= pd.concat(l,axis=1)

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
    dfcat=dfcat.drop(columns=cat_vars,axis=1)
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




#######################################################################################
#NEW Spartan DEFS that each only do 1 thing to make cleaner code 
#TODO add input output statements 
#######################################################################################
#preps data for classifier --only need to do once. 
s = 3 #consolidate s into sequence for clarity 

#only need to do this once. 
dfin = prep_for_classifier(df)






sequence = 0
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
## STEP 1 use the 5 generic models and a voter
###########################################################
# #split specific sequence into X, y components (all features)

def train_evaluate_generic_models(dfin,sequence):
    
    X,y=get_X_y_data_classifier(dfin,sequence=sequence)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
    #train 5 generic models plus voter 
    mnames =[]
    voters =[]  
    #list of future prediction metrics 
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
    
    voter_model = VotingClassifier(estimators=voters)
    voter_model.fit(X_train,y_train)    
    # # predicting the voter model
    #pred_generic = voter_model.predict(X_test)
    
    #print('evaluate combined generic model')
    metricstest_genericCombined = evaluate_model(model=voter_model, X_test=X_test, Y_test=y_test,sequence=sequence)
    metricstest_genericCombined['model']='generic_voter_model' 
    allmodelmetrics = pd.concat([metricstest_generic,metricstest_genericCombined],axis=0)
    return allmodelmetrics

#THIS WORKS TO GET GENERIC METRICS OUT 
l = []
for i in range(5):
    genericout = train_evaluate_generic_models(dfin,sequence=i)
    l.append(genericout)
genericmetricsdf = pd.concat(l,axis=0)
genericmetricsdf.to_excel('generic_metrics.xlsx')

#TODO use Jupyter notebooks to investigate combos 


# ############################################################################################
# ## STEP 2 explore what the best number of features might be and test on each of the 5 models
# ############################################################################################

# #run models thru a loop to evaluate individually with feature reduction
# # then run thru ensemble 


#sequence = 0

def explore_different_number_of_features_used(dfin,sequence):
    X,y=get_X_y_data_classifier(dfin,sequence=sequence)
    #train test
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
    featurenumbers =[100,90,80,70,60,50]
    #features_keep = get_top_n_important_features(X_train,model,n= 100)
    
    featurelist = []
    for i in range(len(featurenumbers)):
        model = generic_models[0] #using RandomForestClassifier for feature reduction
        model = train_generic_model(model,X_train,y_train)
        features_keep = get_top_n_important_features(X_train,model,n= featurenumbers[i])
        featurelist.append(features_keep)    
   
    #step 1 train 5 generic models plus voter 
    #model = models[0]  
    mnames =[]
    voters =[]  
    #list of future prediction metrics 
    
    #using random forest classifier to get feature reduction
    collectmetrics =[]
    X,y=get_X_y_data_classifier(dfin,sequence=sequence)
    
    
    for j in range(len(featurelist)):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
        X_train=X_train[featurelist[j]]
        X_test=X_test[featurelist[j]] 
        
        
        for i in range(5):
            model = generic_models[i]      
            
            model = train_generic_model(model,X_train,y_train)
            metricsout = evaluate_model(model=model, X_test=X_test, Y_test=y_test,sequence=sequence)
            metricsout['numberoffeatures']=len(featurelist[j])
            #model,metricsouttest,crossval,top_n =train_improve_model(model,sequence)
            model_name = type(model).__name__
            mnames.append(model_name)
            voters.append((model_name,model))    
            collectmetrics.append(metricsout)
    metricstest_feature_reduction = pd.concat(collectmetrics,axis=0)
    return metricstest_feature_reduction

#This works 
# l = []
# for i in range(5):
#     featurestestingout = explore_different_number_of_features_used(dfin,sequence=i)
    
#     featurestestingout['sequence']= i
#     l.append(featurestestingout)
# featurestestingmetricsdf = pd.concat(l,axis=0)
# featurestestingmetricsdf.to_excel('features_testing_metrics.xlsx')

#TODO use jupyter notebooks to investigate combos. 




# ####################################################################################
### THIS STEP MAY NOT BE NECESSARY... or replace with using 5 simple models with voting 
# ## ROUND 3 use models with 90 features, then a voter model with the same 90 features
# ####################################################################################


# sequence = 0
# X,y=get_X_y_data_classifier(dfin,sequence=sequence)
# #train test
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
 

# #NOTE 90 appeared to be the best number of features to use 
# featurenumbers=[90]
# #features_keep = get_top_n_important_features(X_train,model,n= 100)

# featurelist = []
# for i in range(len(featurenumbers)):
#     model = generic_models[0] #using RandomForestClassifier for feature reduction
#     model = train_generic_model(model,X_train,y_train)
#     features_keep = get_top_n_important_features(X_train,model,n= featurenumbers[i])
#     featurelist.append(features_keep)    
    

# #step 1 train 5 generic models plus voter 
# #model = models[0]  
# mnames =[]
# voters =[]  
# #list of future prediction metrics 
# lf =[]
# #using random forest classifier to get feature reduction
# collectmetrics =[]


# X,y=get_X_y_data_classifier(dfin,sequence=sequence)
# #train test
# broadermetrics =[]
# for j in range(len(featurelist)):
#     X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
#     X_train=X_train[featurelist[j]]
#     X_test=X_test[featurelist[j]] 
    
    
#     for i in range(5):
#         model = generic_models[i]      
        
#         model = train_generic_model(model,X_train,y_train)
#         metricsout = evaluate_model(model=model, X_test=X_test, Y_test=y_test,sequence=sequence)
#         metricsout['numberoffeatures']=len(featurelist[j])
#         #model,metricsouttest,crossval,top_n =train_improve_model(model,sequence)
#         model_name = type(model).__name__
#         mnames.append(model_name)
#         voters.append((model_name,model))    
#         collectmetrics.append(metricsout)
# metricstest_feature_reduction = pd.concat(collectmetrics,axis=1)
# #metricstest_feature_reduction.to_excel('metricstest_feature_reduction.xlsx',index=None)
 
# #sys.exit()
# #stick this inside each loop for features or just use the best outcome round?
# voter_model2 = VotingClassifier(estimators=voters)
# voter_model2.fit(X_train,y_train)    
# # # predicting the output on the test dataset
# pred_generic = voter_model2.predict(X_test)

# print('evaluate combined generic model')
# metricstest_reducedfeatureCombined = evaluate_model(model=voter_model2, X_test=X_test, Y_test=y_test,sequence=0)



# ####################################################################################
# ## ROUND 4 use models with grid search then a voter model 
# ####################################################################################

# #TODO get more parameters for other models. may need to do a random grid search
# # then save the outputs. 
# #FOR DEV MAY WANT TO REDUCE COMBOS OR USE RANDOM GRID SEARCH.5 models will take a while 
# paramdict ={
#     'GBC_parameters': {
#     "loss":["exponential", "log_loss"],
#     "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    
#     #"min_samples_split": np.linspace(1, 10, 10),
#     #"min_samples_leaf": np.linspace(0.0, 1.0, 10),
#     "max_depth":[3,5,8],
#     #"max_features":["log2","sqrt"],
#     #"criterion": ['squared_error', 'friedman_mse'],
#     #"subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
#     "n_estimators":[10]
#     },
#     'RF_parameters':{
#     'n_estimators': [200, 700],
#     #'max_depth':[10,30,50],
#     #'min_samples_split':[2,3,5],
#     #'min_samples_leaf':[3,5],
#     'max_features': [90,110]
#       }
#     ,'LR_parameters':{'penalty':['l2'], 
#               'C':[1, 10, 100, 1000]}
#     ,'SVC_parameters':{'C': [0.1, 1, 10, 100, 1000],  
#               'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
#               'kernel': ['rbf']  }
#     #Valid parameters are: ['algorithm', 'leaf_size', 'metric', 'metric_params', 'n_jobs', 'n_neighbors', 'p', 'weights'].
#     ,'KNN_parameters': { 'n_neighbors': [3, 5, 7, 9, 11]     }
    
#     }

# def easy_loop(a):
    
#     generic_models = [  GradientBoostingClassifier(),
#         RandomForestClassifier(random_state=42),
#          LogisticRegression(random_state=42,max_iter=1000),
#           SVC(probability=True),
#           KNeighborsClassifier()     ]
#     paramchoices = {0: 'GBC_parameters', 1:'RF_parameters'
#                     ,2:'LR_parameters',
#                     3:'SVC_parameters',4:'KNN_parameters'}
#     return generic_models[a],paramchoices[a]


# def do_grid_search(model,parameters,sequence):
#     #sequence = 0
#     X,y=get_X_y_data_classifier(dfin,sequence=sequence)
#     #train test
#     X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
#     #note: adding this using an item OUTSIDE the definition.
#     X_train =X_train[featurelist[0]]
#     X_test = X_test[featurelist[0]]
#     #print('starting gbc grid search...')
#     precision_scorer = make_scorer(precision_score, zero_division=0) 
#     gbc_grid = GridSearchCV(model, parameters, scoring=precision_scorer, cv=5, n_jobs=-1)
#     gbc_grid.fit(X_train, y_train)
#     df_cv_results=pd.DataFrame.from_dict(gbc_grid.cv_results_)
#     #print('finished grid search')
#     gridmetricsout = evaluate_model(model=gbc_grid, X_test=X_test, Y_test=y_test,sequence=sequence)
#     return df_cv_results,gridmetricsout
    



# l=[]
# gridvotermetrics=[]
# #outer loop for sequences
# for j in range(5):
#     mnames =[]
#     voters =[]  
#     for i in range(5):
#         l1,l2 = easy_loop(i)
#         model_name = type(l1).__name__
#         print('sequence:',j,'  model:',model_name)
#         cv_results,gridmetrics = do_grid_search(model=l1,parameters=paramdict[l2],sequence=j)
#         mnames.append(model_name)
#         voters.append((model_name,model))    
        
#         gridmetrics['model name']=model_name
#         #to do redefine sequence as outer loop
#         gridmetrics['sequence']=j
#         l.append(gridmetrics)
#     voter_gridmodel = VotingClassifier(estimators=voters)
#     X,y=get_X_y_data_classifier(dfin,sequence=j)
#     #train test
#     X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
#     voter_gridmodel.fit(X_train,y_train)    
#     # # predicting the output on the test dataset
#     pred_generic = voter_gridmodel.predict(X_test)
#     print('evaluate combined generic model')
#     metricstest_gridvoter = evaluate_model(model=voter_gridmodel, X_test=X_test, Y_test=y_test,sequence=j)
#     gridvotermetrics.append(metricstest_gridvoter)
        

# multimodelgridmetrics = pd.concat(l)
# multimdoelgridvotermetrics = pd.concat(gridvotermetrics)







