

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
    metricsout ={  'model':[model],
           'sequence':[sequence], 
           'Accurancy':[accuracy],
                    'Precision':[precision],
                    'Recall':[recall],
                    'F1 Score':[f1score]}
    #TODO remove print statement. curate into 1 larger df for easy comparison
    print(metricsout)
    metricsout = pd.DataFrame(metricsout)
    return y_probs,y_pred ,metricsout


#DROP
# def get_classifier_metrics(model, X_test, Y_test,sequence): #, category_names=None):
#     '''INPUT
#     the machine learning model we built earlier
#     X_test data
#     Y_test data
#     sequence number - for reference in metric output
#     the category names (pulled from the Y_test dataframe)
    
#     OUTPUT 
#     dataframe metrics on how well each category performed, including 
#     -model name
#     -accuracy
#     -precision
#     -recall
#     -the F1 score
    
#     the output files are later used in some visuals in the Flask app
       
#     '''

#     y_pred = model.predict(X_test)
#     #y_probs = model.predict_proba(X_test)[:,1] #only get if yes since we can derive no
       
#     accuracy=accuracy_score(Y_test,y_pred)
#     precision=precision_score(Y_test,y_pred,average='weighted',zero_division=1)
#     recall=recall_score(Y_test,y_pred,average='weighted')
#     f1score = f1_score(Y_test,y_pred,average='weighted')    
#     print('\nscores:')
#     metricsout ={  'model':[model],
#            'sequence':[sequence], 
#            'Accurancy':[accuracy],
#                     'Precision':[precision],
#                     'Recall':[recall],
#                     'F1 Score':[f1score]}
#     #TODO remove print statement. curate into 1 larger df for easy comparison
#     #print(temp)
#     metricsout = pd.DataFrame(metricsout)
    
#     return metricsout

#TODO refactor this def. 
# def train_classifier_feature_selection_run_on_next_sequence(df,model,sequence):

#     X,y=get_X_y_data_classifier(df,sequence=s)
    
#     ######RANDOM FOREST CLASSIFIER
#     ###########################################
    
#     X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
#     # #train the generic model with all features to find more important ones
#     #modelclf1 =RandomForestClassifier(random_state=42)
#     model.fit(X_train,y_train)
#     evaluate_classifier_model(model, X_test=X_test, Y_test=y_test,sequence=sequence)
#     print(cross_val_score(model, X_train, y_train, cv=5))
#     #CONDITIONAL BASED ON MODEL TYPE
#     try:
#         #attempt to refit model with reduced number of features
#         top_n = get_top_n_important_features(X_train,model,n= feature_keep)
#         X_train =X_train[top_n]
#         X_test = X_test[top_n]
#         model.fit(X_train,y_train)
#         evaluate_classifier_model(model, X_test=X_test, Y_test=y_test,sequence=sequence)
#         print(cross_val_score(model, X_train, y_train, cv=5))
        
#         #use model on next sequence
#         X,y=get_X_y_data_classifier(dfin,sequence=s+1)
#         # #split into train/test sets 
        
#         X =X[top_n]
#         y_probs,y_pred=evaluate_classifier_model(model, X_test=X, Y_test=y,sequence=sequence)
        
#     except Exception:
#         #SVC and KNN can't use the above route.         
#         X,y=get_X_y_data_classifier(dfin,sequence=s+1)
#         # #split into train/test sets NOT HERE.. next sequence is all new data 
#         #X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
#         y_probs,y_pred=evaluate_classifier_model(model, X_test=X, Y_test=y,sequence=sequence) 

    
#     return y_probs,y_pred,metricsout

#TODO refactor this def. 
def train_classifier_feature_selection_run_on_next_sequence_v2(df,model,sequence):
    '''
    get inputs 
    train model
    test model 
    save metrics
    run prediction on next sequence get those pics 
    also save metrics from that for the very end. 

    '''

    X,y=get_X_y_data_classifier(df,sequence=s)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
    # #train the generic model with all features to find more important ones
    #modelclf1 =RandomForestClassifier(random_state=42)
    model.fit(X_train,y_train)
    evaluate_classifier_model(model, X_test=X_test, Y_test=y_test,sequence=sequence)
    crossval = (cross_val_score(model, X_train, y_train, cv=5))
    print('average cross val:',crossval.mean())
    
    #CONDITIONAL BASED ON MODEL TYPE
    try:
        #attempt to refit model with reduced number of features
        top_n = get_top_n_important_features(X_train,model,n= feature_keep)
        X_train =X_train[top_n]
        X_test = X_test[top_n]
        model.fit(X_train,y_train)
        #evaluate_classifier_model(model, X_test=X_test, Y_test=y_test,sequence=sequence)
        print(cross_val_score(model, X_train, y_train, cv=5))
        
        #use model on NEXT sequence
        X,y=get_X_y_data_classifier(dfin,sequence=s+1)
        X =X[top_n]
        y_probs,y_pred=evaluate_classifier_model(model, X_test=X, Y_test=y,sequence=sequence)
    
    except Exception:
        #SVC and KNN can't use the above route.         
        X,y=get_X_y_data_classifier(dfin,sequence=s+1)
        y_probs,y_pred=evaluate_classifier_model(model, X_test=X, Y_test=y,sequence=sequence) 

    
    return y_probs,y_pred




###workout refactoring 


#inside def 
#def
#inputs: model, sequence, df, try reduce features yes,no 
#outputs ypred yprob, metrics (for 1 model, not multiple. too complex

#make universal
#keep the version with the higher precision
def train_improve_model(model,sequence):
    top_n=None
    prec2=0
    #split
    X,y=get_X_y_data_classifier(dfin,sequence=s)
    #train test
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
    
    # #train the generic model with all features to find more important ones
    #modelclf1 =RandomForestClassifier(random_state=42)
    model.fit(X_train,y_train)
    y_probs,y_pred,metricsout1=evaluate_classifier_model(model, X_test=X_test, Y_test=y_test,sequence=sequence)
    crossval = (cross_val_score(model, X_train, y_train, cv=5))
    print('average cross val:',crossval.mean())
    prec1 = metricsout1['Precision'].iloc[0]
    print('prec1:',prec1)
    #CONDITIONAL BASED ON MODEL TYPE
    try:
        model2 = model
        #attempt to refit model with reduced number of features
        top_n = get_top_n_important_features(X_train,model,n= feature_keep)
        X_train =X_train[top_n]
        X_test = X_test[top_n]
        model2.fit(X_train,y_train)
        y_probs,y_pred,metricsout2=evaluate_classifier_model(model2, X_test=X_test, Y_test=y_test,sequence=sequence)
        crossval2 = (cross_val_score(model2, X_train, y_train, cv=5))
        #print(cross_val_score(model, X_train, y_train, cv=5))
        print('average cross val:',crossval2.mean())
        prec2 = metricsout2['Precision'].iloc[0]
        print('prec2:',prec2)
    except Exception:
        pass #keeps initial training 
    if prec1>prec2:
        modelchoice = model
        finalmetrics =metricsout1
        finalcrossval = crossval
    else:
        modelchoice = model2
        finalmetrics = metricsout2
        finalcrossval = crossval2
    print('final:',modelchoice,finalmetrics['Precision'].iloc[0],finalcrossval.mean() )        
    return modelchoice,finalmetrics,finalcrossval ,top_n


#if top_n exists, need to include that 
def predict_future_data(model,sequence,top_n):
    
    X,y=get_X_y_data_classifier(dfin,sequence=sequence)
    if top_n==None:
        pass
    else:
        X=X[top_n]
    #model.predict(X)
    y_probs,y_pred,metricsout=evaluate_classifier_model(model, X_test=X, Y_test=y,sequence=sequence)
    return y_probs,y_pred,metricsout 


#TODO
'''run  the 5 models through sequence i, get metrics and cv
if possible, reduce features on the model and rerun, collecting metrics
then test on the next sequence to look at drop off

use precision weights before training the ensemble model
do the same thing on the ensemble model, running on sequence i, get metrics and cv

'''




######################################################################
#trying to use reduced features as an option 
###########################################################################
#preps data for classifier --only need to do once. 
dfin = prep_for_classifier(df)


s = 3
sequence = s

# #split specific sequence into X, y components (all features)

generic_models = [RandomForestClassifier(random_state=42),
       GradientBoostingClassifier(),
      LogisticRegression(random_state=42),
      SVC(probability=True),
      KNeighborsClassifier()     ]

#run models thru a loop to evaluate individually 
# then run thru ensemble 

#model = models[0]  
mnames =[]
voters =[]  
lf =[]
for i in range(5):
    model = generic_models[i]   
    
    model,metricsouttest,crossval,top_n =train_improve_model(model,sequence)
    model_name = type(model).__name__
    mnames.append(model_name)
    voters.append((model_name,model))    
    #voters.append(model)
    #sys.exit()
    #for DEV using on next sequence to predict stock picks and test at the end 
    y_probs,y_pred,metricsoutfuture =predict_future_data(model,sequence=s+1,top_n=top_n)
    lf.append(metricsoutfuture)
    
predictedmetricsoutfuture = pd.concat(lf,axis=0)
    
    

combined_model = VotingClassifier(estimators=voters)
#model,metricsouttest,crossval,top_n =train_improve_model(combined_model,sequence)
X,y=get_X_y_data_classifier(dfin,sequence=s)
# train test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
 
combined_model.fit(X_train,y_train)    
# predicting the output on the test dataset
pred_final = combined_model.predict(X_test)
print('evaluate combined model')
accuracy=accuracy_score(y_test,pred_final)
precision=precision_score(y_test,pred_final,average='weighted',zero_division=1)
recall=recall_score(y_test,pred_final,average='weighted')
f1score = f1_score(y_test,pred_final,average='weighted')    
print('\nscores:')
temp ={  'model':'ensemble_model',
       #'sequence':sequence, 
       'Accurancy':accuracy,
                'Precision':precision,
                'Recall':recall,
                'F1 Score':f1score}
print(temp)

#trying on next sequence 
print('\n\n combined on next sequence')
X,y=get_X_y_data_classifier(dfin,sequence=s+1)
# train test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
 
#combined_model.fit(X_train,y_train)    
# predicting the output on the test dataset
pred_final = combined_model.predict(X_test)
print('evaluate combined model')
accuracy=accuracy_score(y_test,pred_final)
precision=precision_score(y_test,pred_final,average='weighted',zero_division=1)
recall=recall_score(y_test,pred_final,average='weighted')
f1score = f1_score(y_test,pred_final,average='weighted')    
print('\nscores:')
temp ={  'model':'ensemble_model',
       #'sequence':sequence, 
       'Accurancy':accuracy,
                'Precision':precision,
                'Recall':recall,
                'F1 Score':f1score}
print(temp)

######################################################################
#end using reduced features as an option 
###########################################################################


sys.exit()
# ##############################################
# s = 0
# dfin = prep_for_classifier(df)
# # #split specific sequence into X, y components (all features)

# models = [RandomForestClassifier(random_state=42),
#        GradientBoostingClassifier(),
#       LogisticRegression(random_state=42),
#       SVC(probability=True),
#       KNeighborsClassifier()     ]
# #model = RandomForestClassifier(random_state=42)
# youts = []
# mnames =[]
# voters =[]
# X,y=get_X_y_data_classifier(dfin,sequence=s)
# for i in range(len(models)):
#     y_probs,y_pred = train_classifier_feature_selection_run_on_next_sequence(df=dfin,model=models[i],sequence=s)    
#     model_name = type(models[i]).__name__
#     print(model_name)
#     X[model_name]=y_probs
#     mnames.append(model_name)
#     voters.append((model_name,models[i]))
    

# #adding probabilities from each model (can't do directly from ensemble model)    
# X['AVG_Prob']=(X[mnames[0]]+X[mnames[1]]+X[mnames[2]]+X[mnames[3]]+X[mnames[4]])/5
# X = X.merge(dfm[dfm['sequence']==s+1],left_index=True,right_index=True,how = 'left')
# X = X.sort_values('AVG_Prob',ascending=False)
# picks = X[:number_of_tickers_to_invest_in ]
# yourtickers = picks.index
# yourtickers = list(yourtickers)
# print('model stock picks:', yourtickers)
# picks['startmoney'] =  starting_money /number_of_tickers_to_invest_in
# picks['endmoney'] = picks['startmoney']+picks['startmoney'] * picks['return']

# dollardiff = picks['endmoney'].sum() - picks['startmoney'].sum() 
# gainloss = picks['endmoney'].sum() /picks['startmoney'].sum() -1

# print('\n\nSUMMARY:')
# print('stock picks dollar change',dollardiff)
# print('stock picks gain loss',gainloss)


# #whatisthis = SPY['Spy Return'][SPY['sequence']==s+1][0]
# print('SPY dollar diff ',starting_money * SPY['Spy Return'][SPY['sequence']==s+1][0])
# print('SPY return',SPY['Spy Return'][SPY['sequence']==s+1][0])

# if gainloss>SPY['Spy Return'][SPY['sequence']==s+1][0]:
#     print('beat the SPY')
# else:
#     print('sorry, you did not beat the SPY this time')
    
# check = dfm[dfm.index.isin(yourtickers)]    
# check.to_excel('check.xlsx')



# #### ensemble on sequence i 
   
    
# X,y=get_X_y_data_classifier(dfin,sequence=s)    
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
  
# combined_model = VotingClassifier(estimators=voters)
# combined_model.fit(X_train,y_train)    
# # predicting the output on the test dataset
# pred_final = combined_model.predict(X_test)
# print('evaluate combined model')


# accuracy=accuracy_score(y_test,pred_final)
# precision=precision_score(y_test,pred_final,average='weighted',zero_division=1)
# recall=recall_score(y_test,pred_final,average='weighted')
# f1score = f1_score(y_test,pred_final,average='weighted')    
# print('\nscores:')
# temp ={  'model':'ensemble_model',
#        #'sequence':sequence, 
#        'Accurancy':accuracy,
#                 'Precision':precision,
#                 'Recall':recall,
#                 'F1 Score':f1score}
# print(temp)

    
# ### ensemble on sequence i + 1 - all new data
# X,y=get_X_y_data_classifier(dfin,sequence=s+1)    
# #combined_model = VotingClassifier(estimators=voters)
# #combined_model.fit(X,y)    
# # predicting the output on the test dataset
# pred_final = combined_model.predict(X)
# print('evaluate combined model')

# accuracy=accuracy_score(y,pred_final)
# precision=precision_score(y,pred_final,average='weighted',zero_division=1)
# recall=recall_score(y,pred_final,average='weighted')
# f1score = f1_score(y,pred_final,average='weighted')    
# print('\nscores:')
# temp ={  'model':'ensemble_model',
#        #'sequence':sequence, 
#        'Accurancy':accuracy,
#                 'Precision':precision,
#                 'Recall':recall,
#                 'F1 Score':f1score}
# print(temp)

# #evaluate_classifier_model(combined_model, X_test=X, Y_test=y)




