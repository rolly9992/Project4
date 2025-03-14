


#TODO need to clean up this code
#TODO delete unncessary commented out code
#TODO incorporate cross val avgs in output... 

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
from sklearn import preprocessing, svm
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.metrics import max_error, mean_absolute_error, median_absolute_error 
from sklearn import linear_model
from sklearn.ensemble import VotingClassifier
pd.options.display.float_format = '{:20,.2f}'.format
import sys 


df = pd.read_excel('model/ml_data.xlsx')
df = df.set_index('ticker')
dfm = df[['better_than_spy','return','next_rolling62_adjustedclose','rolling62_adjustedclose','sequence']]


generic_models = [RandomForestClassifier(random_state=42),
       GradientBoostingClassifier(),
      LogisticRegression(random_state=42),
      SVC(probability=True),
      KNeighborsClassifier()     ]


bools = ['gics_sector_Consumer Discretionary', 'gics_sector_Consumer Staples', 
 'gics_sector_Energy', 'gics_sector_Financials', 'gics_sector_Health Care',
 'gics_sector_Industrials', 'gics_sector_Information Technology', 
 'gics_sector_Materials', 'gics_sector_Real Estate', 'gics_sector_Utilities']


# #Note default n of 100. 
def get_top_n_important_features(df_train,model,n= 100):
    '''INPUT
    X_train dataframe, model, number of features to use 
    NOTE: using a random forest classifier with this particular definition
    OUTPUT 
    list of features to keep
    '''
    try: #was using this for previous models like Random Forest Classifier
        feature_importances = list(zip(df_train.columns.tolist(), model.feature_importances_))
    except:
        topcoeff = model.coef_[0]
        feature_importances = list(zip(df_train.columns.tolist(), topcoeff))
    df_feature_importances = pd.DataFrame(feature_importances, columns=['Feature', 'Model Weight'])
    df_feature_importances=df_feature_importances.sort_values(['Model Weight'],ascending=[False])    
    top_n = df_feature_importances[:n]  #looking at the top 10 
    top_n = top_n.reset_index()
    top_n = top_n.drop('index',axis=1)
    top_n.index += 1 #make the first number 1 since we're looking at a top n. 
    top_n[['Feature','Model Weight']]
    features_keep = top_n['Feature'].tolist()
    
    return features_keep

# ################################################################
# #### CLASSIFIER DEFINITIONS 
# ################################################################
def prep_for_classifier(df):
    '''INPUT 
    dataframe
    OUTPUT 
    modified dataframe with added dummy variables (dropping original columns)
    and normalizing non boolean numerical columns
    '''
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
    
    nonbooleanvars = list(set(num_vars)-set(booleanvars))
    

    #create dummy variables for the categorical variable set then drop the original categorical non numerical columns
    l = [df[cat_vars]]
    for i in range(len(cat_vars)):
        temp = cat_vars[i]
        catout = pd.get_dummies(df[temp],prefix=temp,prefix_sep='_',dummy_na=True,drop_first=True)
        l.append(catout)
    dfcat=pd.concat(l,axis=1)
    dfcat=dfcat.drop(columns=cat_vars,axis=1)
    
    cat_cols = dfcat.columns.tolist()
    for i in range(len(cat_cols)):
        dfcat[cat_cols[i]] = dfcat[cat_cols[i]].astype(int)
    
    df_bool = df[booleanvars]
    df_nonbool = df[nonbooleanvars]
    df_nonbool = (df_nonbool-df_nonbool.mean())/df_nonbool.std()
    new_df = pd.concat([dfcat,seqvar,df_nonbool,df_bool],axis=1)
     
    #we need to remove 1 of the sector dummy cols anyway. This is likely the best one. 
    try:
        new_df = new_df.drop('gics_sector_nan',axis=1)
    except Exception:
        pass
    return new_df

def get_X_y_data_classifier(df,sequence):
    '''INPUT 
    dataframe, sequence (quarter) number
    OUTPUT X and y dataframes for machine learning
    '''    
    df = df[df['sequence']==sequence]
    df = df.drop('sequence',axis=1)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1:]
    y=y['better_than_spy'].values
    return X,y



def train_generic_model(model,X_train,y_train):
    '''INPUT 
    model
    X_train and y_train
    OUTPUT 
    fitted model
    '''
    model.fit(X_train,y_train)
    return model



def train_voter_model(voters,X_train,y_train):
    '''INPUT
    list of voter models, X_train, y_train
    OUTPUT voter model using all input voter models
    '''
    combined_model = VotingClassifier(estimators=voters)
    combined_model.fit(X_train,y_train)  
    return combined_model


def cross_validate_model_mean(model,X_train,y_train):
    '''INPUT
    model, X_train, y_train
    OUTPUT 
    the average cross validation score using cv of 5
    '''
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

#NOTE only need to do this once. 
dfin = prep_for_classifier(df)


# ###########################################################
# ## PHASE 1 use the 5 generic models and a voter model
# ###########################################################

def train_evaluate_generic_model(dfin,sequence):
    '''INPUT 
    DataFrame, sequence
    OUTPUT 
    metrics on a single model
    '''
    X,y=get_X_y_data_classifier(dfin,sequence=sequence)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
    #train 5 generic models plus voter 
    mnames =[]
    voters =[]  
    collectmetrics =[]
    for i in range(5):
        model = generic_models[i]   
        model = train_generic_model(model,X_train,y_train)
        metricsout = evaluate_model(model=model, X_test=X_test, Y_test=y_test,sequence=sequence)
        model_name = type(model).__name__
        mnames.append(model_name)
        voters.append((model_name,model))    
        collectmetrics.append(metricsout)
    metricstest_generic = pd.concat(collectmetrics,axis=0)
    
    voter_model = VotingClassifier(estimators=voters)
    voter_model.fit(X_train,y_train)    

    metricstest_genericCombined = evaluate_model(model=voter_model, X_test=X_test, Y_test=y_test,sequence=sequence)
    metricstest_genericCombined['model']='generic_voter_model' 
    allmodelmetrics = pd.concat([metricstest_generic,metricstest_genericCombined],axis=0)
    return allmodelmetrics

 
def train_evaluate_all_generic_models(dfin):
    '''INPUT
    DataFrame
    OUTPUT 
    metrics on generic models
    '''
    l = []
    for i in range(5):
        genericout = train_evaluate_generic_model(dfin,sequence=i)
        l.append(genericout)
    genericmetricsdf = pd.concat(l,axis=0)
    genericmetricsdf.to_excel('generic_metrics.xlsx')
    print('generic model metrics data created')
    return genericmetricsdf

# #############################################################################################
# ### PHASE 2 explore what the best number of features might be and test on each of the 5 models
# #############################################################################################

def explore_different_number_of_features_used(dfin,sequence):
    '''INPUT
    Dataframe, sequence number
    OUTPUT 
    metrics test on feature reduction 
    '''
    
    X,y=get_X_y_data_classifier(dfin,sequence=sequence)
    #train test
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
    featurenumbers =[100,90,80,70,60,50]
    
    featurelist = []
    for i in range(len(featurenumbers)):
        model = generic_models[0] #using RandomForestClassifier for feature reduction
        model = train_generic_model(model,X_train,y_train)
        features_keep = get_top_n_important_features(X_train,model,n= featurenumbers[i])
        featurelist.append(features_keep)    
   
    #step 1 train 5 generic models plus voter 

    mnames =[]
    voters =[]  
   
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

# #implementing with 90 features as this seemed the best out of the reduced sets.  
def implement_different_feature_number_model_metrics(dfin):
    '''INPUT
    dataframe 
    OUTPUT 
    important features 
    '''
    
    l = []
    for i in range(5):
        featurestestingout = explore_different_number_of_features_used(dfin,sequence=i)
        featurestestingout['sequence']= i
        l.append(featurestestingout)
    featurestestingmetricsdf = pd.concat(l,axis=0)
    featurestestingmetricsdf.to_excel('features_testing_metrics.xlsx')
    print('models with reduced feature number metrics data created')
    return featurestestingmetricsdf

# #####################################################################################
# ### PHASE 3 use models with grid search then a voter model 
# #####################################################################################

#NOTE: not using all possible parameters to save time since multiple periods and models to train. 
paramdict ={
    'GBC_parameters': {
    "loss":["exponential", "log_loss"],
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    #"min_samples_split": np.linspace(1, 10, 10),
    #"min_samples_leaf": np.linspace(0.0, 1.0, 10),
    "max_depth":[3,5,8],
    #"max_features":["log2","sqrt"],
    #"criterion": ['squared_error', 'friedman_mse'],
    #"subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators":[10]
    },
    'RF_parameters':{
    'n_estimators': [200, 700],
    #'max_depth':[10,30,50],
    #'min_samples_split':[2,3,5],
    #'min_samples_leaf':[3,5],
    'max_features': [90,110]
      }
    ,'LR_parameters':{'penalty':['l2'], 
              'C':[1, 10, 100, 1000]}
    ,'SVC_parameters':{'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']  }
    #NOTE Valid parameters are: ['algorithm', 'leaf_size', 'metric', 'metric_params', 'n_jobs', 'n_neighbors', 'p', 'weights'].
    ,'KNN_parameters': { 'n_neighbors': [3, 5, 7, 9, 11]     }
      }

def easy_loop(a):
    '''INPUT
    None
    OUTPUT
    a list of generic models to train in a future function and dictionary of grid parameter choices with numbers
    '''
    generic_models = [  GradientBoostingClassifier(),
        RandomForestClassifier(random_state=42),
         LogisticRegression(random_state=42,max_iter=1000),
          SVC(probability=True),
          KNeighborsClassifier()     ]
    paramchoices = {0: 'GBC_parameters', 1:'RF_parameters'
                    ,2:'LR_parameters',
                    3:'SVC_parameters',4:'KNN_parameters'}
    return generic_models[a],paramchoices[a]


def do_grid_search(model,parameters,sequence):
    X,y=get_X_y_data_classifier(dfin,sequence=sequence)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
    precision_scorer = make_scorer(precision_score, zero_division=0) 
    gbc_grid = GridSearchCV(model, parameters, scoring=precision_scorer, cv=5, n_jobs=-1)
    gbc_grid.fit(X_train, y_train)
    df_cv_results=pd.DataFrame.from_dict(gbc_grid.cv_results_)
    gridmetricsout = evaluate_model(model=gbc_grid, X_test=X_test, Y_test=y_test,sequence=sequence)
    return df_cv_results,gridmetricsout
    


def do_grid_search_on_generic_models(dfin):
    '''INPUT dataframe
    OUTPUT 
    metrics on how all the models with grid search performed
    '''
    l=[]
    gridvotermetrics=[]
    #outer loop for sequences
    for j in range(5):
        mnames =[]
        voters =[]  
        for i in range(5):
            l1,l2 = easy_loop(i)
            model_name = type(l1).__name__
            #print('sequence:',j,'  model:',model_name)
            cv_results,gridmetrics = do_grid_search(model=l1,parameters=paramdict[l2],sequence=j)
            mnames.append(model_name)
            voters.append((model_name,l1))    
            
            gridmetrics['model name']=model_name
            gridmetrics['sequence']=j
            l.append(gridmetrics)
        voter_gridmodel = VotingClassifier(estimators=voters)
        X,y=get_X_y_data_classifier(dfin,sequence=j)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
        voter_gridmodel.fit(X_train,y_train)    
        metricstest_gridvoter = evaluate_model(model=voter_gridmodel, X_test=X_test, Y_test=y_test,sequence=j)
        gridvotermetrics.append(metricstest_gridvoter)
            
    
    multimodelgridmetrics = pd.concat(l)
    multimdoelgridvotermetrics = pd.concat(gridvotermetrics)
    multimdoelgridvotermetrics['model name']='Grid Voter' 
    all_grid_metrics = pd.concat([multimodelgridmetrics,multimdoelgridvotermetrics],axis=0)
    all_grid_metrics.to_excel('all_grid_search_metrics.xlsx')
    print('grid search metrics data created')
    return all_grid_metrics


def evaluate_model_using_future_data(model,dfin,sequence): #, category_names=None):
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
    X,y=get_X_y_data_classifier(dfin,sequence=sequence)
    #NOTE, we are not splitting this into training and testing, but only predicting
    y_pred = model.predict(X)
    accuracy=accuracy_score(y,y_pred)
    precision=precision_score(y,y_pred,average='weighted',zero_division=1)
    recall=recall_score(y,y_pred,average='weighted')
    f1score = f1_score(y,y_pred,average='weighted')    
    metricsout ={ 'test for': ['future data'] ,
        'model':[model],
           'sequence':[sequence], 
           'Accurancy':[accuracy],
                    'Precision':[precision],
                    'Recall':[recall],
                    'F1 Score':[f1score]}
    metricsout = pd.DataFrame(metricsout)
    return metricsout

def evaluate_testing_on_future_data(dfin):
    '''INPUT
    dataframe
    OUTPUT 
    dataframe of multiple model evaluation and average cross validations 
    '''
    
    future_evals =[]
    for j in range(0,4):
        #get sequence to train generic model
        X,y=get_X_y_data_classifier(dfin,sequence=j)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
        voters =[]
        crossvals=[]
        for i in range(len(generic_models)):
            model =train_generic_model(generic_models[i], X_train, y_train)
            cv_avg =cross_validate_model_mean(model,X_train,y_train)
            crossvals.append(cv_avg)
            model_name = type(model).__name__
            #mnames.append(model_name)
            voters.append((model_name,model))  
                   
            #evaluate model on next quarter's data.    
            eval_future_data=evaluate_model_using_future_data(model,dfin,j+1) 
            eval_future_data['sequence']=j+1
            future_evals.append(eval_future_data)
        votermodel=train_voter_model(voters,X_train,y_train)
        cv_avg =cross_validate_model_mean(model,X_train,y_train)
        crossvals.append(cv_avg)
        eval_future_data=evaluate_model_using_future_data(votermodel,dfin,j+1) 
        eval_future_data['sequence']=j+1
        future_evals.append(eval_future_data)
    future_curated_evals= pd.concat(future_evals,axis=0)
    future_curated_evals.reset_index(inplace=True)
    del future_curated_evals['index']
    return future_curated_evals,sum(crossvals)/len(crossvals)





def main():

    generic =train_evaluate_all_generic_models(dfin)
    featurenum = implement_different_feature_number_model_metrics(dfin)
    #print(featurenum.groupby(['numberoffeatures'])['Precision'].mean())
    grid =do_grid_search_on_generic_models(dfin)
    print('\nPrecision averages for model groups:')
    print('\nAverage generic model precision:',generic['Precision'].mean())
    print('\nAverage feature reduction model precisions:')
    print(featurenum.groupby(['numberoffeatures'])['Precision'].mean())
    print('\nAverage grid model precision:',grid['Precision'].mean())
    future_evals,avg_crossvals =evaluate_testing_on_future_data(dfin)
    print('\nAverage precision using future data (ie, next quarter) (with generic models):',future_evals['Precision'].mean())


if __name__ == '__main__':
    main()


