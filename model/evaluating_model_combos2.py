



import pandas as pd 
pd.options.mode.chained_assignment = None
import numpy as np
import matplotlib.pyplot as plt
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
# from sklearn.model_selection import GridSearchCV
# from sklearn import metrics
# from sklearn.metrics import mean_squared_error, r2_score, make_scorer
# from sklearn.metrics import max_error, mean_absolute_error, median_absolute_error 
# from sklearn import linear_model
from sklearn.ensemble import VotingClassifier


pd.options.display.float_format = '{:20,.2f}'.format

# sequences = [0,1,2,3,4,5]


df = pd.read_excel('ml_data.xlsx')
df = df.set_index('ticker')
# dfm = df[['better_than_spy','return','next_rolling62_adjustedclose','rolling62_adjustedclose','sequence']]







# ################################################################
# #### CLASSIFIER DEFINITIONS 
# ################################################################
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




def train_generic_model(model,X_train,y_train):
    model.fit(X_train,y_train)
    return model


def train_voter_model(voters,X_train,y_train):
    combined_model = VotingClassifier(estimators=voters)
    combined_model.fit(X_train,y_train)  
    return combined_model


def cross_validate_model_mean(model,X_train,y_train):
    crossval = (cross_val_score(model, X_train, y_train, cv=5))
    return crossval.mean()


# #TODO refactor 
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

# #only need to do this once. 
dfin = prep_for_classifier(df)



generic_models = [RandomForestClassifier(random_state=42),
       GradientBoostingClassifier(),
      LogisticRegression(random_state=42),
      SVC(probability=True),
      KNeighborsClassifier()     ]

def evaluate_testing_on_future_data():
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

future_evals,avg_crossvals =evaluate_testing_on_future_data()
print('average precision:',future_evals['Precision'].mean())

