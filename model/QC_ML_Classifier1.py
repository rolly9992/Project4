

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
    for i in range(1,5):
        
        temp = pd.DataFrame({'quarter':[i],
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
    y_probs = model.predict_proba(X_test)[:,1] #only get if yes since we can derive no

        
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
    return y_probs


def train_classifier_feature_selection_run_on_next_sequence(df,model,sequence):

    X,y=get_X_y_data_classifier(dfin,sequence=s)
    
    ######RANDOM FOREST CLASSIFIER
    ###########################################
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
    # #train the generic model with all features to find more important ones
    #modelclf1 =RandomForestClassifier(random_state=42)
    model.fit(X_train,y_train)
    evaluate_classifier_model(model, X_test=X_test, Y_test=y_test)
    print(cross_val_score(model, X_train, y_train, cv=5))
    #CONDITIONAL BASED ON MODEL TYPE
    try:
        #attempt to refit model with reduced number of features
        top_n = get_top_n_important_features(X_train,model,n= feature_keep)
        X_train =X_train[top_n]
        X_test = X_test[top_n]
        model.fit(X_train,y_train)
        evaluate_classifier_model(model, X_test=X_test, Y_test=y_test)
        print(cross_val_score(model, X_train, y_train, cv=5))
        #use model on next sequence
        X,y=get_X_y_data_classifier(dfin,sequence=s+1)
        # #split into train/test sets 
        
        X =X[top_n]
        ynext=evaluate_classifier_model(model, X_test=X, Y_test=y)
        
    except Exception:
        #SVC and KNN can't use the above route.         
        X,y=get_X_y_data_classifier(dfin,sequence=s+1)
        # #split into train/test sets 
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
        ynext=evaluate_classifier_model(model, X_test=X, Y_test=y) 

    
    return ynext

##############################################
s = 0
dfin = prep_for_classifier(df)
# #split specific sequence into X, y components (all features)

models = [RandomForestClassifier(random_state=42),
       GradientBoostingClassifier(),
      LogisticRegression(random_state=42),
      SVC(probability=True),
      KNeighborsClassifier()     ]
#model = RandomForestClassifier(random_state=42)
youts = []
mnames =[]
X,y=get_X_y_data_classifier(dfin,sequence=s)
for i in range(5): #len(models)):
    yout = train_classifier_feature_selection_run_on_next_sequence(df=dfin,model=models[i],sequence=s)    
    model_name = type(models[i]).__name__
    print(model_name)
    X[model_name]=yout
    mnames.append(model_name)
    
X['AVG_Prob']=(X[mnames[0]]+X[mnames[1]]+X[mnames[2]]+X[mnames[3]]+X[mnames[4]])/5

X = X.merge(dfm[dfm['sequence']==s],left_index=True,right_index=True,how = 'left')
    
#X,y=get_X_y_data_classifier(dfin,sequence=s)


# X['y']=y
# X['RF_prob']=y1out
# X['GBC_prob']=y2out
# X['LOGR_prob']=y3out
# X['SVC_prob']=y4out
# X['KNN_prob']=y5out
# X['AVG_Prob']=(X['RF_prob']+X['GBC_prob']+X['LOGR_prob']+X['SVC_prob']+X['KNN_prob'])/5




sys.exit()
######RANDOM FOREST CLASSIFIER
###########################################

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
# #train the generic model with all features to find more important ones
#modelclf1 =RandomForestClassifier(random_state=42)
model.fit(X_train,y_train)
evaluate_classifier_model(model, X_test=X_test, Y_test=y_test)
print('cvs:',cross_val_score(model, X_train, y_train, cv=5))
#CONDITIONAL BASED ON MODEL TYPE
try:
    top_n = get_top_n_important_features(X_train,model,n= feature_keep)
    X_train =X_train[top_n]
    X_test = X_test[top_n]
    model.fit(X_train,y_train)
    evaluate_classifier_model(model, X_test=X_test, Y_test=y_test)
    print('cvs:',cross_val_score(model, X_train, y_train, cv=5))
    X,y=get_X_y_data_classifier(dfin,sequence=s+1)
    # # #split into train/test sets 
    #X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
    X =X[top_n] #running entire next sequence data to use predictions for investing
    # X_test = X_test[top_n]        
    ynext=evaluate_classifier_model(model, X_test=X, Y_test=y)
except Exception as e:
    print('exception of',e)
    #pass
    #SVC and KNN can't use the above route.         
    X,y=get_X_y_data_classifier(dfin,sequence=s+1)
    # #split into train/test sets 
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
    ynext=evaluate_classifier_model(model, X_test=X, Y_test=y) 
    #NOTE: even though we're looking at this, there is no data leakage.        
    print(cross_val_score(model, X_train, y_train, cv=5))   




sys.exit()   

X,y=get_X_y_data_classifier(dfin,sequence=s)

######RANDOM FOREST CLASSIFIER
###########################################

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
# #train the generic model with all features to find more important ones
clf1 =RandomForestClassifier(random_state=42)
clf1.fit(X_train,y_train)
evaluate_classifier_model(clf1, X_test=X_test, Y_test=y_test)
print(cross_val_score(clf1, X_train, y_train, cv=5))

top_n = get_top_n_important_features(X_train,clf1,n= feature_keep)
X_train =X_train[top_n]
X_test = X_test[top_n]
clf1.fit(X_train,y_train)
evaluate_classifier_model(clf1, X_test=X_test, Y_test=y_test)
print(cross_val_score(clf1, X_train, y_train, cv=5))

#sys.exit()

######GRANDIENT BOOSTER CLASSIFIER
###########################################
#X,y=get_X_y_data_classifier(dfin,sequence=s)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
clf2 = GradientBoostingClassifier()
clf2.fit(X_train,y_train)
evaluate_classifier_model(clf2, X_test=X_test, Y_test=y_test)
print(cross_val_score(clf2, X_train, y_train, cv=5))

top_n = get_top_n_important_features(X_train,clf2,n= feature_keep)
X_train =X_train[top_n]
X_test = X_test[top_n]
clf2 = GradientBoostingClassifier()
clf2.fit(X_train,y_train)
evaluate_classifier_model(clf2, X_test=X_test, Y_test=y_test)




######LOGISTIC REGRESSION
###########################################
X,y=get_X_y_data_classifier(dfin,sequence=s)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
clf3 =LogisticRegression(random_state=42)
clf3.fit(X_train,y_train)
evaluate_classifier_model(clf3, X_test=X_test, Y_test=y_test)
print(cross_val_score(clf3, X_train, y_train, cv=5))

top_n = get_top_n_important_features(X_train,clf3,n= feature_keep)
X_train =X_train[top_n]
X_test = X_test[top_n]
clf3 =LogisticRegression(random_state=42)
clf3.fit(X_train,y_train)
evaluate_classifier_model(clf3, X_test=X_test, Y_test=y_test)




# ######SVC CLASSIFIER
# ###########################################
X,y=get_X_y_data_classifier(dfin,sequence=s)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
#NOTE: need to explicitly state probabily true for SVCs
clf4 = SVC(probability=True)
clf4.fit(X_train,y_train)
evaluate_classifier_model(clf4, X_test=X_test, Y_test=y_test)
print(cross_val_score(clf4, X_train, y_train, cv=5))

#ROUTE NOT AVAILABLE WITH SVC
# top_n = get_top_n_important_features(X_train,clf4,n= feature_keep)
# X_train =X_train[top_n]
# X_test = X_test[top_n]
# clf4 = SVC(probability=True)
# clf4.fit(X_train,y_train)
# evaluate_classifier_model(clf4, X_test=X_test, Y_test=y_test)

# sys.exit()

######KNN
############################################
X,y=get_X_y_data_classifier(dfin,sequence=s)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
clf5 =KNeighborsClassifier() #n_neighbors=3)
clf5.fit(X_train,y_train)
evaluate_classifier_model(clf5, X_test=X_test, Y_test=y_test)
print(cross_val_score(clf5, X_train, y_train, cv=5))

#THIS ROUTE not available with KNN
# top_n = get_top_n_important_features(X_train,clf5,n= feature_keep)
# X_train =X_train[top_n]
# X_test = X_test[top_n]
# clf5 =KNeighborsClassifier()
# clf5.fit(X_train,y_train)
# evaluate_classifier_model(clf5, X_test=X_test, Y_test=y_test)

#need to bake into a definition... 
print('\n\n###############################################')
print('USING THE 5 MODELS ON THE NEXT QUARTER DATA')
#try the above 5 models on the next quarter
X,y=get_X_y_data_classifier(dfin,sequence=s+1)
# #split into train/test sets 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,shuffle=True)
# #train the generic model with all features to find more important ones
#clf1 =RandomForestClassifier(random_state=42)
#clf1.fit(X_train,y_train)
y1out=evaluate_classifier_model(clf1, X_test=X, Y_test=y)
#print(cross_val_score(clf1, X_train, y_train, cv=5))

#clf2 = GradientBoostingClassifier()
#clf2.fit(X_train,y_train)
y2out=evaluate_classifier_model(clf2, X_test=X, Y_test=y)
#print(cross_val_score(clf2, X_train, y_train, cv=5))

#clf3 =LogisticRegression(random_state=42)
#clf3.fit(X_train,y_train)
y3out=evaluate_classifier_model(clf3, X_test=X, Y_test=y)
#print(cross_val_score(clf3, X_train, y_train, cv=5))

#clf4 = SVC()
#clf4.fit(X_train,y_train)
y4out =evaluate_classifier_model(clf4, X_test=X, Y_test=y)
#print(cross_val_score(clf4, X_train, y_train, cv=5))

#clf5 =KNeighborsClassifier() #n_neighbors=3)
#clf5.fit(X_train,y_train)
y5out = evaluate_classifier_model(clf5, X_test=X, Y_test=y)
#print(cross_val_score(clf5, X_train, y_train, cv=5))

X = X.reset_index()

#y1out = pd.DataFrame(y1out)
#y11 = y1out[1]
X['y']=y
X['RF_prob']=y1out
X['GBC_prob']=y2out
X['LOGR_prob']=y3out
X['SVC_prob']=y4out
X['KNN_prob']=y5out
X['AVG_Prob']=(X['RF_prob']+X['GBC_prob']+X['LOGR_prob']+X['SVC_prob']+X['KNN_prob'])/5






# top_features_df =get_top_n_important_features(X_train,model=clf,n=feature_number,weight=weight_threshold)
# top_features_df.to_excel('top_features.xlsx')       
# top_features = top_features_df['Feature'].tolist()    
#print('number of features determined by grid search:',len(top_features))
#TODO remove the save after dev


# #reducing features to the filtered set. 
# X = X[top_features]
# print('reduced shape:',X.shape)
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.30,random_state=42,shuffle=True)
# print('reduced shapes:',X_train.shape,X_test.shape)
# #retrain the generic model with the filtered X set
# clf.fit(X_train,y_train)    