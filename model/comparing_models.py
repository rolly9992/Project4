
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



#df = pd.read_excel('/home/milo/Documents/Project4/data/machineLearningData.xlsx')
df = pd.read_excel('model/ml_data.xlsx')
df = df.set_index('ticker')
df.shape




#get the numerical variables
num_vars = df.select_dtypes(include=['float','int']).columns
dfnum=df[num_vars]
#corr = dfnum.corr()
#corr #we will need to consider correlated features 



#df.describe()



nulls = df.isna().sum()
nulls.max() #no nulls after manipulation 



#plot the non boolean numerical variables to look at distributions 
#NOTE some distributions look squished due to the "cookie cutter" method to quickly look at all of them. 

booleanvars = [col for col in dfnum.columns if set(dfnum[col].unique()).issubset({0,1})]
#print(len(booleanvars))
nonbooleanvars = list(set(num_vars)-set(booleanvars))
cols = dfnum[nonbooleanvars].columns
dfnum_normalized = (dfnum - dfnum.mean())/dfnum.std()
print(len(cols))
#print out histograms of the nonboolean numeric columns in sets of 4 


#plot the non boolean numerical variables to look at distributions 
#NOTE some distributions look squished due to the "cookie cutter" method to quickly look at all of them. 
cols = dfnum[nonbooleanvars].columns
#print(len(cols))
#print out histograms of the nonboolean numeric columns in sets of 4 
# for i in range(26):
#     temp = dfnum[cols[4*i:4*i+4]]
#     temp.hist(bins=50)
# # #get the remaining leftover 2... 
# temp = dfnum[cols[72:74]]
# temp.hist()



def prep_for_classifier(df):
    df = df[df['sequence']!=4]

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
    



mod_df= prep_for_classifier(df=df)
#print(mod_df.shape)
#print(mod_df['sequence'].unique())



X,y=get_X_y_data_classifier(mod_df,sequence=1)
print(X.shape)
print(y.shape)


def evaluate_classifier_model(model, X_test, Y_test): #, category_names=None):
    '''INPUT
    a machine learning model 
    X_test data
    Y_test data
    the category names (pulled from the Y_test dataframe)
    
    OUTPUT 
    metric files on how well each category performed, including 
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
    print('\nscores:')
    temp ={  'model':[model],
           #'sequence':sequence, 
           'Accurancy':[accuracy],
                    'Precision':[precision],
                    'Recall':[recall],
                    'F1 Score':[f1score]}
    temp = pd.DataFrame(temp)
    print(temp)
    return temp



#borrowing some code from my first project to save a little time and baking it into a definition
def get_top_n_important_features(df_train,model,n,weight=0.02):
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



#top features to pull out in descending rank order (using above def in each model)
topfeatures = 50




#TODO use gradient booster model for classifier and random forest regressor for regressor. 
# retrain model each quarter. use top 30 features. do grid search

#note the sequences are simply shorthand for the 4 quarters we are looking at
sequences = [0,1,2,3] 




########################################################################################################
#train a generic Gradient Boosting Classifier (default parameters) model to then look at model decay in future quarters 
#in other words, use the same model on successive quarters without retraining with new data. 
################################################################################################

# #note the sequences are simply shorthand for the 4 quarters we are looking at
# sequences = [0,1,2,3] 

# dfout = prep_for_classifier(df)
# X,y=get_X_y_data(dfout,sequence=sequences[0])
# print('shapes')
# print(dfout.shape,X.shape,y.shape)

# ### starting ML ##############
# #split into train/test sets 
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42)

# #train a "no frills GradientBoostingClassifier"
# #rf = RandomForestClassifier(random_state=42)
# #rf.fit(X_train,y_train)
# gbc =GradientBoostingClassifier(random_state=42)
# gbc.fit(X_train,y_train)
# top_n_gbc =get_top_n_important_features(X_train,model=gbc,n=topfeatures)

#evaluate_model(model=rf, X_test=X_test, Y_test=y_test)

####################################################################################################
#look at decay.. accuracy, precision, recall, F1 scores without training a new model with new data
####################################################################################################
# for j in range(len(sequences)):
#     print('\n\n\nsequence:',j)
#     #dfout,X,y = prep_for_classifier(df,sequence=sequences[j])
#     X,y=get_X_y_data(dfout,sequence=sequences[j])
#     print('shapes')
#     print(dfout.shape,X.shape,y.shape)

#     ### starting ML ##############
#     #split into train/test sets 
#     X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42)
#     nans = X_train.isna().sum()
#     print('nans over zero?',nans[nans>0])
#     try:
#         evaluate_model(model=gbc, X_test=X_test, Y_test=y_test)
#     except Exception as e:
#         print(e)


# # The gradient booster classifier - using default values and no grid search - does a little better in the first sequence than the random forest with 66.6 accuracy and 68.8 precision. But, using the same model in successive quarters without retraining shows a steeper decay with the last sequence's accuracy at only 47.1 and precision of 52.3. Conclusion: we will need to retrain data for classifiers to attempt to avert potential score declines.  

# top_n_rf



# top_n_gbc



# #combined "best of both lists to use as feature subset"
# topfilter1 = top_n_rf[top_n_rf['Model Weight']>0.01]
# topfilter1 = topfilter1['Feature'].tolist()

# type(top_n_gbc)

# topfilter2 = top_n_gbc[top_n_gbc['Model Weight']>.02]
# topfilter2 = topfilter2['Feature'].tolist()

# combinedfilter = set(topfilter1).union(set(topfilter2))
# combinedfilter = list(combinedfilter)


# #correlations of features seem reasonably low. nothing more than in the .60s
# corr = dfout[combinedfilter].corr()
# corr.iloc[:,20:] #note - checked in 2 sets since otherwise middle ones don't show. 






########################################################################################################
#train a generic Gradient Boosting Classifier with the subset of features (still using default parameters)  to then look at model decay in future quarters 
#in other words, use the same model on successive quarters without retraining with new data. 
################################################################################################

#note the sequences are simply shorthand for the 4 quarters we are looking at
sequences = [0,1,2,3] 

#TODO need to train with all features to find out relative importance of features
dfout = prep_for_classifier(df)


X,y=get_X_y_data_classifier(dfout,sequence=sequences[0])

print('shapes')
print(dfout.shape,X.shape,y.shape)

### starting ML ##############
#split into train/test sets 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42)

#train the generic gbc with all features to find more important ones 
gbc =GradientBoostingClassifier(random_state=42)
gbc.fit(X_train,y_train)
top_features =get_top_n_important_features(X_train,model=gbc,n=50,weight=0.01)

print((top_features))
print(len(top_features))

#reducing features to more important ones. 
X = X[top_features]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42)

#scoring = {'accuracy': make_scorer(accuracy_score),
#           'precision': make_scorer(precision_score),'recall':make_scorer(recall_score)}


print('starting grid search')
#TODO if we retrain need to consider trade off of computation time for 2 models times 4 quarters. 
parameters = {
  
    "learning_rate": [0.01, 0.025, 0.05, 0.1, 0.15, 0.2],
    "min_samples_split": np.linspace(0.1, 0.5, 12),
    "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "max_depth":[3,5,8],
    #"max_features":["log2","sqrt"],
    #"criterion": ["friedman_mse",  "mae"],
    #"subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    #"n_estimators":[10]
    }
#using precision as the metric to improve
gbcgrid = GridSearchCV(GradientBoostingClassifier(), parameters,scoring='precision',refit=False,cv=3, n_jobs=-1)

gbcgrid.fit(X_train, y_train)

best_params = gbcgrid.best_params_
#converting the clf.cv_results to dataframe
df_results =pd.DataFrame.from_dict(gbcgrid.cv_results_)
df_results.to_excel('gbc_results.xlsx')

# Create a new instance of the model with the best parameters
improved_gbc = GradientBoostingClassifier(**best_params,random_state=42)

# Fit the model on the training data
improved_gbc.fit(X_train, y_train)

evaluate_classifier_model(model=improved_gbc, X_test=X_test, Y_test=y_test)


#### now loop thru 


# dfout = prep_for_classifier(df)
# X,y=get_X_y_data_classifier(dfout,sequence=sequences[0])
# #X = X[combinedfilter]
# print('shapes')
# print(dfout.shape,X.shape,y.shape)

# ### starting ML ##############
# #split into train/test sets 
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42)
# ##train a "no frills Random Forest Classifier"
# #TODO use improved gbc model 
# #rf = RandomForestClassifier(random_state=42)
# #rf.fit(X_train,y_train)



#top_n_rf =get_top_n_important_features(X_train,model=rf,n=10)

l=[]

for j in range(len(sequences)):
    print('\n\n\nsequence:',j)
    #dfout,X,y = prep_for_classifier(df,sequence=sequences[j])
    X,y=get_X_y_data_classifier(dfout,sequence=sequences[j])
    X = X[top_features]
    print('shapes')
    print(dfout.shape,X.shape,y.shape)

    ### starting ML ##############
    #split into train/test sets 
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42)
    try:
        metrics = evaluate_classifier_model(model=improved_gbc, X_test=X_test, Y_test=y_test)
        metrics['sequence'] = np.nan 
        metrics['sequence'].iloc[0] = j
        print('metrics for:',j,'\n',metrics)
        l.append(metrics)
        
    except Exception as e:
        print('exception of:',e)
#print(l)
metricresults = pd.concat(l,axis=1)
print(metricresults)








# def prep_for_regressor(df):
#     df = df[df['sequence']!=4]

#     #keeping return in ML would mean data leakage. 
#     df = df.drop('return',axis=1) #we will use this later. But we would not know this in advance. 
#     #pulling out sequence separately. we do not want to normalize this particular column. 
#     seqvar = pd.DataFrame(df['sequence'])
#     df = df.drop('sequence',axis=1)
#     df=df.drop('better_than_spy',axis=1)
    
#     #borrowing some code from project 1 
#     num_vars = df.select_dtypes(include=['float','int']).columns
#     cat_vars = df.select_dtypes(include=['object']).columns
#     dfnum = df[num_vars]
#     booleanvars = [col for col in dfnum.columns if set(dfnum[col].unique()).issubset({0,1})]
#     #print(len(booleanvars))
#     nonbooleanvars = list(set(num_vars)-set(booleanvars))
    

#     #create dummy variables for the categorical variable set then drop the original categorical non numerical columns
#     l = [df[cat_vars]]
#     for i in range(len(cat_vars)):
#         temp = cat_vars[i]
#         catout = pd.get_dummies(df[temp],prefix=temp,prefix_sep='_',dummy_na=True,drop_first=True)
#         l.append(catout)
#     dfcat=pd.concat(l,axis=1)
#     dfcat=dfcat.drop(columns=cat_vars,axis=0)
#     #print(df.shape)
#     #print(dfcat.shape) #expecting an increase due to adding dummies. 
#     cat_cols = dfcat.columns.tolist()
#     for i in range(len(cat_cols)):
#         dfcat[cat_cols[i]] = dfcat[cat_cols[i]].astype(int)
    
#     df_bool = df[booleanvars]
#     df_nonbool = df[nonbooleanvars]
#     df_nonbool = (df_nonbool-df_nonbool.mean())/df_nonbool.std()
#     new_df = pd.concat([dfcat,seqvar,df_nonbool,df_bool],axis=1)

#     return new_df
# def get_X_y_data_regressor(df,sequence):
#     #print(df.shape)
    
#     df = df[df['sequence']==sequence]
#     #print(df.shape)
#     df = df.drop('sequence',axis=1)
           
#     X = df #.iloc[:,:-1]
#     X = X.drop('next_rolling62_adjustedclose',axis=1)

#     #X.to_excel('X.xlsx')    
#     #print(X.columns.tolist())
#     y = df['next_rolling62_adjustedclose'].values
#     y=df['next_rolling62_adjustedclose'].values
#     #y=y['better_than_spy'].values
#     return X,y
    


# # In[158]:


# def evaluate_regressor_model(model, X_test, y_test): #, category_names=None):
#     '''INPUT
#     the machine learning model we built earlier
#     X_test data
#     Y_test data
#     the category names (pulled from the Y_test dataframe)
    
#     OUTPUT 
#     excel metric files on how well each category performed, including 
#     -accuracy
#     -precision
#     -recall
#     -the F1 score
    
#     the output files are later used in some visuals in the Flask app
       
#     '''

#     y_pred = model.predict(X_test)
#     #y#cols = Y_test.columns.tolist()
#     #category_names =ycols
#     #y_pred2 = pd.DataFrame(y_pred,columns=ycols)
#     #l = []
#     #for i in range(len(ycols)):
        
#     #source:https://stackoverflow.com/questions/50789508/random-forest-regression-how-do-i-analyse-its-performance-python-sklearn
#     print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_pred))
#     print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_pred))
#     print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#     mape = np.mean(np.abs((y_test - y_pred) / np.abs(y_test)))
#     print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))
#     print('Accuracy:', round(100*(1 - mape), 2))
#     print('Score:',r2_score(y_test, y_pred))
#     #print(temp)
#     #return temp


# # In[ ]:





# # In[152]:


# rdf = prep_for_regressor(df)
# rdf.head()


# # In[164]:



# ########################################################################################################
# #train a generic Random Forest Regressor (default parameters) model to then look at model decay in future quarters 
# #in other words, use the same model on successive quarters without retraining with new data. 
# ################################################################################################

# #note the sequences are simply shorthand for the 4 quarters we are looking at
# sequences = [0,1,2,3] 

# dfout = prep_for_regressor(df)
# X,y=get_X_y_data_regressor(dfout,sequence=sequences[0])
# print('shapes')
# print(dfout.shape,X.shape,y.shape)

# ### starting ML ##############
# #split into train/test sets 
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42)
# #train a "no frills Random Forest Classifier"
# rfr = RandomForestRegressor(random_state=42)
# rfr.fit(X_train,y_train)

# top_n_rfr =get_top_n_important_features(X_train,model=rf,n=topfeatures)
# print('how the model ranks the same feature subset')
# print(top_n_rfr)
# #evaluate_model(model=rf, X_test=X_test, Y_test=y_test)

# ####################################################################################################
# #look at decay.. accuracy, precision, recall, F1 scores without training a new model with new data
# ####################################################################################################
# l=[]

# for j in range(len(sequences)):
#     print('\n\n\nsequence:',j)
#     #dfout,X,y = prep_for_classifier(df,sequence=sequences[j])
#     X,y=get_X_y_data_regressor(dfout,sequence=sequences[j])
#     print('shapes')
#     print(dfout.shape,X.shape,y.shape)

#     ### starting ML ##############
#     #split into train/test sets 
#     X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42)
#     try:
#         evaluate_regressor_model(model=rfr, X_test=X_test, y_test=y_test)
        
        
#     except Exception as e:
#         print(e)


# # In[ ]:





# # In[ ]:





# # In[156]:


# #trying regressor models  

# ########################################################################################################
# #train a generic Random Forest Regressor (default parameters) model to then look at model decay in future quarters 
# #in other words, use the same model on successive quarters without retraining with new data. 
# ################################################################################################

# #note the sequences are simply shorthand for the 4 quarters we are looking at
# sequences = [0,1,2,3] 

# dfout = prep_for_regressor(df)
# #X,y=get_X_y_data(dfout,sequence=sequences[0])
# X,y = get_X_y_data_regressor(dfout,sequence=0)

# print('shapes')
# print(X.shape,y.shape)

# ### starting ML ##############
# #split into train/test sets 
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42)
# #train a "no frills Random Forest Classifier"
# #rf = RandomForestClassifier(random_state=42)
# #rf.fit(X_train,y_train)
# #gbc =GradientBoostingClassifier(random_state=42)
# #gbc.fit(X_train,y_train)
# rfreg = RandomForestRegressor(random_state=42)
# rfreg.fit(X_train,y_train)

# #evaluate_model(model=rf, X_test=X_test, Y_test=y_test)

# ####################################################################################################
# #look at decay.. accuracy, precision, recall, F1 scores without training a new model with new data
# ####################################################################################################
# for j in range(4): #len(sequences)):
#     print('\n\n\nsequence:',j)
#     #dfout,X,y = prep_for_classifier(df,sequence=sequences[j])
#     X,y=get_X_y_data_regressor(dfout,sequence=sequences[j])
#     print('shapes')
#     print(dfout.shape,X.shape,y.shape)

#     ### starting ML ##############
#     #split into train/test sets 
#     X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42)
#     y_pred = rfreg.predict(X_test)
    
#     #source:https://stackoverflow.com/questions/50789508/random-forest-regression-how-do-i-analyse-its-performance-python-sklearn
#     print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_pred))
#     print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_pred))
#     print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#     mape = np.mean(np.abs((y_test - y_pred) / np.abs(y_test)))
#     print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))
#     print('Accuracy:', round(100*(1 - mape), 2))
#     print('Score:',r2_score(y_test, y_pred))
#     #     #try:
#     #    evaluate_model(model=rfreg, X_test=X_test, Y_test=y_test)
#     #except Exception as e:
#     #    print(e)


# # In[169]:


# #trying regressor models  

# ########################################################################################################
# #train a generic Multiple Linear Regression (default parameters) model to then look at model decay in future quarters 
# #in other words, use the same model on successive quarters without retraining with new data. 
# ################################################################################################

# #note the sequences are simply shorthand for the 4 quarters we are looking at
# sequences = [0,1,2,3] 

# dfout = prep_for_regressor(df)
# #X,y=get_X_y_data(dfout,sequence=sequences[0])
# X,y = get_X_y_data_regressor(dfout,sequence=0)

 


