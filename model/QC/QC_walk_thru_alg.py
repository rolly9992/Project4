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
import seaborn as sns

import matplotlib.pyplot as plt
from scipy import stats
pd.options.display.float_format = '{:20,.2f}'.format
import sys 
from sklearn import svm

feature_number =100 
weight_threshold = 0.005

sequences = [0,1,2,3,4]

#2480,107
df = pd.read_excel('ml_data.xlsx')
df = df.set_index('ticker')




#this may be taken care of previously.. 
#def prep_for_classifier(df):
df = df[df['sequence']!=5]

#keeping return in ML would mean data leakage. 2480,106
df = df.drop('return',axis=1) #we will use this later. But we would not know this in advance. 

#pulling out sequence separately. we do not want to normalize this particular column. 
seqvar = pd.DataFrame(df['sequence'])
df = df.drop('sequence',axis=1) #2480 105 

#2480 104
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
new_df = pd.concat([dfcat,seqvar,df_nonbool,df_bool],axis=1) #2480, 115

#in case the nan sector field does notexist in each sequence or all blanks. 
#we need to remove 1 of the sector dummy cols anyway. This is the best one. 
try:
    new_df = new_df.drop('gics_sector_nan',axis=1)
except Exception:
    pass
new_df.to_excel('new_df.xlsx')
#return new_df

#sys.exit()

#def get_X_y_data_classifier(df,sequence):
    #print(df.shape)
sequence = 0
new_df = new_df[new_df['sequence']==sequence] #496, 113
#print(df.shape)
new_df = new_df.drop('sequence',axis=1)
       
X = new_df.iloc[:,:-1]

#X.to_excel('X.xlsx')    
#print(X.columns.tolist())
y = new_df.iloc[:,-1:]
y=y['better_than_spy'].values
#return X,y

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.30,random_state=42)

weight = 0.005
n = 100

clf =RandomForestClassifier(random_state=42)
clf.fit(X_train,y_train) 
#def get_top_n_important_features(df_train,model,n,weight):


    
feature_importances = list(zip(X_train.columns.tolist(), clf.feature_importances_))

df_feature_importances = pd.DataFrame(feature_importances, columns=['Feature', 'Model Weight'])
filtered_features =df_feature_importances[df_feature_importances['Model Weight']>=weight] #screening out the lower rated features 
sorted_features=filtered_features.sort_values(['Model Weight'],ascending=[False])
top_n = sorted_features[:n] #looking at the top 10 

top_n = top_n.reset_index()
top_n = top_n.drop('index',axis=1)
top_n.index += 1 #make the first number 1 since we're looking at a top 10.
top_n[['Feature','Model Weight']]  #96,2

#return top_n

   

#def evaluate_classifier_model(model, X_test, Y_test): #, category_names=None):
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

y_pred = clf.predict(X_test)
y_pred_avg = y_pred.mean()

    
accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred,average='weighted',zero_division=1)
recall=recall_score(y_test,y_pred,average='weighted')
f1score = f1_score(y_test,y_pred,average='weighted')    
# print('\nscores:')
# temp ={  'model':model,
#        #'sequence':sequence, 
#        'Accurancy':accuracy,
#                 'Precision':precision,
#                 'Recall':recall,
#                 'F1 Score':f1score}
# print(temp)
# #return temp

# sys.exit()
# #def train_classifier_model(df,sequence):

# #TODO need to train with all features to find out relative importance of features
# dfin = prep_for_classifier(df)
    
# X,y=get_X_y_data_classifier(dfin,sequence=sequence)

# print('shapes')
# print(dfin.shape,X.shape,y.shape)

# ### starting ML ##############
# #split into train/test sets 
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.30,random_state=42)

# #train the generic model with all features to find more important ones
# # although using 
# clf =RandomForestClassifier(random_state=42)
# clf.fit(X_train,y_train)

# top_features =get_top_n_important_features(X_train,model=clf,n=feature_number,weight=weight_threshold)

# #TODO remove the save after dev
# top_features.to_excel('top_features.xlsx')

# print('number of features being used:',len(top_features))

# #reducing features to more important ones. 
# X = X[top_features]
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.30,random_state=42)



# print('starting grid search')
# '''REFERENCE: categories to use in params:['bootstrap', 'ccp_alpha', 
# 'class_weight', 'criterion', 'max_depth', 
# 'max_features', 'max_leaf_nodes', 'max_samples', 'min_impurity_decrease',
# 'min_samples_leaf', 'min_samples_split', 'min_weight_fraction_leaf', 
# 'monotonic_cst', 'n_estimators', 'n_jobs', 'oob_score', 'random_state', 
# 'verbose', 'warm_start'] '''
# parameters = {
# 'n_estimators': [20,50,150,300],
#  'max_depth': [2, 5, 10,None],
# 'min_samples_split': [2, 5, 10],
# 'min_samples_leaf': [1, 2, 4],
# 'oob_score':[True],
# #'max_features':[None,50,75]
#     }


# #trying to find the best precision here
# clfgrid = GridSearchCV(clf, parameters,scoring='precision',refit=False,cv=5, n_jobs=-1)

# clfgrid.fit(X_train, y_train)

# best_params = clfgrid.best_params_
# print('best params:',best_params)
# print('best score:',clfgrid.best_score_)
# #converting the clf.cv_results to dataframe
# df_results =pd.DataFrame.from_dict(clfgrid.cv_results_)
# df_results.to_excel('grid_results.xlsx')

# # Create a new instance of the model with the best parameters
# improved_clf = RandomForestClassifier(**best_params,random_state=42)

# # Fit the model on the training data
# improved_clf.fit(X_train, y_train)

# #evaluate_classifier_model(model=improved_clf, X_test=X_test, Y_test=y_test)
# #return improved_clf,top_features

