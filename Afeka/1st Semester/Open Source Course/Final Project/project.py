#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-block alert-success">
# <b>to do:</b>
#     <BR> &#9730; Data description as appendix?
#     <BR> &#9730; something with sunburst?
#     <BR> &#9730; Add explenations for every step
#     <BR> &#9730; Add theoretical background on every model 
# </div>

# <div id=header class="alert alert-block alert-info">
# <b>Final Project For Course 236502001 - כלים טכנולוגיות קוד פתוח למערכות תבוניות </b>
#         <BR>Presented By:
#         <BR>&emsp;1. Tali Presaizen 123456789
#         <BR>&emsp;2. Mor Atiya 123456789
#         <BR>&emsp;3. Amir Yorav 123456789
#     <BR>Kaggle user: <a href=”https://www.google.com">KKK</a></div>

#  # Table of Contents  
# 1. [Introduction](#Introduction)   
#     1. [Data Set Description](#Data-Set-Description)  
#     1. [Variable Description](#Variable-Description) 
# 1. [Imports](#-Imports) 
# 1. [Exploring The Data](#Exploring-The-Data) 
#     1. [Loading](#Loading) 
#     1. [Missing Values](#Missing-Values)
#     1. [Variable Correlations](#Variable-Correlations)
#     1. [Target Variable](#Target-Variable)
#     1. [Outliers](#Outliers)
# 1. [Prepering Data for models](#Prepering-Data-for-models)     
#     1. [Encoding](#Encoding)
#     1. [Splitting the Data](#Splitting-the-Data)   
#     1. [Feature Scaling](#Feature-Scaling) 
# 1. [Feature Selection](#-Feature-Selection)
#     1. [Ridge-Lasso-Elasticnet](#Ridge-Lasso-Elasticnet) 
#     1. [Trees](#Trees) 
#     1. [PCA](#PCA)
# 1. [SGD Model](#SGD-Model)     
#     1. [Tune Hyper Parameters](#SGD---Tune-Hyper-Parameters)
#     1. [Evaluating on Validation set](#SGD---Evaluating-on-Validation-set)
#     1. [Model Submission](#SGD---Model-Submission)     
# 1. [Random Forest Model](#Random-Forest-Model)     
#     1. [Tune Hyper Parameters](#RF---Tune-Hyper-Parameters)
#     1. [Evaluating on Validation set](#RF---Evaluating-on-Validation-set)     
#     1. [Model Submission](#RF-Model-Submission) 
# 1. [Model TTT](#Model-TTT)     
#     1. [Tune Hyper Parameters](#TM---Tune-Hyper-Parameters)   
#     1. [Evaluating on Validation set](#TM---Evaluating-on-Validation-set)     
#     1. [Submission](#TM---Submission) 
# 1. [Ensemble of the Models](#Ensemble-of-the-Models)     
#     1. [Create Ensemble](#Create-Ensemble)    
#     1. [Evaluating on Validation set](#Evaluating-on-Validation-set)     
# 1. [Final Submission](#Conclusions) 
# 1. [Conclusions](#Conclusions) 
# 1. [References](#References)    

# <table id=Introduction style="width:100%">
# <tr><td><h1>Introduction</h1></td><td><a href="#header"><img src="https://30percentclub.org/wp-content/uploads/2021/07/back-to-top-icon-01.svg" width="70" height="70" align="right"></a></td></tr>
# </table>

# ### Data Set Description

# ### Variable Description

# <table id=-Imports style="width:100%">
# <tr><td><h1>Imports</h1></td><td><a href="#header"><img src="https://30percentclub.org/wp-content/uploads/2021/07/back-to-top-icon-01.svg" width="70" height="70" align="right"></a></td></tr>
# </table> 

# In[1]:


# import numpy, matplotlib, etc.
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# define plt settings
plt.rcParams["font.size"] = 8
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["figure.figsize"] = (5,5)
get_ipython().run_line_magic('matplotlib', 'inline')

# sklearn imports
from sklearn import metrics
from sklearn import pipeline
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import neural_network
from sklearn import model_selection
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeavePOut
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

#for some statistics
from scipy.stats import uniform
from scipy import stats
from scipy.stats import norm, skew 
from statistics import mean 


# <table id=Exploring-The-Data style="width:100%">
# <tr><td><h1>Exploring The Data</h1></td><td><a href="#header"><img src="https://30percentclub.org/wp-content/uploads/2021/07/back-to-top-icon-01.svg" width="70" height="70" align="right"></a></td></tr>
# </table>

# ### Loading

# In[2]:


train_data = pd.read_csv(r"C:\Users\efiattia\Desktop\mor\ai projects\train.csv")
test_data = pd.read_csv(r"C:\Users\efiattia\Desktop\mor\ai projects\test.csv")


# In[3]:


train_data


# In[4]:


train_data.shape


# In[5]:


test_data


# <div class="alert alert-block alert-success">
# We see that <strong>test</strong> have all the columns except the <strong>'SalePrice'</strong> which is our target
# </div>

# In[6]:


# concat all data so we can make all the preprocessing on all
target = train_data['SalePrice']
all_data = pd.concat([train_data.drop('SalePrice',axis=1), test_data])


# In[7]:


all_data.shape


# ---

#  #### visual reports

# from autoviz.AutoViz_Class import AutoViz_Class
# 
# AV = AutoViz_Class()
# dft = AV.AutoViz("", depVar='SalePrice', dfte=train_data, verbose=2)
# 
#  Note: verbose=0 or 1 generates charts and displays them in your local Jupyter notebook.
#        verbose=2 saves plots in your local machine under AutoViz_Plots directory and does not display charts.

# import sweetviz as sv
# analyze_report = sv.analyze(train_data)
# analyze_report.show_html()

# ---

# In[8]:


all_data.describe()


# In[9]:


all_data.info()


# <div class="alert alert-block alert-success">
# Comment <strong>here</strong>
# </div>

# ### Missing Values

# In[10]:


# find missing data percentage for each column
def find_missing(df):
    df_na =(df.isnull().sum() / len(df)) * 100
    df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending=False)[:30]
    missing_df = pd.DataFrame({'Missing Data' :df_na})
    return missing_df


# <div class="alert alert-block alert-warning">
# <h1><b>Q: </b>do we need to remove features with very high missing data %? </h1>
# </div>
# 

# In[11]:


na_data = find_missing(all_data)
na_data


# In[12]:


f, ax = plt.subplots(figsize=(20, 5))
plt.xticks(rotation='90')
sns.barplot(x=na_data.index, y=na_data['Missing Data'])
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


# <div class="alert alert-block alert-success">
# Comment <strong>here</strong>
# </div>

# In[13]:


#PoolQC : data description says NA means "No Pool". That make sense, given the huge ratio of missing value (+99%) and majority of houses have no Pool at all in general.
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

#MiscFeature : data description says NA means "no misc feature"
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

#Alley : data description says NA means "no alley access"
all_data["Alley"] = all_data["Alley"].fillna("None")

#Fence : data description says NA means "no fence"
all_data["Fence"] = all_data["Fence"].fillna("None")

#FireplaceQu : data description says NA means "no fireplace"
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

#LotFrontage : Since the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood , we can fill in missing values by the median LotFrontage of the neighborhood.
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

#GarageType, GarageFinish, GarageQual and GarageCond : Replacing missing data with None
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')

#GarageYrBlt, GarageArea and GarageCars : Replacing missing data with 0 (Since No garage = no cars in such garage.)
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)

#BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath : missing values are likely zero for having no basement
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)

#BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2 : For all these categorical basement-related features, NaN means that there is no basement.
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')

#MasVnrArea and MasVnrType : NA most likely means no masonry veneer for these houses. We can fill 0 for the area and None for the type.
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

#MSZoning (The general zoning classification) : 'RL' is by far the most common value. So we can fill in missing values with 'RL'
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

#Utilities : For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling. We can then safely remove it.
all_data = all_data.drop(['Utilities'], axis=1)

#Functional : data description says NA means typical
all_data["Functional"] = all_data["Functional"].fillna("Typ")

#Electrical : It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

#KitchenQual: Only one NA value, and same as Electrical, we set 'TA' (which is the most frequent) for the missing value in KitchenQual.
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

#Exterior1st and Exterior2nd : Again Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

#SaleType : Fill in again with most frequent which is "WD"
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

#MSSubClass : Na most likely means No building class. We can replace missing values with None
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")


# In[14]:


missing = find_missing(all_data)
missing


# <div class="alert alert-block alert-success">
# Hurray! No missing data left!
# </div>

# ### Variable Correlations

# In[15]:


def plot_correlation_heatmap(df):
    plt.figure(figsize=(16, 10))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cut_off = 0.6  # only show cells with abs(correlation) at least this value
    extreme_1 = 0.75  # show with a star
    extreme_2 = 0.85  # show with a second star
    extreme_3 = 0.90  # show with a third star
    mask |= np.abs(corr) < cut_off
    corr = corr[~mask]  # fill in NaN in the non-desired cells

    remove_empty_rows_and_cols = True
    if remove_empty_rows_and_cols:
        wanted_cols = np.flatnonzero(np.count_nonzero(~mask, axis=1))
        wanted_rows = np.flatnonzero(np.count_nonzero(~mask, axis=0))
        corr = corr.iloc[wanted_cols, wanted_rows]

    annot = [[f"{val:.4f}"
          + ('' if abs(val) < extreme_1 else '*')  # add one star if abs(val) >= extreme_1
          + ('' if abs(val) < extreme_2 else '**')  # add an extra star if abs(val) >= extreme_2
          + ('' if abs(val) < extreme_3 else '***')  # add yet an extra star if abs(val) >= extreme_3
          for val in row] for row in corr.to_numpy()]
    heatmap = sns.heatmap(corr, vmin=-1, vmax=1, annot=annot, fmt='', annot_kws={"fontsize":15})
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 20}, pad=16)
    plt.xticks(rotation=90, fontsize=20) 
    plt.yticks(rotation=0, fontsize=20) 
    plt.show()


# <div class="alert alert-block alert-success">
# Comment <strong>here</strong>
# </div>

# In[16]:


plot_correlation_heatmap(train_data)


# <div class="alert alert-block alert-success">
# General Comment <strong>here</strong>
# </div>

# <div class="alert alert-block alert-success">removed:
# <BR>GarageArea
# <BR>TotRmsAbvGrd
# <BR>GarageYrBlt
# <BR>1stFlrSF
# </div>

# ### Target Variable

# In[17]:


def QQ_plot(data, measure):
    fig = plt.figure(figsize=(20,7))

    #Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(data)

    #Kernel Density plot
    fig1 = fig.add_subplot(121)
    sns.histplot(data, fit=norm)
    fig1.set_title(measure + ' Distribution ( mu = {:.2f} and sigma = {:.2f} )'.format(mu, sigma), loc='center')
    fig1.set_xlabel(measure)
    fig1.set_ylabel('Frequency')

    #QQ plot
    fig2 = fig.add_subplot(122)
    res = stats.probplot(data, plot=fig2)
    fig2.set_title(measure + ' Probability Plot (skewness: {:.6f} and kurtosis: {:.6f} )'.format(data.skew(), data.kurt()), loc='center')

    plt.tight_layout()
    plt.show()


# In[18]:


#QQ_plot(target, 'Sales Price')


# In[19]:


# plot the target distribution
def plot_hist(data):
    plt.figure(figsize=(16, 5))
    sns.histplot(data);
    (mu, sigma) = norm.fit(data) # Get the fitted parameters used by the function
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    fig = plt.figure()
    plt.figure(figsize=(16, 5))
    res = stats.probplot(data, plot=plt)
    plt.show()


# In[20]:


plot_hist(target)


# <div class="alert alert-block alert-success">
# Comment: <strong>competition loss is RMSLE so we need to log-transform y</strong>
# </div>

# In[21]:


target = np.log1p(target)


# In[22]:


plot_hist(target)


# <div class="alert alert-block alert-success">
# Comment <strong> </strong>
# </div>

# #### SalePrice against OverallQual

# In[23]:


figure = plt.figure(figsize = (10,5))
ax = sns.lineplot(x = 'OverallQual', y = 'SalePrice', data = train_data)
plt.title("SalePrice against OverallQual", fontdict={'fontsize': 20})
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20) 

#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
f, ax = plt.subplots(figsize=(10, 5))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20)

plt.show()


# <div class="alert alert-block alert-success">
# Comment <strong>here</strong>
# </div>

# In[24]:


corr_predictors = ['GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'GarageArea', 'GarageCars', 'OverallQual']
# GrLivArea
idxs_GrLivArea = list(train_data[(train_data['GrLivArea']>4000) & (train_data['SalePrice']<300000)].index)
# TotalBsmtSF
idxs_TotalBsmtSF = list(train_data[(train_data['TotalBsmtSF']>4000) & (train_data['SalePrice']<300000)].index)
# 1stFlrSF
idxs_1stFlrSF = list(train_data[(train_data['1stFlrSF']>4000) & (train_data['SalePrice']<300000)].index)
# TotalBsmtSF
idxs_GarageArea = list(train_data[(train_data['GarageArea']>1200) & (train_data['SalePrice']<300000)].index)
all_idxs=[idxs_GrLivArea, idxs_TotalBsmtSF, idxs_1stFlrSF, idxs_GarageArea,[0] ,[0]]


# In[25]:


# plot_corr_predictors
def plot_corr_predictors(predictor_names):
    plt.figure(figsize=(20,10))
    
    for count, predictor in enumerate(corr_predictors, start=1):
        plt.subplot(2,3,count)
        #main plot
        ax = sns.scatterplot(x = predictor, y = 'SalePrice', data = train_data, hue='OverallQual', palette= 'YlOrRd')
        #outliers
        #ax = sns.scatterplot(x = train_data[predictor][all_idxs[count-1]], y = train_data['SalePrice'][all_idxs[count-1]])
        plt.title("SalePrice against " + predictor, fontdict={'fontsize': 20})
        
        
        x = mean(train_data[predictor][all_idxs[count-1]])
        y = mean(train_data['SalePrice'][all_idxs[count-1]])
        
        ax.annotate('Outliers\n zone', xy=(x-100, y+0), xytext=(-20,20), 
            textcoords='offset points', ha='center', va='baseline',
            bbox=dict(boxstyle='round,pad=0.2', fc='maroon', alpha=0.3),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', 
                            color='red'))

    plt.tight_layout(pad=4.0)
    plt.show()    


# In[26]:


plot_corr_predictors(corr_predictors)


# GarageCars_grouped = train_data.groupby('GarageCars')['SalePrice'].mean().reset_index()
# OverallQual_grouped = train_data.groupby('OverallQual')['SalePrice'].mean().reset_index()
# 
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# 
# ax1.pie(GarageCars_grouped['SalePrice'], labels=GarageCars_grouped['GarageCars'], autopct='%1.1f%%')
# ax1.set_title('GarageCars')
# 
# ax2.pie(OverallQual_grouped['SalePrice'], labels=OverallQual_grouped['OverallQual'], autopct='%1.1f%%')
# ax2.set_title('OverallQual')
# 
# 
# plt.show()

# In[27]:


# plot_corr_predictors
def plot_bars(predictor_names):
    plt.figure(figsize=(20,10))
    
    for count, predictor in enumerate(predictor_names, start=1):
        grouped = train_data.groupby(predictor)['SalePrice'].mean().reset_index()
        plt.subplot(2,3,count)
        ax = sns.barplot(x = grouped[predictor], y = grouped['SalePrice'])
        plt.title(predictor + " VS SalePrice against ", fontdict={'fontsize': 20})

    plt.tight_layout(pad=4.0)
    plt.show()   


# In[28]:


bar_predictors = ['GarageCars', 'OverallQual']
plot_bars(bar_predictors)


# <div class="alert alert-block alert-success">
# Comment <strong>here</strong>
# </div>

# <div class="alert alert-block alert-success">
# Comment <strong>here</strong>
# </div>

# <table id=Prepering-Data-for-models style="width:100%">
# <tr><td><h1>Prepering Data for models</h1></td><td><a href="#header"><img src="https://30percentclub.org/wp-content/uploads/2021/07/back-to-top-icon-01.svg" width="70" height="70" align="right"></a></td></tr>
# </table>

# ### Outliers

# In[29]:


Outliers = list(set(idxs_GrLivArea+idxs_TotalBsmtSF+idxs_1stFlrSF+idxs_GarageArea))
Outliers


# In[30]:


# remove categorial features which are highly correlated with other features
all_data = all_data.drop(['GarageArea', 'TotRmsAbvGrd', 'GarageYrBlt', '1stFlrSF'], axis=1)


# ### Encoding

# In[31]:


# Transforming some numerical variables that are really categorical

#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)


# - justify

# In[32]:


# Label Encoding some categorical variables that may contain information in their ordering set

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold', 'OverallCond')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print(all_data.shape)


# In[33]:


#Adding one more important feature
#Since area related features are very important to determine house prices, we add one more feature which is the total area of basement, first and second floor areas of each house
# Adding total sqfootage feature 
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['GrLivArea']


# In[34]:


all_data = pd.get_dummies(all_data)
print(all_data.shape)


# ### Splitting the data

# In[35]:


ntest = test_data.shape[0]+1 #number of rows in test data
train = all_data.iloc[:ntest, :]
test = all_data.iloc[ntest:, :]
print(train.shape, test.shape)


# In[36]:


feature_names = train.columns


# In[37]:


train = train.drop(Outliers)
target = target.drop(Outliers)


# In[38]:


#removing 'Id' column which is just the DB table index
train_data = train_data.drop('Id',axis=1) 
test_Ids = test_data['Id']
test_data = test_data.drop('Id',axis=1) 


# In[39]:


X_train, X_validate, y_train, y_validate = train_test_split(train,target,test_size=0.3,random_state=0)


# In[40]:


X_train


# <table id=-Feature-Selection style="width:100%">
# <tr><td><h1>Feature Selection</h1></td><td><a href="#header"><img src="https://30percentclub.org/wp-content/uploads/2021/07/back-to-top-icon-01.svg" width="70" height="70" align="right"></a></td></tr>
# </table>

# ### Ridge-Lasso-Elasticnet

# In[113]:


model = make_pipeline(StandardScaler(), SGDRegressor(penalty='l2', random_state=0))
alpha_space = {'alpha': uniform(0, 1)}#np.linspace(0, 1, 100)}
alphas = []
for i in range(30):
    ridge_cv = RandomizedSearchCV(model[1], alpha_space, cv=5)
    ridge_cv.fit(X_train, y_train)
    alphas.append(ridge_cv.best_params_['alpha'])


# In[114]:


best_alpha = np.array(alphas).mean()
best_alpha


# In[115]:


# Train a Ridge Regression model
ridge = make_pipeline(StandardScaler(), SGDRegressor(penalty='l2', alpha = best_alpha ,random_state=0))
ridge.fit(X_train, y_train)

# Plot the feature importances of ridge
coef = ridge[1].coef_
features = feature_names
importance = np.abs(coef)
sorted_idx = np.argsort(importance)

plt.figure(figsize=(10,40))
plt.barh(range(train.shape[1]), importance[sorted_idx])
plt.yticks(range(train.shape[1]), features[sorted_idx])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Ridge Regression Feature Importance")
plt.show()
ridge_features = features[sorted_idx]


# In[141]:


ridge_most = ridge_features[::-1][:50]
#ridge_most = X_train[most[:50]]
ridge_most


# In[142]:


# find the best alpha penalty for lasso Regression model
model = make_pipeline(StandardScaler(), SGDRegressor(penalty='l1', random_state=0))
alpha_space = {'alpha': np.linspace(0, 0.1, 100)}
alphas = []
for i in range(30):
    lasso_cv = RandomizedSearchCV(model[1], alpha_space, cv=5)
    lasso_cv.fit(X_train, y_train)
    alphas.append(lasso_cv.best_params_['alpha'])
best_alpha = np.array(alphas).mean()
best_alpha


# In[143]:


# Train a lasso Regression model
lasso = make_pipeline(StandardScaler(), SGDRegressor(penalty='l1', alpha = best_alpha ,random_state=0))
lasso.fit(X_train, y_train)

# Plot the feature importances of lasso
coef = lasso[1].coef_
features = feature_names
importance = np.abs(coef)
sorted_idx = np.argsort(importance)

plt.figure(figsize=(10,40))
plt.barh(range(train.shape[1]), importance[sorted_idx])
plt.yticks(range(train.shape[1]), features[sorted_idx])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("lasso Regression Feature Importance")
plt.show()
lasso_features = features[sorted_idx]


# In[134]:


lasso_most = lasso_features[::-1]
lasso_most
most = X_train[most[:50]]


# <div class="alert alert-block alert-warning">
# <h1><b>example:</b> </h1>
# </div>
# 

# In[49]:


model = make_pipeline(StandardScaler(), SGDRegressor(penalty='l2',alpha=best_alpha, random_state=0))
scores = cross_val_score(model, ridge_most, y_train, cv=15)
print(f"{scores.mean():.3f} (+/- {scores.std():.3f})")


# ### Trees

# In[83]:


model = ExtraTreesRegressor()
model.fit(X_train, y_train)
plt.figure(figsize=(10, 10))
feature_rank = pd.Series(model.feature_importances_, index = feature_names)
feature_rank.nlargest(50).plot(kind = "barh")


# In[84]:


tree_select = feature_rank.sort_values(ascending=False)[:50]
tree_select.index


# In[85]:


tree_most = X_train[tree_select.index]


# In[89]:


most_featurs = pd.Series(list(set(ridge_most) & set(lasso_most) & set(tree_most)))
most_featurs


# <div class="alert alert-block alert-warning">
# <h1><b>example:</b> </h1>
# </div>
# 

# In[54]:


model = make_pipeline(StandardScaler(), SGDRegressor(penalty='l2',alpha=best_alpha, random_state=0))
scores = cross_val_score(model, tree_most, y_train, cv=15)
print(f"{scores.mean():.3f} (+/- {scores.std():.3f})")


# ### PCA

# <div class="alert alert-block alert-warning">
# <h1><b>to do:</b> Look at the referrence and redo..</h1>
# </div>
# 

# In[55]:


# Apply PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_train)

# Get the loadings (coefficients) of the features in the principal components
loadings = pca.components_.T

# Create a DataFrame to display the loadings
loadings_df = pd.DataFrame(loadings, columns=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10"], index=feature_names)

# Sort the loadings by the magnitude of the coefficients
loadings_df["abs_loadings"] = np.abs(loadings_df.max(axis=1))
loadings_df = loadings_df.sort_values("abs_loadings", ascending=False)
top_30 = loadings_df[:30]
# Plot a bar graph of the feature names versus the loadings
plt.figure(figsize=(10,10))
plt.barh(top_30.index, top_30["abs_loadings"], align='center')
plt.xlabel('Absolute Loading Value')
plt.ylabel('Feature')
plt.title('Feature Importance (PCA)')
plt.show()


# In[56]:


top_30.index


# In[57]:


PCA_df = train[top_30.index]


# <table id=SGD-Model style="width:100%">
# <tr><td><h1>SGD Model</h1></td><td><a href="#header"><img src="https://30percentclub.org/wp-content/uploads/2021/07/back-to-top-icon-01.svg" width="70" height="70" align="right"></a></td></tr>
# <td><h2>Using Ridge, Lasso, and Elasticnet regulariztions</h2></td><td></td>
# </table>

# ### SGD - Tune Hyper Parameters

# In[58]:


# Define the parameter distribution for random search
param_dist = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'alpha': [1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5],
}

# Create an instance of the SGDRegressor class
sgd = make_pipeline(StandardScaler(), SGDRegressor())

# Run the random search with 5-fold cross-validation   
random_search = RandomizedSearchCV(sgd[1], param_distributions=param_dist, n_iter=20, cv=5, n_jobs=-1)
random_search.fit(X_train, y_train)

# Print the best hyperparameters found
print("Best hyperparameters: ", random_search.best_params_)


# In[59]:


sgd = make_pipeline(StandardScaler(), SGDRegressor(penalty= 'l1', max_iter= 1000, alpha= 0.10))
scores = cross_val_score(sgd, X_train, y_train, cv=15)
print(f"{scores.mean():.3f} (+/- {scores.std():.3f})")


# ### SGD - Evaluating on Validation set

# In[60]:


#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train)
    rmse= np.sqrt(-cross_val_score(model, X_validate, y_validate, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# In[61]:


rmsle_cv(sgd)


# ### SGD - Model Submission 

# In[62]:


def Submission(name, model):
    model.fit(train, target)
    predictions = np.exp(model.predict(test))
    output = pd.DataFrame({'Id': test_Ids, 'SalePrice': predictions})
    file_name = name + '_submission.csv'
    output.to_csv(file_name, index=False)
    return predictions


# In[63]:


pred1 = Submission('sgd', sgd)


# In[64]:


sns.histplot(pred)


# In[ ]:


sns.histplot(pred1)


# <table id=Random-Forest-Model style="width:100%">
# <tr><td><h1>Random Forest Model</h1></td><td><a href="#header"><img src="https://30percentclub.org/wp-content/uploads/2021/07/back-to-top-icon-01.svg" width="70" height="70" align="right"></a></td></tr>
# </table>

# ### RF - Tune Hyper Parameters

# ### RF - Evaluating on Validation set

# ### RF - Model Submission

# <table id=lin-reg style="width:100%">
# <tr><td><h1>Linear Regression Model or Support Vector Regression</h1></td><td><a href="#header"><img src="https://30percentclub.org/wp-content/uploads/2021/07/back-to-top-icon-01.svg" width="70" height="70" align="right"></a></td></tr>
# <td><h2>Using PCA</h2></td><td></td>
# </table>

# ### LR - Tune Hyper Parameters

# ### LR - Evaluating on Validation set 

# ### LR - Submission

# <table id=Ensemble-of-the-Models style="width:100%">
# <tr><td><h1>Ensemble of the Models</h1></td><td><a href="#header"><img src="https://30percentclub.org/wp-content/uploads/2021/07/back-to-top-icon-01.svg" width="70" height="70" align="right"></a></td></tr>
# </table>

# ### Create Ensemble

# ### Evaluating on Validation set

# <table id=Final-Submission style="width:100%">
# <tr><td><h1>Final Submission</h1></td><td><a href="#header"><img src="https://30percentclub.org/wp-content/uploads/2021/07/back-to-top-icon-01.svg" width="70" height="70" align="right"></a></td></tr>
# </table>

# <table id=Conclusions style="width:100%">
# <tr><td><h1>Conclusions</h1></td><td><a href="#header"><img src="https://30percentclub.org/wp-content/uploads/2021/07/back-to-top-icon-01.svg" width="70" height="70" align="right"></a></td></tr>
# </table>

# - improving modules
# - learned from others
# - tip of the iceberg
# 

# <table id=References style="width:100%">
# <tr><td><h1>References</h1></td><td><a href="#header"><img src="https://30percentclub.org/wp-content/uploads/2021/07/back-to-top-icon-01.svg" width="70" height="70" align="right"></a></td></tr>
# </table>

# Main:
# https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
# 
# 
# PCA:
# 
# https://www.kaggle.com/code/mgmarques/houses-prices-complete-solution
# 
# 
# https://www.kaggle.com/code/massquantity/all-you-need-is-pca-lb-0-11421-top-4
# 
# https://www.kaggle.com/code/willkoehrsen/introduction-to-feature-selection
# 
# preprocessing, , heatmap, PCA, submission:
# https://www.kaggle.com/code/ryanholbrook/feature-engineering-for-house-prices
# 
# Ensemble:
# https://www.kaggle.com/code/lavanyashukla01/how-i-made-top-0-3-on-a-kaggle-competition
# 
# 
# https://www.kaggle.com/code/pmarcelino/comprehensive-data-exploration-with-python
# 
# submission example: 
# https://www.kaggle.com/code/alexisbcook/titanic-tutorial/notebook
# 
# Missing Data refference from: 
# https://www.kaggle.com/code/serigne/stacked-regressions-top-4-on-leaderboard
# 
# heatmap:
# https://stackoverflow.com/questions/66171071/how-to-restrict-a-correlation-heatmap-to-interesting-cells-and-add-stars-to-mark
# 
# https://stackoverflow.com/questions/71350386/how-to-change-the-font-labels-of-heatmap
# 
# outliers:
# https://medium.com/analytics-vidhya/removing-outliers-understanding-how-and-what-behind-the-magic-18a78ab480ff
# 
# https://chat.openai.com/chat
# 
# https://www.geeksforgeeks.org/how-to-set-the-spacing-between-subplots-in-matplotlib-in-python/
# 
# https://realpython.com/python-enumerate/
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html
# 
# https://matplotlib.org/stable/gallery/color/named_colors.html
# 
# https://stackoverflow.com/questions/9074996/how-to-annotate-point-on-a-scatter-automatically-placed-arrow
# 
# https://www.w3schools.com/
# 
# 
# https://unicode.org/emoji/charts/full-emoji-list.html

# In[ ]:




